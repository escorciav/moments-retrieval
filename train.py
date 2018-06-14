import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from didemo import Didemo
from model import MCN, TripletMEE
from loss import IntraInterTripletMarginLoss, IntraInterMarginLoss
from evaluation import video_evaluation
from utils import Multimeter, ship_to

RAW_PATH = Path('data/raw')
MODALITY = ['rgb', 'flow', 'all']
ARCHITECTURES = ['mcn', 'tmee']
OPTIMIZER = ['sgd', 'adam']
EVAL_BATCH_SIZE = 1
PATIENCE_LIMIT = 10

parser = argparse.ArgumentParser(description='DiDeMo')

parser.add_argument('--lr', type=float, default=0.05,
                            help='initial learning rate')
parser.add_argument('--epochs', type=int, default=100,
                            help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                            help='batch size')
parser.add_argument('--margin', type=float, default=0.1,
                            help='MaxMargin margin value')
parser.add_argument('--lr_decay', type=float, default=0.95,
                            help='Learning rate decay')
parser.add_argument('--lr_step', type=float, default=30,
                            help='Learning rate epoch to decay')
parser.add_argument('--n_display', type=int, default=60,
                            help='Information display frequence')

parser.add_argument('--gpu-id', type=int, default=-1,
                    help='Use of GPU')
parser.add_argument('--n-cpu', type=int, default=4,
                    help='Number of CPU')

parser.add_argument('--feat', default='rgb', choices=MODALITY,
                    help='kind of modality')
parser.add_argument('--arch', default='mcn', choices=ARCHITECTURES,
                    help='Type of architecture')

# TODO: make it a directory
parser.add_argument('--logfile', default='',
                    help='Logging file')
# TODO: enable
# parser.add_argument('--text_cluster_size', type=int, default=32,
#                             help='Text cluster size')
# parser.add_argument('--seed', type=int, default=1,
#                     help='Initial Random Seed')

parser.add_argument('--optimizer', type=str, default='sgd', choices=OPTIMIZER,
                    help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Nesterov Momentum for SGD')
parser.add_argument('--clip-grad', type=float, default=10,
                    help='clip gradients')

# Added for debugging
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed (-1 := random)')

args = parser.parse_args()
# hyper-parameter search different learning rate
LRS = [10, 15, 30, 45]
if args.optimizer == 'sgd':
    LR = [1e-1, 1e-2, 1e-3]
    LRD = [0.1, 0.5, 0.75]
else:
    LR = [1e-2, 5e-4, 1e-4, 5e-5]
    LRD = [0.9, 0.95]
random.shuffle(LRS)
random.shuffle(LR)
random.shuffle(LRD)
args.lr = LR[0]
args.lr_decay = LRD[0]
args.lr_step = LRS[0]
if args.arch == 'mcn':
    args.optimizer = 'sgd'
else:
    random.shuffle(OPTIMIZER)
    args.optimizer = OPTIMIZER[0]


def main(args):
    setup_rng(args)
    setup_logging(args)

    args.device = 'cpu'
    if args.gpu_id >= 0 and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_id}')
    logging.info('Launching training')
    logging.info(args)

    rgb_feat_path = RAW_PATH / 'average_fc7.h5'
    flow_feat_path = RAW_PATH / 'average_global_flow.h5'
    train_list_path = RAW_PATH / 'train_data.json'
    val_list_path = RAW_PATH / 'val_data.json'

    if args.feat == 'rgb':
        cues = {'rgb': {'file': rgb_feat_path}}
    elif args.feat == 'flow':
        cues = {'flow': {'file': flow_feat_path}}
    else:
        cues = {'rgb': {'file': rgb_feat_path},
                'flow': {'file': flow_feat_path}}

    logging.info('Pre-loading features... This may take a couple of minutes.')
    train_dataset = Didemo(train_list_path, cues=cues)
                           # loc=True, max_words=50)
    val_dataset = Didemo(val_list_path, cues=cues, test=True)

    # Setup data loaders
    logging.info('Setting-up loaders')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.n_cpu,
                                  collate_fn=train_dataset.collate_data)
    val_dataloader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE,
                                shuffle=False, num_workers=args.n_cpu,
                                collate_fn=val_dataset.collate_test_data)

    net, ranking_loss, optimizer = setup_model(args, train_dataset)
    lr_schedule = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_step, args.lr_decay)

    logging.info('Starting optimization...')
    best_r1 = 0.0
    patience = 0
    for epoch in range(args.epochs):
        lr_schedule.step()
        train_epoch(args, net, ranking_loss, train_dataloader, optimizer, epoch)
        r1_i = validation(args, net, None, val_dataloader)

        if r1_i > best_r1:
            patience = 0
            best_r1 = r1_i
            logging.info(f'Hit jackpot r@1: {best_r1:.4f}')
            # TODO get results in testing
        else:
            patience += 1

        if patience == PATIENCE_LIMIT:
            break
    logging.info(f'Best r@1: {best_r1:.4f}')
    dumping_arguments(args, best_r1)


def train_epoch(args, net, ranking_loss, loader, optimizer, epoch):
    logging.info(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    net.train()
    for it, minibatch in enumerate(loader):
        if args.gpu_id >= 0:
            minibatch_ = minibatch
            minibatch = ship_to(minibatch, args.device)

        embeddings = net(*minibatch)
        loss, _, _ = ranking_loss(*embeddings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (it + 1) % args.n_display == 0:
            logging.info(f'Epoch: {epoch + 1}, '
                  f'status: {it / len(loader):.2f}, '
                  f'loss: {running_loss / args.n_display:.4f}')
            running_loss = 0.0


def validation(args, net, ranking_loss, loader):
    logging.info(f'* Evaluation')
    dataset = loader.dataset
    meters = Multimeter(keys=['iou', 'r@1', 'r@5'])
    net.eval()
    for it, minibatch in enumerate(loader):
        if args.gpu_id >= 0:
            minibatch_ = minibatch
            minibatch = ship_to(minibatch, args.device)
        results, descending = net.predict(*minibatch)
        # TODO: port evaluation to GPU
        _, idx = results.sort(descending=descending)
        idx_h = idx.to('cpu')
        predictions = [dataset.segments[i] for i in idx_h]
        gt = dataset.metadata[it]['times']
        performance = video_evaluation(gt, predictions)
        meters.update(performance)
    logging.info(f'{meters.report()}')
    return meters.meters[1].avg


def dumping_arguments(args, r1):
    if len(args.logfile) == 0:
        return
    args.best_r1 = r1
    result_file = args.logfile + '.json'
    device = args.device
    with open(result_file, 'w') as f:
        args.device = None
        json.dump(vars(args), f)
    args.device = device


def setup_logging(args):
    log_prm = dict(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)
    if len(args.logfile) > 1:
        log_prm['filename'] = args.logfile + '.log'
        log_prm['filemode'] = 'w'
    logging.basicConfig(**log_prm)


def setup_model(args, dataset):
    # TODO clean the mess
    logging.info('Setting-up model and criterion')
    if args.arch == 'tmee':
        logging.info('Model: TripletMEE')

        # MESS: data specific stuff
        feat_0 = dataset[0]
        text_embedding = 1000
        crossmodal_embedding = 100
        visual_embedding = {k: (v.shape[0], crossmodal_embedding)
                            for k, v in feat_0[2].items()}
        assert visual_embedding.keys() == cues.keys()
        text_dim = feat_0[0].shape[1]
        max_length = feat_0[0].shape[0]
        extra = dict(word_size=text_dim, lstm_layers=args.rnn_layers,
                     max_length=max_length)

        net = TripletMEE(text_embedding, visual_embedding, **extra)
        opt_parameters = net.parameters()
        ranking_loss = IntraInterMarginLoss(margin=args.margin)
    else:
        logging.info('Model: MCN')

        # MESS: data specific stuff
        feat_0 = dataset[0]
        text_dim = feat_0[0].shape[1]
        video_modality_dim = feat_0[2][args.feat].shape[0]
        max_length = feat_0[0].shape[0]
        mcn_setup = dict(visual_size=video_modality_dim, lang_size=text_dim,
                        max_length=max_length)

        net = MCN(**mcn_setup)
        opt_parameters = net.optimization_parameters(args.lr)
        ranking_loss = IntraInterTripletMarginLoss(margin=args.margin)

    net.train()
    if args.gpu_id >= 0:
        logging.info('Transferring model and criterion to GPU')
        net.to(args.device)
        ranking_loss.to(args.device)

    # Optimizer
    logging.info(f'Setting-up optimizer: {args.optimizer}')
    if args.optimizer == 'adam':
        optimizer = optim.Adam(opt_parameters, lr=args.lr)
    elif args.optimizer == 'sgd':
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
        optimizer = optim.SGD(opt_parameters, lr=args.lr,
                              momentum=args.momentum)
    return net, ranking_loss, optimizer


def setup_rng(args):
    if args.seed < 1:
        args.seed = random.randint(0, 2**16)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


if __name__ == '__main__':
    main(args)