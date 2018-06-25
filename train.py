import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from optim import Nesterov
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
METRICS = ['iou', 'r@1', 'r@5']
TRACK = 'r@1'

RGB_FEAT_PATH = RAW_PATH / 'average_fc7.h5'
FLOW_FEAT_PATH = RAW_PATH / 'average_global_flow.h5'
TRAIN_LIST_PATH = RAW_PATH / 'train_data.json'
VAL_LIST_PATH = RAW_PATH / 'val_data.json'
TEST_LIST_PATH = RAW_PATH / 'test_data.json'

parser = argparse.ArgumentParser(description='MCN training DiDeMo')

# Features
parser.add_argument('--feat', default='rgb', choices=MODALITY,
                    help='kind of modality')
parser.add_argument('--arch', default='mcn', choices=ARCHITECTURES,
                    help='Type of architecture')

# MCN
parser.add_argument('--margin', type=float, default=0.1,
                    help='MaxMargin margin value')
parser.add_argument('--w-inter', type=float, default=0.2,
                    help='Inter-loss weight')
parser.add_argument('--w-intra', type=float, default=0.5,
                    help='Intra-loss weight')
# TODO: enable
# parser.add_argument('--text_cluster_size', type=int, default=32,
#                     help='Text cluster size')

# Device specific
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--gpu-id', type=int, default=-1,
                    help='Use of GPU')
parser.add_argument('--n-cpu', type=int, default=4,
                    help='Number of CPU')

# Optimization
parser.add_argument('--lr', type=float, default=0.05,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=107,
                    help='upper epoch limit')
parser.add_argument('--optimizer', type=str, default='sgd', choices=OPTIMIZER,
                    help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Nesterov Momentum for SGD')
parser.add_argument('--lr-decay', type=float, default=0.95,
                    help='Learning rate decay')
parser.add_argument('--lr-step', type=float, default=30,
                    help='Learning rate epoch to decay')
parser.add_argument('--clip-grad', type=float, default=10,
                    help='clip gradients')
parser.add_argument('--patience', type=int, default=10,
                    help='stop optimization if not improvements')
# Logging
parser.add_argument('--logfile', default='',
                    help='Logging file')
parser.add_argument('--n-display', type=int, default=60,
                    help='Information display frequence')

# Reproducibility
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed (-1 := random)')

args = parser.parse_args()

# hyper-parameter search different learning rate
# TODO: move to yaml
PATIENCE = [-1, 12, 24]
MARGIN = [0.1, 0.2, 0.5]
LW_INTER_INTRA = [(0.2, 0.5), (0.5, 0.5), (0.1, 0.5)]
LRS = [12, 36, 72]
MOMENTUM = [0.9, 0.95]
CAFFE_SETUP = [True, False]
if args.arch == 'mcn':
    OPTIMIZER = ['sgd', 'nesterov']
    random.shuffle(OPTIMIZER)
    args.optimizer = OPTIMIZER[0]
else:
    random.shuffle(OPTIMIZER)
    args.optimizer = OPTIMIZER[0]
if args.optimizer == 'sgd':
    LR = [1e-1, 1e-2, 1e-3]
    LRD = [0.1, 0.5, 0.5, 0.75, 0.75]
elif args.optimizer == 'nesterov':
    LR = [1e-1, 1e-2, 5e-3]
    LRD = [0.1, 0.5, 0.75]
else:
    LR = [1e-2, 5e-4, 1e-4, 5e-5]
    LRD = [0.9, 0.95]
random.shuffle(LRS)
args.lr_step = LRS[0]
random.shuffle(LR)
args.lr = LR[0]
random.shuffle(LRD)
args.lr_decay = LRD[0]
random.shuffle(MOMENTUM)
args.patience = MOMENTUM[0]
random.shuffle(PATIENCE)
args.patience = PATIENCE[0]
if args.patience == -1:
    args.lr_step = 36
random.shuffle(MARGIN)
args.margin = MARGIN[0]
random.shuffle(LW_INTER_INTRA)
args.w_inter, args.w_intra = LW_INTER_INTRA[0]
random.shuffle(CAFFE_SETUP)
args.caffe_setup = CAFFE_SETUP[0]

def main(args):
    setup_rng(args)
    setup_logging(args)

    args.device = 'cpu'
    if args.gpu_id >= 0 and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_id}')
    logging.info('Launching training')
    logging.info(args)

    if args.feat == 'rgb':
        cues = {'rgb': {'file': RGB_FEAT_PATH}}
    elif args.feat == 'flow':
        cues = {'flow': {'file': FLOW_FEAT_PATH}}
    else:
        raise NotImplementedError
        cues = {'rgb': {'file': RGB_FEAT_PATH},
                'flow': {'file': flow_feat_path}}

    logging.info('Pre-loading features... This may take a couple of minutes.')
    train_dataset = Didemo(train_list_path, cues=cues)
    val_dataset = Didemo(val_list_path, cues=cues, test=True)
    test_dataset = Didemo(test_list_path, cues=cues, test=True)

    # Setup data loaders
    logging.info('Setting-up loaders')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.n_cpu,
                              collate_fn=train_dataset.collate_data)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE,
                            shuffle=False, num_workers=args.n_cpu,
                            collate_fn=val_dataset.collate_test_data)
    test_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE,
                             shuffle=False, num_workers=args.n_cpu,
                             collate_fn=val_dataset.collate_test_data)

    net, ranking_loss, optimizer = setup_model(args, train_dataset)
    lr_schedule = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_step, args.lr_decay)

    logging.info('Starting optimization...')
    best_result = 0.0
    performance_test = {i: best_result for i in METRICS}
    patience = 0
    for epoch in range(args.epochs):
        lr_schedule.step()
        train_epoch(args, net, ranking_loss, train_loader, optimizer, epoch)
        performance_val = validation(args, net, None, val_loader)
        val_result = performance_val[TRACK]

        if val_result > best_result:
            patience = 0
            best_result = val_result
            logging.info(f'Hit jackpot {TRACK}: {best_result:.4f}')
            performance_test = validation(args, net, None, test_loader)
        else:
            patience += 1

        if patience == args.patience:
            break
    args.epochs = epoch + 1
    if args.patience == -1:
        performance_test = validation(args, net, None, test_loader)

    logging.info(f'Best val r@1: {best_result:.4f}')
    dumping_arguments(args, performance_val, performance_test)


def train_epoch(args, net, ranking_loss, loader, optimizer, epoch):
    time_meters = Multimeter(keys=['Data', 'Batch'])
    running_loss = 0.0

    logging.info(f'Epoch: {epoch + 1}')
    net.train()
    end = time.time()
    for it, minibatch in enumerate(loader):
        if args.gpu_id >= 0:
            minibatch_ = minibatch
            minibatch = ship_to(minibatch, args.device)
        # measure elapsed time
        data_time = time.time() - end
        end = time.time()

        embeddings = net(*minibatch)
        loss, _, _ = ranking_loss(*embeddings)
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
        optimizer.step()
        time_meters.update([data_time, time.time() - end])
        end = time.time()

        running_loss += loss.item()
        if (it + 1) % args.n_display == 0:
            logging.info(f'Epoch: [{epoch + 1}]'
                         f'[{100 * it / len(loader):.2f}]\t'
                         f'{time_meters.report()}\t'
                         f'Loss {running_loss / args.n_display:.4f}')
            running_loss = 0.0


def validation(args, net, ranking_loss, loader):
    time_meters = Multimeter(keys=['Batch', 'Eval'])
    meters = Multimeter(keys=METRICS)
    dataset = loader.dataset

    logging.info(f'* Evaluation')
    net.eval()
    with torch.no_grad():
        end = time.time()
        for it, minibatch in enumerate(loader):
            if args.gpu_id >= 0:
                minibatch_ = minibatch
                minibatch = ship_to(minibatch, args.device)
            results, descending = net.predict(*minibatch)
            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            # TODO: port evaluation to GPU
            _, idx = results.sort(descending=descending)
            idx_h = idx.to('cpu')
            predictions = [dataset.segments[i] for i in idx_h]
            gt = dataset.metadata[it]['times']
            performance_i = video_evaluation(gt, predictions)
            meters.update(performance_i)
            time_meters.update([batch_time, time.time() - end])
            end = time.time()
    logging.info(f'{time_meters.report()}\t{meters.report()}')
    performance = meters.dump()
    return performance


def dumping_arguments(args, val_performance, test_performance):
    if len(args.logfile) == 0:
        return
    result_file = args.logfile + '.json'
    device = args.device
    # Update dict with performance and remove non-serializable stuff
    args.device = None
    args_dict = vars(args)
    args_dict.update({f'val_{k}': v for k, v in val_performance.items()})
    args_dict.update({f'test_{k}': v for k, v in test_performance.items()})
    with open(result_file, 'w') as f:
        json.dump(args_dict, f)
    args.device = device


def setup_logging(args):
    log_prm = dict(format='%(asctime)s:%(levelname)s:%(message)s',
                   level=logging.DEBUG)
    if len(args.logfile) > 1:
        log_prm['filename'] = args.logfile + '.log'
        log_prm['filemode'] = 'w'
    logging.basicConfig(**log_prm)


def setup_model(args, dataset):
    # TODO clean the mess
    logging.info('Setting-up model and criterion')
    if args.arch == 'tmee':
        raise NotImplementedError
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
        ranking_loss = IntraInterMarginLoss(
            margin=args.margin, w_inter=args.w_inter,
            w_intra=args.w_intra)
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
        opt_parameters = net.optimization_parameters(
            args.lr, args.caffe_setup)
        ranking_loss = IntraInterTripletMarginLoss(
            margin=args.margin, w_inter=args.w_inter,
            w_intra=args.w_intra)

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
        optimizer = optim.SGD(opt_parameters, lr=args.lr,
                              momentum=args.momentum)
    elif args.optimizer == 'nesterov':
        optimizer = Nesterov(opt_parameters, lr=args.lr,
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