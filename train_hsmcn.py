import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from optim import SGDCaffe
import torch.optim as optim
from torch.utils.data import DataLoader

import didemo
from didemo import DidemoSMCNHeterogeneous, DidemoSMCN
from model import SMCN
from loss import IntraInterMarginLoss
from evaluation import video_evaluation
from utils import Multimeter, MutableSampler, ship_to

DIDEMO_RAW_PATH = Path('data/raw')
TRAIN_PATH = Path('data/interim/didemo_yfcc100m/')
MODALITY = ['rgb']
OPTIMIZER = ['sgd', 'sgd_caffe']
EVAL_BATCH_SIZE = 1
METRICS = ['iou', 'r@1', 'r@5']
TRACK = 'r@1'

RGB_FEAT_PATH = Path('data/interim/didemo_yfcc100m/')
TRAIN_LIST_PATH = TRAIN_PATH / 'train_data.json'
VAL_LIST_PATH = DIDEMO_RAW_PATH / 'val_data.json'
TEST_LIST_PATH = DIDEMO_RAW_PATH / 'test_data.json'

parser = argparse.ArgumentParser(description='SMCN-Htg training DiDeMo')
# Training data
parser.add_argument('--train-list', type=Path, default=TRAIN_LIST_PATH,
                    help='JSON-file with training data')
# Features
parser.add_argument('--feat', default='rgb', choices=MODALITY,
                    help='kind of modality')
parser.add_argument('--rgb-path', type=Path, default=[RGB_FEAT_PATH],
                    nargs='+',
                    help='HDF5-files with RGB features')
# Model features
parser.add_argument('--no-loc', action='store_false', dest='loc',
                    help='Remove TEF features')
parser.add_argument('--no-context', action='store_false', dest='context',
                    help='Remove global video representation')
# Model-specific hyper-parameters
parser.add_argument('--visual-hidden', type=int, default=500,
                    help='Hidden unit in MLP visual stream')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate in visual stream')
parser.add_argument('--embedding-size', type=int, default=100,
                    help='Dimensionaity of cross-modal embedding')
parser.add_argument('--lang-hidden', type=int, default=1000,
                    help='Dimensionaity of cross-modal embedding')
# Model-specific optimizer
parser.add_argument('--margin', type=float, default=0.1,
                    help='MaxMargin margin value')
parser.add_argument('--w-inter', type=float, default=0.2,
                    help='Inter-loss weight')
parser.add_argument('--w-intra', type=float, default=0.5,
                    help='Intra-loss weight')
parser.add_argument('--sw-intra', type=float, default=[1.0, 0.0], nargs=2,
                    help='Source-weight on Intra-loss')
parser.add_argument('--original-setup', action='store_true',
                    help='Enable original optimization policy')
# Sampling multiple data sources (video and images)
parser.add_argument('--sampling-scheme', default='skip',
                    choices=didemo.SAMPLING_SCHEMES,
                    help='Sampling scheme to mix video and image data')
# TODO: I should not do that as other researchers also should write clean and
# easy to read and run code. touche ;P
parser.add_argument('--sampler-kwargs', default=dict(epoch=10), type=eval,
                    help='key-value pair to conrol sampling scheme')
# Device specific
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--gpu-id', type=int, default=-1,
                    help='Use of GPU')
parser.add_argument('--num-workers', type=int, default=6,
                    help='Number of processes')
# Optimization
parser.add_argument('--lr', type=float, default=0.05,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=108,
                    help='upper epoch limit')
parser.add_argument('--optimizer', type=str, default='sgd_caffe',
                    choices=OPTIMIZER,
                    help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Momentum for SGD')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='Learning rate decay')
parser.add_argument('--lr-step', type=float, default=30,
                    help='Learning rate epoch to decay')
parser.add_argument('--clip-grad', type=float, default=10,
                    help='clip gradients')
parser.add_argument('--patience', type=int, default=-1,
                    help='stop optimization if not improvements')
# Logging
parser.add_argument('--logfile', default='',
                    help='Logging file')
parser.add_argument('--n-display', type=int, default=300,
                    help='Information display frequence')
# Hyper-parameter search
parser.add_argument('--hps', action='store_true',
                    help='Enable use of hps.yaml in folder of logfile')
# Reproducibility
parser.add_argument('--seed', type=int, default=1701,
                    help='random seed (-1 := random)')
# Debug
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

def main(args):
    setup_logging(args)
    setup_hyperparameters(args)
    setup_rng(args)

    args.device = device_name = 'cpu'
    if args.gpu_id >= 0 and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_id}')
        device_name = torch.cuda.get_device_name(args.gpu_id)
    logging.info('Launching training')
    logging.info(args)
    logging.info(f'Device: {device_name}')

    # TODO: clean mess with cues
    cues = {'rgb': {'file': args.rgb_path}}
    cues_val_test = {'rgb': {'file': args.rgb_path[0]}}
    logging.info('Pre-loading features... This may take a couple of minutes.')
    train_dataset = DidemoSMCNHeterogeneous(
        args.train_list, cues=cues, context=args.context, loc=args.loc,
        sampling_scheme=args.sampling_scheme,
        sampler_kwargs=args.sampler_kwargs, DEBUG=args.debug)
    val_dataset = DidemoSMCN(VAL_LIST_PATH, cues=cues_val_test, test=True,
                             context=args.context, loc=args.loc)
    test_dataset = DidemoSMCN(TEST_LIST_PATH, cues=cues_val_test, test=True,
                              context=args.context, loc=args.loc)

    # Setup data loaders
    logging.info('Setting-up loaders')
    sampler = MutableSampler(num_instances=len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=train_dataset.collate_data,
                              sampler=sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=val_dataset.collate_test_data,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=val_dataset.collate_test_data,
                             pin_memory=True)

    net, ranking_loss, optimizer = setup_model(args, train_dataset)
    lr_schedule = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_step, args.lr_decay)

    logging.info('Starting optimization...')
    # on training start
    # TODO update next line when resuming from checkpoint
    train_dataset.update_sampler(0, sampler)
    best_result = 0.0
    performance_test = {i: best_result for i in METRICS}
    patience = 0
    for epoch in range(args.epochs):
        # on epoch begin
        lr_schedule.step()

        train_epoch(args, net, ranking_loss, train_loader, optimizer, epoch)

        # on epoch end
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
        train_dataset.update_sampler(epoch + 1, sampler)
    args.epochs = epoch + 1
    if args.patience == -1:
        performance_test = validation(args, net, None, test_loader)

    logging.info(f'Best val r@1: {best_result:.4f}')
    dumping_arguments(args, performance_val, performance_test)


def train_epoch(args, net, criterion, loader, optimizer, epoch):
    time_meters = Multimeter(keys=['Data', 'Batch'])
    running_loss = 0.0

    logging.info(f'Epoch: {epoch + 1}')
    net.train()
    end = time.time()
    for it, minibatch in enumerate(loader):
        if args.gpu_id >= 0:
            minibatch_ = minibatch
            minibatch = ship_to(minibatch, args.device)
        source_ids = minibatch[1]
        # measure elapsed time
        data_time = time.time() - end
        end = time.time()

        compared_embeddings = net(*minibatch[2:])
        iw_intra = args.sw_intra[source_ids]
        loss, _, _ = criterion(*compared_embeddings, iw_intra=iw_intra)
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


def validation(args, net, criterion, loader):
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
            results, descending = net.predict(*minibatch[2:])
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
    args.rgb_path = [str(i) for i in args.rgb_path]
    args.train_list = str(args.train_list)
    args.sw_intra = args.sw_intra.tolist()
    args_dict = vars(args)
    args_dict.update({f'val_{k}': v for k, v in val_performance.items()})
    args_dict.update({f'test_{k}': v for k, v in test_performance.items()})
    with open(result_file, 'w') as f:
        json.dump(args_dict, f)
    args.device = device


def setup_hyperparameters(args):
    if not args.hps:
        return
    filename = Path(args.logfile).parent / 'hps.yaml'
    if not filename.exists():
        logging.error(f'Ignoring HPS. Not found {filename}')
        return
    with open(filename, 'r') as fid:
        config = yaml.load(fid)
    logging.info('Proceeding to perform random HPS')
    args_dview = vars(args)
    for k, v in config.items():
        if not isinstance(v, list):
            args_dview[k] = v
            continue
        random.shuffle(v)
        args_dview[k] = v[0]


def setup_logging(args):
    log_prm = dict(format='%(asctime)s:%(levelname)s:%(message)s',
                   level=logging.DEBUG)
    if len(args.logfile) > 1:
        log_prm['filename'] = args.logfile + '.log'
        log_prm['filemode'] = 'w'
    logging.basicConfig(**log_prm)


def setup_model(args, dataset):
    logging.info('Model: SMCN')
    mcn_setup = dict(visual_size=dataset.visual_size[args.feat],
                     lang_size=dataset.language_size,
                     max_length=dataset.max_words,
                     embedding_size=args.embedding_size,
                     dropout=args.dropout,
                     visual_hidden=args.visual_hidden,
                     lang_hidden=args.lang_hidden)
    net = SMCN(**mcn_setup)
    opt_parameters = net.optimization_parameters(
        args.lr, args.original_setup)
    ranking_loss = IntraInterMarginLoss(
        margin=args.margin, w_inter=args.w_inter,
        w_intra=args.w_intra)
    args.sw_intra = torch.tensor(args.sw_intra)

    net.train()
    if args.gpu_id >= 0:
        logging.info('Transferring model and criterion to GPU')
        net.to(args.device)
        ranking_loss.to(args.device)
        args.sw_intra = args.sw_intra.to(args.device)

    # Optimizer
    logging.info(f'Setting-up optimizer: {args.optimizer}')
    if args.optimizer == 'adam':
        optimizer = optim.Adam(opt_parameters, lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(opt_parameters, lr=args.lr,
                              momentum=args.momentum)
    elif args.optimizer == 'sgd_caffe':
        optimizer = SGDCaffe(opt_parameters, lr=args.lr,
                             momentum=args.momentum)
    else:
        raise ValueError(f'Unknow optimizer {args.optimizer}')
    return net, ranking_loss, optimizer


def setup_rng(args):
    if args.seed < 1:
        args.seed = random.randint(0, 2**16)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


if __name__ == '__main__':
    main(args)