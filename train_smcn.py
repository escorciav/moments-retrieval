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

from didemo import DidemoSMCN
from model import SMCN
from loss import IntraInterMarginLoss
from evaluation import video_evaluation
from utils import Multimeter, ship_to
from utils import get_git_revision_hash

RAW_PATH = Path('data/raw')
MODALITY = ['rgb', 'flow']
OPTIMIZER = ['sgd', 'sgd_caffe']
EVAL_BATCH_SIZE = 1
METRICS = ['iou', 'r@1', 'r@5']
TRACK = 'r@1'

RGB_FEAT_PATH = RAW_PATH / 'average_fc7.h5'
FLOW_FEAT_PATH = RAW_PATH / 'average_global_flow.h5'
TRAIN_LIST_PATH = RAW_PATH / 'train_data.json'
VAL_LIST_PATH = RAW_PATH / 'val_data.json'
TEST_LIST_PATH = RAW_PATH / 'test_data.json'

parser = argparse.ArgumentParser(description='SMCN training DiDeMo')
# Features
parser.add_argument('--feat', default='rgb', choices=MODALITY,
                    help='kind of modality')
parser.add_argument('--rgb-path', type=Path, default=RGB_FEAT_PATH,
                    help='HDF5-file with RGB features')
# Model features
parser.add_argument('--no-loc', action='store_false', dest='loc',
                    help='Remove TEF features')
parser.add_argument('--no-context', action='store_false', dest='context',
                    help='Remove global video representation')
# Model
parser.add_argument('--margin', type=float, default=0.1,
                    help='MaxMargin margin value')
parser.add_argument('--w-inter', type=float, default=0.2,
                    help='Inter-loss weight')
parser.add_argument('--w-intra', type=float, default=0.5,
                    help='Intra-loss weight')
parser.add_argument('--original-setup', action='store_true',
                    help='Enable original optimization policy')
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
parser.add_argument('--no-shuffle', action='store_false', dest='shuffle',
                    help='Disable suffle dataset after each epoch')
# Logging
parser.add_argument('--logfile', default='',
                    help='Logging file')
parser.add_argument('--n-display', type=int, default=15,
                    help='Information display frequence')
# Hyper-parameter search
parser.add_argument('--hps', action='store_true',
                    help='Enable use of hps.yaml in folder of logfile')
# Reproducibility
parser.add_argument('--seed', type=int, default=1701,
                    help='random seed (-1 := random)')
# Debug
parser.add_argument('--debug', action='store_true')
parser.add_argument('--per-sample', action='store_true')

args = parser.parse_args()

def main(args):
    setup_logging(args)
    setup_hyperparameters(args)
    setup_rng(args)

    args.device = device_name = 'cpu'
    if args.gpu_id >= 0 and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_id}')
        device_name = torch.cuda.get_device_name(args.gpu_id)
    logging.info('Git revision hash: ' + get_git_revision_hash())
    logging.info('Launching training')
    logging.info(args)
    logging.info(f'Device: {device_name}')

    if args.feat == 'rgb':
        cues = {'rgb': {'file': args.rgb_path}}
    elif args.feat == 'flow':
        cues = {'flow': {'file': FLOW_FEAT_PATH}}

    logging.info('Pre-loading features... This may take a couple of minutes.')
    train_dataset = DidemoSMCN(TRAIN_LIST_PATH, cues=cues,
                               context=args.context, loc=args.loc)
    val_dataset = DidemoSMCN(VAL_LIST_PATH, cues=cues, test=True,
                             context=args.context, loc=args.loc)
    test_dataset = DidemoSMCN(TEST_LIST_PATH, cues=cues, test=True,
                              context=args.context, loc=args.loc)

    # Setup data loaders
    logging.info('Setting-up loaders')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=args.shuffle,
                              num_workers=args.num_workers,
                              collate_fn=train_dataset.collate_data)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=val_dataset.collate_test_data)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=val_dataset.collate_test_data)

    net, ranking_loss, optimizer = setup_model(args, train_dataset)
    lr_schedule = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_step, args.lr_decay)

    logging.info('Starting optimization...')
    best_result = 0.0
    performance_test = {i: best_result for i in METRICS}
    patience = 0
    performance_per_sample = []
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
            if args.per_sample:
                performance_per_sample = validation(
                    args, net, None, val_loader, per_sample=True)
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_result': best_result,
                'optimizer' : optimizer.state_dict(),
                })
        else:
            patience += 1

        if patience == args.patience:
            break
    args.epochs = epoch + 1
    if args.patience == -1:
        performance_test = validation(args, net, None, test_loader)
        if args.per_sample:
            performance_per_sample = validation(
                args, net, None, val_loader, per_sample=True)
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_result': best_result,
            'optimizer' : optimizer.state_dict(),
            })

    logging.info(f'Best val r@1: {best_result:.4f}')
    dumping_arguments(args, performance_val, performance_test,
                      performance_per_sample)


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
        # measure elapsed time
        data_time = time.time() - end
        end = time.time()

        compared_embeddings = net(*minibatch[2:])
        loss, _, _ = criterion(*compared_embeddings)
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


def validation(args, net, criterion, loader, per_sample=False):
    time_meters = Multimeter(keys=['Batch', 'Eval'])
    meters = Multimeter(keys=METRICS)
    dataset = loader.dataset
    results_per_sample = []

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
            if per_sample:
                annotation_id = dataset.metadata[it]['annotation_id']
                results_per_sample.append([annotation_id] + performance_i)
    logging.info(f'{time_meters.report()}\t{meters.report()}')

    if per_sample:
        return results_per_sample
    performance = meters.dump()
    return performance


def dumping_arguments(args, val_performance, test_performance,
                      performance_per_sample):
    if len(args.logfile) == 0:
        return
    result_file = args.logfile + '.json'
    device = args.device
    # Update dict with performance and remove non-serializable stuff
    args.device = None
    args.rgb_path = str(args.rgb_path)
    args_dict = vars(args)
    args_dict.update({f'val_{k}': v for k, v in val_performance.items()})
    args_dict.update({f'test_{k}': v for k, v in test_performance.items()})
    with open(result_file, 'w') as f:
        json.dump(args_dict, f)
    if args.per_sample:
        with open(args.logfile + '.csv', 'x') as fid:
            fid.write('{},{},{},{}\n'.format('annotation_id', *METRICS))
            status = [fid.write('{},{},{},{}\n'.format(*i))
                      for i in performance_per_sample]
    args.device = device


def save_checkpoint(args, state):
    if len(args.logfile) == 0:
        return
    torch.save(state, args.logfile + '_checkpoint.pth.tar')


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
                     max_length=dataset.max_words)
    net = SMCN(**mcn_setup)
    opt_parameters = net.optimization_parameters(
        args.lr, args.original_setup)
    ranking_loss = IntraInterMarginLoss(
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