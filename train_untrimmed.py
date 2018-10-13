"Train *MCN in untrimmed data"
import argparse
import logging
import time
from itertools import product
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from optim import SGDCaffe
import torch.optim as optim

import dataset_untrimmed
import model
import proposals
from loss import IntraInterMarginLoss
from evaluation import single_moment_retrieval
from utils import Multimeter
from utils import collate_data, collate_data_eval, ship_to
from utils import setup_logging, setup_rng
from utils import dumping_arguments, save_checkpoint
from utils import get_git_revision_hash

OPTIMIZER = ['sgd', 'sgd_caffe']
EVAL_BATCH_SIZE = 1
TOPK_IOU_POINTS = tuple(product((1, 5), (0.5, 0.7)))
METRICS = [f'r@{k},{iou}' for k, iou in TOPK_IOU_POINTS]
TRACK = 'r@1,0.7'

parser = argparse.ArgumentParser(description='*MCN training')
# Data
parser.add_argument('--train-list', type=Path, default='non-existent',
                    help='JSON-file with training instances')
parser.add_argument('--val-list', type=Path, default='non-existent',
                    help='JSON-file with training instances')
parser.add_argument('--test-list', type=Path, default='non-existent',
                    help='JSON-file with training instances')
# Features
parser.add_argument('--feat', default='rgb',
                    help='Record the type of feature used (modality)')
parser.add_argument('--h5-path', type=Path, default='non-existent',
                    required=True, help='HDF5-file with features')
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
# Hyper-parameters to explore search space (inference)
parser.add_argument('--min-length', type=float, default=3,
                    help='Minimum length of slidding windows (seconds)')
parser.add_argument('--num-scales', type=int, default=8,
                    help='Number of scales in a multi-scale linear slidding '
                         'window')
parser.add_argument('--stride', type=float, default=3,
                    help='stride of the slidding window (seconds)')
# Device specific
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--gpu-id', type=int, default=-1, help='GPU device')
parser.add_argument('--num-workers', type=int, default=6,
                    help='Number of processes')
# Optimization
parser.add_argument('--lr', type=float, default=0.05,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=108,
                    help='upper epoch limit')
parser.add_argument('--optimizer', type=str, default='sgd_caffe',
                    choices=OPTIMIZER, help='type of optimizer')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Momentum for SGD')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='Learning rate decay')
parser.add_argument('--lr-step', type=float, default=30,
                    help='Learning rate epoch to decay')
parser.add_argument('--clip-grad', type=float, default=10,
                    help='clip gradients')
parser.add_argument('--patience', type=int, default=-1,
                    help='stop optimization if there is no improvements')
parser.add_argument('--no-shuffle', action='store_false', dest='shuffle',
                    help='Disable suffle dataset after each epoch')
# Logging
parser.add_argument('--logfile', default='', help='Logging file')
parser.add_argument('--n-display', type=int, default=15,
                    help='Information display frequence')
parser.add_argument('--augmented-results', action='store_true',
                    help='Dump results to assess model')
parser.add_argument('--not-serialize', action='store_false', dest='serialize',
                    help='Avoid dumping .pth.tar with model parameters')
# Reproducibility
parser.add_argument('--seed', type=int, default=1701,
                    help='random seed (-1 := random)')
# Debug
parser.add_argument('--debug', action='store_true',
                    help=('yield incorrect results! only to verify things '
                          '(loaders, ops) can run well'))
# TODO: HPS

args = parser.parse_args()


def main(args):
    setup_logging(args)
    setup_rng(args)

    args.device = device_name = 'cpu'
    if args.gpu_id >= 0 and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_id}')
        device_name = torch.cuda.get_device_name(args.gpu_id)
    logging.info(f'Git revision hash:  {get_git_revision_hash()}')
    logging.info('*MCN in untrimmed videos')
    logging.info(args)
    logging.info(f'Device: {device_name}')

    train_loader, val_loader, test_loader = setup_dataset(args)
    net, ranking_loss, optimizer = setup_model(args, train_loader.dataset)
    lr_schedule = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_step, args.lr_decay)

    logging.info('Starting optimization...')
    best_result = 0.0
    performance_test = None
    patience = 0
    for epoch in range(args.epochs):
        lr_schedule.step()
        train_epoch(args, net, ranking_loss, train_loader, optimizer, epoch)
        performance_val = validation(args, net, val_loader)
        val_result = performance_val[TRACK]

        if val_result > best_result:
            patience = 0
            best_result = val_result
            logging.info(f'Hit jackpot {TRACK}: {best_result:.4f}')
            performance_test = check_testing(args, net, test_loader)
            save_checkpoint(args, {'epoch': epoch + 1,
                                   'state_dict': net.state_dict(),
                                   'best_result': best_result})
        else:
            patience += 1

        if patience == args.patience:
            break
    args.epochs = epoch + 1
    if args.patience == -1:
        performance_test = check_testing(args, net, test_loader)
        save_checkpoint(args, {'epoch': epoch + 1,
                               'state_dict': net.state_dict(),
                               'best_result': best_result})

    logging.info(f'Best val {TRACK}: {best_result:.4f}')
    # TODO: update to dump
    dumping_arguments(args, performance_val, performance_test)


def train_epoch(args, net, criterion, loader, optimizer, epoch):
    "Update model making a whole pass over dataset in loader"
    ind = 2 if args.debug else 0
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

        compared_embeddings = net(*minibatch[ind:])
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


def validation(args, net, loader):
    "Eval model"
    ind = 2 if args.debug else 0
    time_meters = Multimeter(keys=['Batch', 'Eval'])
    meters = Multimeter(keys=METRICS)
    logging.info(f'* Evaluation')
    net.eval()
    with torch.no_grad():
        end = time.time()
        for it, minibatch in enumerate(loader):
            if args.gpu_id >= 0:
                minibatch_ = minibatch
                minibatch = ship_to(minibatch, args.device)
            results, descending = net.predict(*minibatch[ind:-2])
            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            _, idx = results.sort(descending=descending)
            gt_segment, segments = minibatch[-2:]
            sorted_segments = segments[idx, :]
            # Note: these next two lines look a bit slower and stupid, #sorry
            hit_k_iou = single_moment_retrieval(
                gt_segment, sorted_segments, TOPK_IOU_POINTS)
            meters.update([i.item() for i in hit_k_iou])
            time_meters.update([batch_time, time.time() - end])
            end = time.time()
    logging.info(f'{time_meters.report()}\t{meters.report()}')
    performance = meters.dump()
    return performance


def check_testing(args, net, loader):
    """Lazy function to make comparsion = to json parsing.

    Note: Please do not fool yourself picking model based on testing
    """
    if loader is None:
        return None
    return validation(args, net, loader)


def setup_dataset(args):
    "Setup dataset and loader"
    subset_files = [('train', args.train_list), ('val', args.val_list),
                    ('test', args.test_list)]
    cues = {args.feat: {'file': args.h5_path}}
    proposal_generator = proposals.SlidingWindowMSFS(
        args.min_length, args.num_scales, args.stride, unique=True)
    extras_dataset_configs = [
        # Training
        {'debug': args.debug},
        # Validation or Testing
        {'eval': True, 'proposals_interface': proposal_generator,
         'debug': args.debug}
    ]
    extras_loaders_configs = [
        # Training
        {'shuffle': args.shuffle, 'collate_fn': collate_data,
         'batch_size': args.batch_size},
        # Validation or Testing
        {'shuffle': False, 'collate_fn': collate_data_eval,
         'batch_size': EVAL_BATCH_SIZE}
    ]

    logging.info('Setting-up datasets and loaders')
    logging.info('Pre-loading features... '
                 'It may take a couple of minutes (Glove!)')
    loaders = []
    for i, (subset, filename) in enumerate(subset_files):
        index_config = 0 if i == 0 else -1
        extras_dataset = extras_dataset_configs[index_config]
        extras_loader = extras_loaders_configs[index_config]
        if not filename.exists():
            logging.info(f'Not found {subset}-list: {filename}')
            loaders.append(None)
            continue
        logging.info(f'Found {subset}-list: {filename}')
        dataset = dataset_untrimmed.UntrimmedMCN(
            filename, cues=cues, loc=args.loc, context=args.context,
            **extras_dataset)
        logging.info(f'Setting loader')
        loaders.append(
            DataLoader(dataset, num_workers=args.num_workers,
                       **extras_loader)
        )

    train_loader, val_loader, test_loader = loaders
    return train_loader, val_loader, test_loader


def setup_model(args, train_dataset):
    "Setup model, criterion and optimizer"
    logging.info('Setting-up MCN')
    mcn_setup = dict(visual_size=train_dataset.visual_size[args.feat],
                     lang_size=train_dataset.language_size,
                     max_length=train_dataset.max_words)
    net = model.MCN(**mcn_setup)

    # opt_parameters = None
    # if train_dataset is not None:
    opt_parameters = net.optimization_parameters(
        args.lr, args.original_setup)
    criterion = IntraInterMarginLoss(
        margin=args.margin, w_inter=args.w_inter,
        w_intra=args.w_intra)

    logging.info('Setting-up model mode')
    net.train()
    # if train_dataset is None:
    #     net.eval()

    if args.gpu_id >= 0:
        logging.info('Transferring model to GPU')
        net.to(args.device)
        # if ranking_loss is not None:
        criterion.to(args.device)

    logging.info(f'Setting-up optimizer: {args.optimizer}')
    # if opt_parameters is None:
    #     return net, None, None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(opt_parameters, lr=args.lr,
                              momentum=args.momentum)
    elif args.optimizer == 'sgd_caffe':
        optimizer = SGDCaffe(opt_parameters, lr=args.lr,
                             momentum=args.momentum)
    else:
        raise ValueError(f'Unknow optimizer {args.optimizer}')
    return net, criterion, optimizer


if __name__ == '__main__':
    main(args)
