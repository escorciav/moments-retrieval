"Train&evaluate MCN/CAL models"
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
import loss
from evaluation import single_moment_retrieval, didemo_evaluation
from dataset_untrimmed import TemporalFeatures
from utils import Multimeter, Tracker
from utils import collate_data, collate_data_eval, ship_to
from utils import setup_hyperparameters, setup_logging, setup_rng, setup_metrics
from utils import dumping_arguments, load_args_from_snapshot, save_checkpoint
from utils import get_git_revision_hash
from utils import MutableSampler
from np_segments_ops import non_maxima_suppresion

OPTIMIZER = ['sgd', 'sgd_caffe']
EVAL_BATCH_SIZE = 1
TOPK, IOU_THRESHOLDS = (1, 5), (0.5, 0.7)
MAX_TOPK = max(TOPK)
METRICS = [f'r@{k},{iou}' for iou, k in product(IOU_THRESHOLDS, TOPK)]
METRICS_OLD = ['iou', 'r@1', 'r@5']
VARS_TO_RECORD = ['id', 'hit@iou,k', 'scores', 'topk_segments']
TRACK = 'r@1,0.7'
BEST_RESULT = 0.0
TOPK_DIDEMO = torch.tensor([1, 5]).float()

parser = argparse.ArgumentParser(
    description='MCN/CAL training',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
# Data
parser.add_argument('--train-list', type=Path, default=Path('non-existent'),
                    help='JSON-file with training instances')
parser.add_argument('--val-list', type=Path, default=Path('non-existent'),
                    help='JSON-file with validation instances')
parser.add_argument('--test-list', type=Path, default=Path('non-existent'),
                    help='JSON-file with testing instances')
# Architecture
parser.add_argument('--arch', choices=model.MOMENT_RETRIEVAL_MODELS,
                    default='MCN', help='model architecture')
parser.add_argument('--snapshot', type=Path, default=Path(''),
                    help=('JSON with hyper-parameters. It also expects  a '
                          '.pth.tar file in the same directory to load the'
                          'model parameters. Those are placed into a '
                          'key called `state_dict`.'))
# Arch hyper-parameters
parser.add_argument('--visual-hidden', type=int, default=500,
                    help='Hidden unit in MLP visual stream')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate in visual stream')
parser.add_argument('--embedding-size', type=int, default=100,
                    help='Dimensionaity of cross-modal embedding')
parser.add_argument('--lang-hidden', type=int, default=1000,
                    help='Dimensionality of sentence representation')
parser.add_argument('--visual-layers', type=int, default=1,
                    help='Number of layers in visual encoder')
parser.add_argument('--unit-vector', action='store_true',
                    help='Enable embedding normalization')
# Program control
parser.add_argument('--evaluate', action='store_true',
                    help='only run the model in the val set')
# Features
parser.add_argument('--feat', default='rgb',
                    help='Record the type of feature used (modality)')
parser_visual_info_grp = parser.add_mutually_exclusive_group()
parser_visual_info_grp.add_argument(
    '--h5-path', type=Path, default='non-existent',
    help='HDF5-file with features')
parser_visual_info_grp.add_argument(
    '--clip-length', type=float, default=None,
    help='Clip length in seconds')
# Model features
parser.add_argument('--loc', type=TemporalFeatures.from_string,
                    default=TemporalFeatures.TEMPORAL_ENDPOINT,
                    choices=list(TemporalFeatures),
                    help='Kind of temporal moment feature')
parser.add_argument('--no-context', action='store_false', dest='context',
                    help='Remove global video representation')
# Model
parser.add_argument('--margin', type=float, default=0.1,
                    help='MaxMargin margin value')
parser.add_argument('--w-inter', type=float, default=0.2,
                    help='Inter-loss weight')
parser.add_argument('--w-intra', type=float, default=0.5,
                    help='Intra-loss weight')
# TODO: add weight for clip loss
parser.add_argument('--c-inter', type=float, default=0.2,
                    help='Clip-inter-loss weight')
parser.add_argument('--c-intra', type=float, default=0.5,
                    help='Clip-intra-loss weight')
parser.add_argument('--original-setup', action='store_true',
                    help='Enable original optimization policy')
parser.add_argument('--proposals-in-train', action='store_true',
                    help='Sample negative from proposals during training')
parser.add_argument('--negative-sampling-iou', type=float, default=0.35,
                    help='Amount of IoU to consider proposals as negatives')
parser.add_argument('--freeze-visual', action='store_true')
# parser.add_argument('--freeze-visual-encoder', action='store_true')
parser.add_argument('--freeze-lang', action='store_true')
parser.add_argument('--context-window', type=int, default=None,
                    help=('Size of context windows around each clip. '
                          'Valid only for SMCN.'))
parser.add_argument('--bias-to-single-clips', type=float, default=0,
                    help='Upsample single clip moments, 0 means no bias.')
parser.add_argument('--clip-loss', action='store_true')
parser.add_argument('--ground-truth-rate', type=float, default=1.0,
                    help='Pos moment augmentation if its lower than 1')
parser.add_argument('--prob-proposal-nextto', type=float, default=-1.0,
                    help=('Prob to sample negatives next to moments. -1'
                          'means disabled. Increase it to sample often.'))
parser.add_argument('--h5-path-nis', type=Path, default=None,
                    help='HDF5-file for negative importance sampling')
parser.add_argument('--nis-k', type=int, default=None,
                    help='Only sample negative videos from top-k')
# Hyper-parameters concerning proposals (candidates) to score
parser.add_argument('--proposal-interface', default='SlidingWindowMSRSS',
                    choices=proposals.PROPOSAL_SCHEMES,
                    help='Type of proposals spanning search space')
parser.add_argument('--min-length', type=float, default=1.5,
                    help='Minimum length of sliding windows (seconds)')
parser.add_argument('--scales', type=int, nargs='+',
                    default=list(range(2, 17, 2)),
                    help='Relative durations for sliding windows')
parser.add_argument('--stride', type=float, default=0.5,
                    help=('Relative stride for sliding windows [0, 1]. '
                          'Check SlidingWindowMSRSS details'))
parser.add_argument('--nms-threshold', type=float, default=0.5)
# Device specific
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--gpu-id', type=int, default=-1, help='GPU device')
parser.add_argument('--num-workers', type=int, default=6,
                    help='Num workers during training')
parser.add_argument('--num-workers-eval', type=int, default=6,
                    help='Num workers for evaluation (useful long videos)')
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
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--no-shuffle', action='store_false', dest='shuffle',
                    help='Disable suffle dataset after each epoch')
# train/eval callbacks
parser.add_argument('--patience', type=int, default=-1,
                    help=('stop optimization if there is no improvements '
                          'after poking test-list by this amount.'))
parser.add_argument('--min-epochs', type=float, default=0.45,
                    help=('Ignore patience at the begining. Given as ratio '
                          'wrt total numbers of epochs.'))
parser.add_argument('--eval-on-epoch', type=int, default=-1,
                    help=('Check validation in multiples of these epoch.'
                          'Raises an error if the actual `test-list` is given '
                          'and `eval_on_epoch` > 0'))
parser.add_argument('--not-serialize', action='store_false', dest='serialize',
                    help='Avoid dumping .pth.tar with model parameters')
parser.add_argument('--dump-results', action='store_true',
                    help='Dump HDF5 with per sample results in val & test')
parser.add_argument('--save-on-epoch', type=int, default=-1,
                    help='Dump every given number of epochs')
parser.add_argument('--force-eval-end', action='store_true',
                    help='Force eval at the end of training')
# Logging
parser.add_argument('--logfile', type=Path, default=Path(''),
                    help='Logging file')
parser.add_argument('--n-display', type=float, default=0.1,
                    help='logging rate during epoch')
parser.add_argument('--enable-tb', action='store_true',
                    help='Log to tensorboard')
# Reproducibility
parser.add_argument('--seed', type=int, default=1701,
                    help='random seed (-1 := random)')
# Hyper-parameter search
parser.add_argument('--hps', action='store_true',
                    help='Enable use of hps.yml in folder of logfile')
# Debug
parser.add_argument('--debug', action='store_true',
                    help=('yield incorrect results! only to verify things '
                          '(loaders, ops) can run well'))

args = parser.parse_args()


def main(args):
    setup_logging(args)
    if load_args_from_snapshot(args):
        if len(args.snapshot.name) > 0:
            # Override snapshot config with user argument
            args = parser.parse_args(namespace=args)
            logging.info(f'Loaded snapshot config: {args.snapshot}')
    else:
        logging.error('Unable to load {args.snapshot}, procedding with args.')
    setup_hyperparameters(args)
    setup_rng(args)
    setup_metrics(args, TOPK, IOU_THRESHOLDS, TOPK_DIDEMO)

    args.device = device_name = 'cpu'
    if args.gpu_id >= 0 and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_id}')
        device_name = torch.cuda.get_device_name(args.gpu_id)
        if args.nms_threshold >= 1:
            args.topk = args.topk.cuda(device=args.device, non_blocking=True)
            args.topk_ = args.topk_.cuda(
                device=args.device, non_blocking=True)
    logging.info(f'Git revision hash: {get_git_revision_hash()}')
    args.git_hash = get_git_revision_hash()
    logging.info(f'{args.arch} in untrimmed videos')
    logging.info(args)
    logging.info(f'Device: {device_name}')

    train_loader, val_loader, test_loader = setup_dataset(args)
    net, ranking_loss, optimizer = setup_model(
        args, train_loader, test_loader)

    if len(args.snapshot.name) > 0 and args.snapshot.exists():
        logging.info(f'Load model-parameters from {args.snapshot}')
        filename = args.snapshot.with_suffix('.pth.tar')
        snapshot = torch.load(filename).get('state_dict')
        if snapshot is not None:
            net.load_state_dict(snapshot)
        else:
            logging.error('Fail loading parameters, proceeding without them.')

    if args.evaluate:
        if not args.snapshot.exists():
            logging.error('Aborting due to lack of snapshot')
            return
        logging.info('Evaluating model')
        # Update `arg.logfile` re-using basename from `args.snapshot` and
        # append eval to avoid overwriting
        args.logfile = args.snapshot.with_suffix('')
        args.logfile = args.logfile.with_name(args.logfile.stem + '_eval')
        rst, per_sample_rst = evaluate(args, net, test_loader)
        dumping_arguments(args, test_performance=rst,
                          perf_per_sample_test=per_sample_rst)
        return

    lr_schedule = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_step, args.lr_decay)

    logging.info('Starting optimization...')
    # on training begin
    n_display_float = args.n_display
    total_epochs = args.epochs
    args.min_epochs = max(1, int(total_epochs * args.min_epochs))
    patience, args.epochs, best_result = 0, 0, BEST_RESULT
    if args.eval_on_epoch > 0:
        logging.info(f'Epoch: {args.epochs}')
        perf_val, perf_per_sample_val = evaluate(args, net, val_loader)
        perf_test, perf_per_sample_test = evaluate(
            args, net, test_loader, subset='test')
        performance = perf_test if perf_test is None else perf_val
        best_result = performance[TRACK]
    for epoch in range(1, total_epochs + 1):
        # on epoch begin
        args.epochs += 1
        args.n_display = max(int(n_display_float * len(train_loader)), 1)
        lr_schedule.step()

        train_epoch(args, net, ranking_loss, train_loader, optimizer)

        # on epoch ends
        # eval on val/test-lists
        if args.eval_on_epoch > 0 and epoch % args.eval_on_epoch == 0:
            logging.info('Trigger evaluation')
            perf_val, perf_per_sample_val = evaluate(args, net, val_loader)
            perf_test, perf_per_sample_test = evaluate(
                args, net, test_loader, subset='test')
            performance = perf_test if perf_test is None else perf_val
            current_result = performance[TRACK] if performance else best_result
            if args.patience >= 0 and current_result > best_result:
                patience = 0
                best_result = current_result
                save_checkpoint(args, {'state_dict': net.state_dict()})
            else:
                patience += 1

        # dump model at given epoch
        if args.save_on_epoch > 0 and epoch % args.save_on_epoch == 0:
            save_checkpoint(args, {'state_dict': net.state_dict()}, record=True)

        # stop training when if no improvement
        if patience == args.patience and epoch > args.args.min_epochs:
            break
    # on train ends
    if args.eval_on_epoch < 0 or args.force_eval_end:
        logging.info('Evaluation after training finished')
        perf_val, perf_per_sample_val = evaluate(args, net, val_loader)
        perf_test, perf_per_sample_test = evaluate(
            args, net, test_loader, subset='test')
        save_checkpoint(args, {'state_dict': net.state_dict()})

    logging.info(f'Best val {TRACK}: {best_result:.4f}')
    dumping_arguments(args, perf_val, perf_test,
                      perf_per_sample_val, perf_per_sample_test)


def train_epoch(args, net, criterion, loader, optimizer):
    "Update model making a whole pass over dataset in loader"
    ind = 2 if args.debug else 0
    time_meters = Multimeter(keys=['Data', 'Batch'])
    running_loss = 0.0
    logging.info(f'Epoch: {args.epochs}')
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

        loss_it = loss.item()
        running_loss += loss_it
        n_iter = it + args.epochs * len(loader)
        if args.writer:
            args.writer.add_scalar('train/loss', loss_it, n_iter)
        if (it + 1) % args.n_display == 0:
            logging.info(f'Epoch: [{args.epochs}]'
                         f'[{100 * it / len(loader):.2f}]\t'
                         f'{time_meters.report()}\t'
                         f'Loss {running_loss / args.n_display:.4f}')
            running_loss = 0.0
    if args.writer:
        args.writer.add_scalar('train/loss-end', running_loss, args.epochs)


def validation(args, net, loader):
    "Eval model"
    ind = 2 if args.debug else 0
    time_meters = Multimeter(keys=['Batch', 'Eval'])
    meters_1, meters_2 = Multimeter(keys=METRICS), None
    if args.proposal_interface == 'DidemoICCV17SS':
        meters_2 = Multimeter(keys=METRICS_OLD)
    tracker = Tracker(keys=VARS_TO_RECORD) if args.dump_results else None
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

            gt_segment, segments = minibatch[-2:]
            if args.nms_threshold < 1:
                # TODO(tier-1): port to torch
                gt_segment, segments = minibatch_[-2:]
                scores = results.cpu()
                idx = non_maxima_suppresion(
                    segments.numpy(), -scores.numpy(), args.nms_threshold)
                sorted_segments = segments[idx]
            else:
                scores, idx = results.sort(descending=descending)
                sorted_segments = segments[idx, :]

            hit_k_iou = single_moment_retrieval(
                gt_segment, sorted_segments, topk=args.topk)
            if meters_2:
                iou_r_at_ks = didemo_evaluation(
                    gt_segment, sorted_segments, args.topk_)
                meters_2.update([i.item() for i in iou_r_at_ks])

            # TODO(tier-2;profile): seems a bit slow
            meters_1.update([i.item() for i in hit_k_iou])
            time_meters.update([batch_time, time.time() - end])
            if tracker:
                # artifact to ease Tracker job for few number of segments
                if scores.shape[0] < MAX_TOPK:
                    n_times = round(MAX_TOPK / scores.shape[0])
                    scores = scores.repeat(n_times)
                    sorted_segments = sorted_segments.repeat(n_times, 1)
                tracker.append(it, hit_k_iou, scores[: MAX_TOPK],
                               sorted_segments[:MAX_TOPK, :])
            end = time.time()
    logging.info(f'{time_meters.report()}\t{meters_1.report()}')
    if meters_2:
        logging.info(f'DiDeMo metrics: {meters_2.report()}')
    if tracker:
        tracker.freeze()
        performance = (meters_1.dump(), tracker.data)
        logging.info(f'Results aggregation completed')
        return performance
    return meters_1.dump(), None


def evaluate(args, net, loader, subset='val'):
    """Lazy function to make comparsion = to json parsing.

    Note: Please do not fool yourself picking model based on testing
    """
    if loader is None:
        return None, None
    perf_overall, perf_per_sample = validation(args, net, loader)
    if perf_overall and args.writer:
        for metric, value in perf_overall.items():
            args.writer.add_scalar(f'{subset}/{metric}', value, args.epochs)
    return perf_overall, perf_per_sample


def sampler_biased_single_clip_moment(dataset, rate):
    """Bias sampling towards single clip moments

    TODO: experimental feature. Remove if it's useless.
    """
    raise ValueError('Deprecated')
    if 'didemo' in str(dataset.json_file):
        return None
    clip_length = dataset.proposals_interface.stride
    ind_per_length = {}
    for it, moment_data in enumerate(dataset.metadata):
        moment = moment_data['times'][0]
        length = moment[1] - moment[0]
        num_clips = int(length // clip_length)
        if num_clips not in ind_per_length:
            ind_per_length[num_clips] = []
        ind_per_length[num_clips].append(it)
    max_instances_in_bucket = max([len(i) for i in ind_per_length.values()])
    num_instances_single_clip = len(ind_per_length[0])
    upsample_rate = int(rate * max_instances_in_bucket //
                        num_instances_single_clip)
    ind_per_length[0] = ind_per_length[0] * upsample_rate
    return MutableSampler(sum(ind_per_length.values(), []))


def setup_dataset(args):
    "Setup dataset and loader"
    # model dependend part
    if args.arch == 'MCN':
        dataset_name = 'UntrimmedMCN'
    elif args.arch == 'SMCN':
        dataset_name = 'UntrimmedSMCN'
    else:
        raise ValueError(f'Unsuported arch: {args.arch}, call 911!')

    # Make sure test_list doesn't cheating
    search_for = 'test'
    risk_of_cheating = (not args.evaluate and
                        search_for in args.test_list.name and
                        args.eval_on_epoch > 0)
    if 'activitynet' in args.test_list.name:
        search_for = 'val.json'
    if risk_of_cheating:
        raise ValueError('Risk of cheating. Please read prolog.')

    # Save loading time in case [train, val]-list exits
    if args.evaluate:
        logging.info('Overriding [train, val]-list to save time :)')
        args.train_list = Path('non-existent')
        args.val_list = Path('non-existent')

    subset_files = [('train', args.train_list), ('val', args.val_list),
                    ('test', args.test_list)]
    cues, no_visual = {args.feat: None}, True
    if args.h5_path.exists():
        no_visual = False
        cues = {args.feat: {'file': args.h5_path}}
    elif args.clip_length is None:
        raise ValueError('clip-length is required without visual features')
    proposal_generator = proposals.__dict__[args.proposal_interface](
        args.min_length, args.scales, args.stride)
    proposal_generator_train = None
    if args.proposals_in_train:
        proposal_generator_train = proposal_generator
    extras_dataset_configs = [
        # Training
        {'proposals_interface': proposal_generator_train,
         'ground_truth_rate': args.ground_truth_rate,
         'prob_nproposal_nextto': args.prob_proposal_nextto,
         'sampling_iou': args.negative_sampling_iou,
         'h5_nis': args.h5_path_nis, 'nis_k': args.nis_k},
        # Validation or Testing
        {'eval': True, 'proposals_interface': proposal_generator}
    ]
    extras_loaders_configs = [
        # Training
        {'shuffle': args.shuffle, 'collate_fn': collate_data,
         'batch_size': args.batch_size, 'num_workers': args.num_workers},
        # Validation or Testing
        {'shuffle': False, 'collate_fn': collate_data_eval,
         'batch_size': EVAL_BATCH_SIZE, 'num_workers': args.num_workers_eval}
    ]

    logging.info('Setting-up datasets and loaders')
    logging.info('Pre-loading features... '
                 'It may take a couple of minutes (Glove!)')
    loaders = []
    for i, (subset, filename) in enumerate(subset_files):
        index_config = 0 if i == 0 else -1
        extras_dataset = extras_dataset_configs[index_config]
        if dataset_name == 'UntrimmedSMCN':
            extras_dataset['w_size'] = args.context_window
            extras_dataset['clip_mask'] = args.clip_loss
        extras_loader = extras_loaders_configs[index_config]
        if not filename.exists():
            logging.info(f'Not found {subset}-list: {filename}')
            loaders.append(None)
            continue
        logging.info(f'Found {subset}-list: {filename}')
        dataset = dataset_untrimmed.__dict__[dataset_name](
            filename, cues=cues, loc=args.loc, context=args.context,
            no_visual=no_visual, debug=args.debug,
            clip_length=args.clip_length,
            **extras_dataset)
        logging.info(f'Setting loader')
        if subset == 'train' and args.bias_to_single_clips > 0:
            extras_loader['shuffle'] = False
            extras_loader['sampler'] = sampler_biased_single_clip_moment(
                dataset, args.bias_to_single_clips)
        loaders.append(
            DataLoader(dataset, **extras_loader)
        )

    train_loader, val_loader, test_loader = loaders
    return train_loader, val_loader, test_loader


def setup_model(args, train_loader=None, test_loader=None):
    "Setup model, criterion and optimizer"
    if train_loader is not None:
        dataset = train_loader.dataset
    elif test_loader is not None:
        dataset = test_loader.dataset
    else:
        raise ValueError('Either train or test list must exists')
    logging.info('Setting-up model')
    arch_setup = dict(
        visual_size=dataset.visual_size[args.feat],
        lang_size=dataset.language_size,
        max_length=dataset.max_words,
        embedding_size=args.embedding_size,
        dropout=args.dropout,
        visual_hidden=args.visual_hidden,
        lang_hidden=args.lang_hidden,
        visual_layers=args.visual_layers,
        unit_vector=args.unit_vector
    )
    if args.clip_loss:
        args.arch = 'SMCNCL'
    net = model.__dict__[args.arch](**arch_setup)

    opt_parameters, criterion = None, None
    if train_loader is not None:
        logging.info('Setting-up criterion')
        opt_parameters = net.optimization_parameters(
            args.lr, args.original_setup, freeze_visual=args.freeze_visual,
            freeze_lang=args.freeze_lang,
            # freeze_visual_encoder=args.freeze_visual_encoder)
        )

        criterion_ = 'IntraInterMarginLoss'
        criterion_prm = dict(margin=args.margin, w_inter=args.w_inter,
                             w_intra=args.w_intra)
        if args.clip_loss:
            criterion_ = 'IntraInterMarginLossPlusClip'
            criterion_prm['c_inter'] = args.c_inter
            criterion_prm['c_intra'] = args.c_intra
        criterion = loss.__dict__[criterion_](**criterion_prm)

    logging.info('Setting-up model mode')
    net.train()
    if train_loader is None:
        net.eval()

    if args.gpu_id >= 0:
        logging.info('Transferring model to GPU')
        net.to(args.device)
        if criterion is not None:
            criterion.to(args.device)

    logging.info(f'Setting-up optimizer: {args.optimizer}')
    if opt_parameters is None:
        return net, None, None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(opt_parameters, lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd_caffe':
        optimizer = SGDCaffe(opt_parameters, lr=args.lr,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Unknow optimizer {args.optimizer}')
    return net, criterion, optimizer


if __name__ == '__main__':
    main(args)
