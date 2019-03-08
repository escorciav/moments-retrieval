import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

import dataset_untrimmed
import model
import proposals
from corpus import LoopOverKVideos
from evaluation import CorpusVideoMomentRetrievalEval
from utils import setup_logging, get_git_revision_hash

parser = argparse.ArgumentParser(
    description='Corpus Retrieval 2nd Stage Evaluation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data
parser.add_argument('--test-list', type=Path, required=True,
                    help='JSON-file with corpus instances')
parser.add_argument('--h5-path', type=Path, default='non-existent',
                    help='HDF5-file with features')
# Architecture
parser.add_argument('--snapshot', type=Path, required=True,
                    help='JSON-file of model')
parser.add_argument('--h5-1ststage', type=Path, required=True,
                    help='HDF5-file of 1st stage results')
parser.add_argument('--k-first', type=int, required=True,
                    help='K first retrieved resuslts')
# Evaluation parameters
parser.add_argument('--topk', nargs='+', type=int,
                    default=[1, 10, 100],
                    help='top-k values to compute in ascending order.')
# Dump results and logs
parser.add_argument('--dump', action='store_true',
                    help='Save log in text file and json')
parser.add_argument('--logfile', type=Path, default='',
                    help='Logging file')
parser.add_argument('--n-display', type=float, default=0.2,
                    help='logging rate during epoch')
parser.add_argument('--disable-tqdm', action='store_true',
                    help='Disable progress-bar')
# Debug
parser.add_argument('--debug', action='store_true',
                    help=('yield incorrect results! to verify things are'
                          'glued correctly (dataset, model, eval)'))
args = parser.parse_args()


def main(args):
    "Put all the pieces together"
    if args.dump:
        args.disable_tqdm = True
        if len(args.logfile.name) == 0:
            basename = args.snapshot.with_suffix('')
            args.logfile = basename.with_name(
                args.snapshot.stem + '_corpus-2nd-eval')
        if args.logfile.exists():
            raise ValueError(
                f'{args.logfile} already exists. Please provide a logfile or'
                'backup existing results.')
    setup_logging(args)

    logging.info('Corpus Retrieval Evaluation for 2nd Stage')
    load_hyperparameters(args)
    logging.info(args)

    engine_prm = {}
    if args.arch == 'MCN':
        args.dataset = 'UntrimmedMCN'
    elif args.arch == 'SMCN':
        args.dataset = 'UntrimmedSMCN'
    else:
        ValueError('Unknown/unsupported architecture')

    logging.info('Loading dataset')
    if args.h5_path.exists():
        dataset_novisual = False
        dataset_cues = {args.feat: {'file': args.h5_path}}
    else:
        raise NotImplementedError('WIP')
    proposals_interface = proposals.__dict__[args.proposal_interface]
    proposals_prm = dict(
        length=args.min_length, stride=args.stride,
        num_scales=args.num_scales, unique=True)
    dataset_setup = dict(
        json_file=args.test_list, cues=dataset_cues, loc=args.loc,
        context=args.context, debug=args.debug, eval=True,
        no_visual=dataset_novisual,
        proposals_interface=proposals_interface(**proposals_prm)
    )
    dataset = dataset_untrimmed.__dict__[args.dataset](**dataset_setup)
    if args.arch == 'SMCN':
        logging.info('Set padding on UntrimmedSMCN dataset')
        dataset.set_padding(False)

    logging.info('Setting up models')
    arch_setup = dict(
        visual_size=dataset.visual_size[args.feat],
        lang_size=dataset.language_size,
        max_length=dataset.max_words,
        embedding_size=args.embedding_size,
        visual_hidden=args.visual_hidden,
        lang_hidden=args.lang_hidden,
        visual_layers=args.visual_layers,
    )
    net = model.__dict__[args.arch](**arch_setup)
    filename = args.snapshot.with_suffix('.pth.tar')
    snapshot_ = torch.load(
        filename, map_location=lambda storage, loc: storage)
    net.load_state_dict(snapshot_['state_dict'])
    net.eval()

    engine = LoopOverKVideos(
        dataset, net, args.h5_1ststage, topk=args.k_first)

    logging.info('Launch evaluation...')
    # log-scale up to the end of the database
    if len(args.topk) == 1 and args.topk[0] == 0:
        exp = int(np.floor(np.log10(engine.num_moments)))
        args.topk = [10**i for i in range(0, exp + 1)]
        args.topk.append(engine.num_moments)
    num_instances_retrieved = []
    judge = CorpusVideoMomentRetrievalEval(topk=args.topk)
    args.n_display = max(int(args.n_display * len(dataset.metadata)), 1)
    for it, query_metadata in tqdm(enumerate(dataset.metadata),
                                   disable=args.disable_tqdm):
        vid_indices, segments = engine.query(
            query_metadata['language_input'], description_ind=it)
        judge.add_single_predicted_moment_info(
            query_metadata, vid_indices, segments, max_rank=engine.num_moments)
        num_instances_retrieved.append(len(vid_indices))
        if args.disable_tqdm and (it + 1) % args.n_display == 0:
            logging.info(f'Processed queries [{it}/{len(dataset.metadata)}]')

    logging.info('Summarizing results')
    num_instances_retrieved = np.array(num_instances_retrieved)
    logging.info(f'Number of queries: {len(judge.map_query)}')
    logging.info(f'Number of proposals: {engine.num_moments}')
    retrieved_proposals_median = int(np.median(num_instances_retrieved))
    retrieved_proposals_min = int(num_instances_retrieved.min())
    if (num_instances_retrieved != engine.num_moments).any():
        logging.info('Triggered approximate search')
        logging.info('Median numbers of retrieved proposals: '
                     f'{retrieved_proposals_median:d}')
        logging.info('Min numbers of retrieved proposals: '
                     f'{retrieved_proposals_min:d}')
    result = judge.evaluate()
    _ = [logging.info(f'{k}: {v}') for k, v in result.items()]
    if args.dump:
        filename = args.logfile.with_suffix('.json')
        logging.info(f'Dumping results into: {filename}')
        with open(filename, 'x') as fid:
            for key, value in result.items():
                result[key] = float(value)
            result['snapshot'] = str(args.snapshot)
            result['corpus'] = str(args.test_list)
            result['1ststage'] = str(args.h5_1ststage)
            result['topk'] = args.topk
            result['iou_threshold'] = judge.iou_thresholds
            result['k_first'] = args.k_first
            result['median_proposals_retrieved'] = retrieved_proposals_median
            result['min_proposals_retrieved'] = retrieved_proposals_min
            result['date'] = datetime.now().isoformat()
            result['git_hash'] = get_git_revision_hash()
            json.dump(result, fid, indent=1)


def load_hyperparameters(args):
    "Update args with model hyperparameters"
    logging.info('Parsing JSON files with hyper-parameters')
    with open(args.snapshot, 'r') as fid:
        hyper_prm = json.load(fid)

    for key, value in hyper_prm.items():
        if not hasattr(args, key):
            setattr(args, key, value)
        else:
            logging.debug(f'Ignored hyperparam: {key}')


if __name__ == '__main__':
    main(args)