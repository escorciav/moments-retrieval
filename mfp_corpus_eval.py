"""Baseline for moment retrieval from a corpus of videos

Baselines considered here are Moment Frequency Prior and Random Chance
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from corpus import DummyMomentRetrievalFromProposalsTable
from evaluation import CorpusVideoMomentRetrievalEval
from moment_freq_prior import setup_dataset, setup_model
from utils import load_args_from_snapshot, setup_logging, get_git_revision_hash

parser = argparse.ArgumentParser(
    description='Corpus Retrieval Baseline Evaluation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data
parser.add_argument('--test-list', type=Path, required=True,
                    help='JSON-file with corpus instances')
parser.add_argument('--snapshot', type=Path, required=True,
                    help=('JSON file with config. It expects to find an '
                          'NPZ-file in the same directory with the model '
                          'parameters'))
# Baseline
parser.add_argument('--chance', action='store_true',
                    help='Perform random chance as opposed to MFP')
# Evaluation parameters
parser.add_argument('--topk', nargs='+', type=int, default=[1, 5],
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
parser.add_argument('--enable-tb', action='store_true',
                    help='Log to tensorboard. Nothing logged by this program')
args = parser.parse_args()


def main(args):
    if args.dump:
        args.disable_tqdm = True
        if len(args.logfile.name) == 0:
            args.logfile = args.snapshot_args.with_suffix('').with_name(
                args.snapshot_args.stem + '_corpus-eval')
        if args.logfile.exists():
            raise ValueError(
                f'{args.logfile} already exists. Please provide a logfile or'
                'backup existing results.')
    setup_logging(args)
    if load_args_from_snapshot(args):
        if len(args.snapshot.name) > 0:
            # Override snapshot config with user argument
            args = parser.parse_args(namespace=args)
            logging.info(f'Loaded snapshot config: {args.snapshot}')
            # This is a dirty trick as we plan to ignore "train"-dataset
            args.train_list = args.test_list
    else:
        logging.error('Unable to load {}, procedding with args.')

    logging.info('Baseline moment retrieval from a corpus of videos')
    logging.info(args)

    logging.info('Loading dataset')
    _, dataset = setup_dataset(args)

    logging.info('Setting up model')
    model = setup_model(args)

    file_npz = args.snapshot.with_suffix('.npz')
    if not file_npz.exists():
        raise ValueError(f'Not found: {file_npz}')
    logging.info('Evaluating moment frequency prior')
    model.load(file_npz)
    # logging.info('Evaluating random chance')
    # raise NotImplementedError('WIP')

    logging.info('Creating database alas indexing corpus')
    engine = DummyMomentRetrievalFromProposalsTable(
        dataset, dataset.cues)
    engine.indexing()

    logging.info('Computing score of all candidates in database')
    N = len(engine.proposals)
    # Random chance corresponds as all the proposals having the same score
    proposals_score = np.ones((N), dtype=float) / N
    if not args.chance:
        for i, proposal in enumerate(engine.proposals):
            video_ind = engine.video_indices[i].item()
            video_id = engine.dataset.videos[video_ind]
            duration_i = engine.dataset._video_duration(video_id)
            proposal_np = proposal.numpy()[None, :]
            proposals_score[i] = model.predict(proposal_np, duration_i)
    ind = np.argsort(-proposals_score)
    video_indices = engine.video_indices[ind]
    proposals = engine.proposals[ind, :]
    sorted_scores = proposals_score[ind]
    breaks = sorted_scores[1:] - sorted_scores[:-1]
    subset_ind = np.concatenate([[-1], (breaks != 0).nonzero()[0]])
    # Form buckets with indices of proposals with the same score. Random chance
    # corresponds to a single bucket with all the proposals
    if args.chance:
        subsets = [np.arange(N)]
    else:
        subsets = [np.arange(ind_ + 1, subset_ind[i + 1] + 1)
                for i, ind_ in enumerate(subset_ind[:-1])]
        subsets += [np.arange(subset_ind[-1] + 1, len(proposals))]

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

        # Shuffle each bucket
        for subset_i in subsets:
            np.random.shuffle(subset_i)
        ind = np.concatenate(subsets, axis=0)

        segments = proposals[ind, :]
        vid_indices = video_indices[ind]
        judge.add_single_predicted_moment_info(
            query_metadata, vid_indices, segments, max_rank=engine.num_moments)
        num_instances_retrieved.append(len(vid_indices))
        if args.disable_tqdm and (it + 1) % args.n_display == 0:
            logging.info(f'Processed queries [{it}/{len(dataset.metadata)}]')

    logging.info('Summarizing results')
    moments_scanned_median = np.median(num_instances_retrieved)
    logging.info(f'Number of queries: {len(judge.map_query)}')
    logging.info(f'Number of proposals: {engine.num_moments}')
    logging.info('Median numbers of moments scanned: '
                 f'{int(moments_scanned_median):d}')
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
            result['topk'] = args.topk
            result['iou_threshold'] = judge.iou_thresholds
            result['git_hash'] = get_git_revision_hash()
            json.dump(result, fid, indent=1)


if __name__ == '__main__':
    main(args)