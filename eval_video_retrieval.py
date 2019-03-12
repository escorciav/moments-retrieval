import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from dataset_untrimmed import UntrimmedBasedMCNStyle
from utils import setup_logging, get_git_revision_hash
from utils import unique2d_perserve_order

parser = argparse.ArgumentParser(
    description='Text2Video Retrieval Evaluation over dumped results',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data
parser.add_argument('--test-list', type=Path, required=True,
                    help='JSON-file with corpus instances')
# Architecture
parser.add_argument('--snapshot', type=Path, required=True,
                    help='JSON-file of model. Only for stats (must exists)')
parser.add_argument('--h5-1ststage', type=Path, required=True,
                    help='HDF5-file with retrieval results')
# Evaluation parameters
parser.add_argument('--topk', nargs='+', type=int,
                    default=[1, 10, 100],
                    help='top-k values to compute in ascending order.')
# Dump results and logs
parser.add_argument('--dump', action='store_true',
                    help='Save log in text file and json')
parser.add_argument('--logfile', type=Path, default='',
                    help='Logging file')
parser.add_argument('--output-prefix', type=Path, default='',
                    help="")
parser.add_argument('--n-display', type=float, default=0.2,
                    help='logging rate during epoch')
parser.add_argument('--disable-tqdm', action='store_true',
                    help='Disable progress-bar')
parser.add_argument('--enable-tb', action='store_true',
                    help='Log to tensorboard. Nothing logged by this program')
args = parser.parse_args()


def main(args):
    "Quick and dirty evaluation"
    if not args.snapshot.exists():
        raise ValueError('Please provide snapshot')

    if args.dump:
        args.disable_tqdm = True
        if len(args.logfile.name) == 0:
            basename = args.snapshot.with_suffix('')
            args.logfile = basename.parent.joinpath(
                args.output_prefix, basename.stem + '_text2video-eval')
            if not args.logfile.parent.exists():
                args.logfile.parent.mkdir()
        if args.logfile.exists():
            raise ValueError(
                f'{args.logfile} already exists. Please provide a logfile or'
                'backup existing results.')
    setup_logging(args)

    logging.info('Eval video retrieval from description')
    logging.info(args)

    logging.info('Setting up dataset')
    dataset = UntrimmedBasedMCNStyle(
        args.test_list, cues={}, no_visual=True, clip_length=3,
        # We use debug as we are not going to use visual or language data
        debug=True)

    logging.info('Loading retrieval results')
    with h5py.File(args.h5_1ststage, 'r') as fid:
        query2videos_ind = fid['vid_indices'][:]
    if query2videos_ind.shape[1] > dataset.num_videos:
        query2videos_ind = unique2d_perserve_order(query2videos_ind)

    logging.info('Launch evaluation...')
    # log-scale up to the end of the database
    if len(args.topk) == 1 and args.topk[0] == 0:
        exp = int(np.floor(np.log10(dataset.num_videos)))
        args.topk = [10**i for i in range(0, exp + 1)]
        args.topk.append(dataset.num_videos)
    args.n_display = max(int(args.n_display * len(dataset.metadata)), 1)
    topk_ = np.array(args.topk)
    hit = np.zeros_like(topk_)
    rank = []
    for it, query_metadata in tqdm(enumerate(dataset.metadata),
                                   disable=args.disable_tqdm):
        pred_vid_indices = query2videos_ind[it, :]

        # Evaluation
        gt_vid_index = query_metadata['video_index']
        rank_it = (pred_vid_indices == gt_vid_index).nonzero()[0][0] + 1
        hit_it = topk_ >= rank_it
        hit += hit_it
        rank.append(rank_it)

        if args.disable_tqdm and (it + 1) % args.n_display == 0:
            logging.info(f'Processed queries [{it}/{len(dataset.metadata)}]')
    recall = hit / len(dataset)
    median_rank = int(np.median(np.array(rank)))

    logging.info('Summarizing results')
    logging.info(f'Number of queries: {len(dataset)}')
    logging.info(f'Number of videos: {dataset.num_videos}')
    _ = [logging.info(f'{args.topk[i]}: {float(v)}')
         for i, v in enumerate(recall)]
    logging.info(f'MedRank: {median_rank}')
    if args.dump:
        filename = args.logfile.with_suffix('.json')
        logging.info(f'Dumping results into: {filename}')
        data = {}
        with open(filename, 'x') as fid:
            data['snapshot'] = str(args.snapshot)
            data['corpus'] = str(args.test_list)
            data['1ststage'] = str(args.h5_1ststage)
            data['topk'] = args.topk
            data['recall'] = recall.tolist()
            data['MedRank'] = median_rank
            data['date'] = datetime.now().isoformat()
            data['git_hash'] = get_git_revision_hash()
            json.dump(data, fid, indent=1)


if __name__ == '__main__':
    main(args)