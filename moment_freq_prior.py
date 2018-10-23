"""Frequency prior baseline

Note: trivial baseline based on the original MCN model. A more sophisticated
adaption should take into account the video length. However, this would yield
to continuous PDF which should be approximated via KDE or similar techniques.

Please reach @escorciav out if you are interested in implementing such kind of
baseline. @ModarTensai helped me to sketch out a nice approach which we could
not implement due to lack of time.
1. Estimate f, a 2D-pdf of start-point and duration in train set.
2. Given video proposals
    a. Estimate the prob of each proposas i.e. p_i <- f(s_i) where s_i is a
       given segment proposal and p_i is the likelihood of being sampled.
3. From there you can sample with replacement, possibly applying NMS to
   sample with "diversity", or return the sorted segments.
"""
import argparse
import logging
from itertools import product
from pathlib import Path

import numpy as np
import torch

import dataset_untrimmed
import proposals
from evaluation import single_moment_retrieval
from np_segments_ops import iou as segment_iou
from utils import setup_logging
from utils import Multimeter

# TODO(tier-2;release): mirror this constants from init file
TOPK_IOU_POINTS = tuple(product((1, 5), (0.5, 0.7)))
METRICS = [f'r@{k},{iou}' for k, iou in TOPK_IOU_POINTS]

parser = argparse.ArgumentParser(description='Frequency prior')
parser.add_argument('--train-list', type=Path, default='non-existent',
                    help='JSON-file with training instances')
parser.add_argument('--test-list', type=Path, default='non-existent',
                    help='JSON-file with training instances')
# Features
parser.add_argument('--feat', default='rgb',
                    help='Record the type of feature used (modality)')
parser.add_argument('--h5-path', type=Path, default='non-existent',
                    required=True, help='HDF5-file with features')
# Hyper-parameters to explore search space (inference)
parser.add_argument('--min-length', type=float, default=3,
                    help='Minimum length of slidding windows (seconds)')
parser.add_argument('--num-scales', type=int, default=8,
                    help='Number of scales in a multi-scale linear slidding '
                         'window')
parser.add_argument('--stride', type=float, default=3,
                    help='stride of the slidding window (seconds)')
# Logging
parser.add_argument('--logfile', type=Path, default='', help='Logging file')

args = parser.parse_args()


def main(args):
    setup_logging(args)
    logging.info('Setting-up datasets')
    train_dataset, test_dataset = setup_dataset(args)
    logging.info('Estimating prior')
    moment_freq_prior = DiscreteFrequencyPrior()
    # "training"
    for i, data in enumerate(train_dataset):
        gt_segments, pred_segments = data[-2:]
        moment_freq_prior.update(gt_segments, pred_segments)
    moment_freq_prior.fit()

    # testing
    logging.info(f'* Evaluation')
    pred_segments = torch.from_numpy(moment_freq_prior.predict())
    meters = Multimeter(keys=METRICS)
    for i, data in enumerate(test_dataset):
        gt_segments = torch.from_numpy(data[-2])
        hit_k_iou = single_moment_retrieval(
            gt_segments, pred_segments, TOPK_IOU_POINTS)
        meters.update([i.item() for i in hit_k_iou])
    logging.info(f'{meters.report()}')


def setup_dataset(args):
    "Setup dataset and loader"
    proposal_generator = proposals.SlidingWindowMSFS(
        args.min_length, args.num_scales, args.stride, unique=True)
    subset_files = [('train', args.train_list), ('test', args.test_list)]
    cues = {args.feat: {'file': args.h5_path}}
    datasets = []
    for i, (subset, filename) in enumerate(subset_files):
        datasets.append(
            dataset_untrimmed.UntrimmedMCN(
                filename, eval=True, proposals_interface=proposal_generator,
                # we don't care about language only visual for lazy reasons
                debug=True, cues=cues)
        )
    return datasets


class DiscreteFrequencyPrior():
    "Compute the frequency prior of segments in dataset"

    def __init__(self):
        self.table = {}
        self.num_segments = None
        self.frequency = None
        self.segments = None

    def update(self, gt_segments, pred_segments):
        "Count how often a given segment appears"
        # TODO: use a soft-criteria such as IOU over pred_segments
        ind = np.unique(
            segment_iou(pred_segments, gt_segments).argmax(axis=0))
        gt_segments_mapped_to_pred = pred_segments[ind, :]
        assert len(gt_segments_mapped_to_pred.shape) == 2
        for segment in gt_segments_mapped_to_pred:
            hashable_segment = tuple(segment.tolist())
            if hashable_segment not in self.table:
                self.table[hashable_segment] = 0
            self.table[hashable_segment] += 1

    def fit(self):
        "Sort table"
        assert len(self.table) > 0
        self.num_segments = len(self.table)
        segments = np.array(list(self.table.keys()))
        frequency = np.array(list(self.table.values()))
        ind = np.argsort(frequency)
        self.frequency = frequency[ind]
        self.segments = segments[ind, :].astype(np.float32)

    def predict(self):
        return self.segments


if __name__ == '__main__':
    main(args)
