"""Frequency prior baseline

Huge thanks to @ModarTensai for a helpful discussion that elucidated the
procedure for the KDE approach.

TODO?
1. Implement sample with replacement, possibly applying NMS, in between. Such
   that we sample with "diversity" instead od returning the sorted segments.
"""
import argparse
import logging
from itertools import product
from pathlib import Path

import numpy as np
import torch
from scipy.stats import gaussian_kde

import dataset_untrimmed
import proposals
from evaluation import single_moment_retrieval
from np_segments_ops import iou as segment_iou
from np_segments_ops import non_maxima_suppresion
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
# Type pf freq prior approximation
parser.add_argument('--kde', action='store_true',
                    help='Perform continous analysis')
# Features
parser.add_argument('--feat', default='tef',
                    help='Record the type of feature used (modality)')
# Hyper-parameters to explore search space (inference)
parser.add_argument('--min-length', type=float, default=3,
                    help='Minimum length of slidding windows (seconds)')
parser.add_argument('--num-scales', type=int, default=8,
                    help='Number of scales in a multi-scale linear slidding '
                         'window')
parser.add_argument('--stride', type=float, default=3,
                    help='stride of the slidding window (seconds)')
parser.add_argument('--nms-threshold', type=float, default=0.5)
# Logging
parser.add_argument('--logfile', type=Path, default='', help='Logging file')

args = parser.parse_args()


def main(args):
    setup_logging(args)
    logging.info('Moment frequency prior')
    logging.info(args)
    logging.info('Setting-up datasets')
    train_dataset, test_dataset = setup_dataset(args)
    logging.info('Estimating prior')
    moment_freq_prior = DiscreteFrequencyPrior()
    if args.kde:
        moment_freq_prior = KDEFrequencyPrior()
    # "training"
    for i, data in enumerate(train_dataset):
        gt_segments, pred_segments = data[-2:]
        duration_i = video_duration(train_dataset, i)
        moment_freq_prior.update(gt_segments, pred_segments, duration_i)
    logging.info('Model fitting')
    moment_freq_prior.fit()

    # testing
    logging.info(f'* Evaluation')
    if not args.kde:
        pred_segments = torch.from_numpy(moment_freq_prior.predict())
    meters = Multimeter(keys=METRICS)
    for i, data in enumerate(test_dataset):
        duration_i = video_duration(train_dataset, i)
        gt_segments = torch.from_numpy(data[-2])
        if args.kde:
            pred_segments_ = data[-1]
            prob = moment_freq_prior.predict(pred_segments_, duration_i)
            if args.nms_threshold < 1:
                ind = non_maxima_suppresion(
                    pred_segments_, prob, args.nms_threshold)
            else:
                ind = prob.argsort()[::-1]
            pred_segments = torch.from_numpy(pred_segments_[ind, :])
        hit_k_iou = single_moment_retrieval(
            gt_segments, pred_segments, TOPK_IOU_POINTS)
        meters.update([i.item() for i in hit_k_iou])
    logging.info(f'{meters.report()}')


def video_duration(dataset, moment_index):
    "Return duration of video of a given moment"
    video_id = dataset.metadata[moment_index]['video']
    return dataset.metadata_per_video[video_id]['duration']


def setup_dataset(args):
    "Setup dataset and loader"
    proposal_generator = proposals.SlidingWindowMSFS(
        args.min_length, args.num_scales, args.stride, unique=True)
    subset_files = [('train', args.train_list), ('test', args.test_list)]
    datasets = []
    for i, (subset, filename) in enumerate(subset_files):
        datasets.append(
            dataset_untrimmed.UntrimmedMCN(
                filename, proposals_interface=proposal_generator,
                # we don't care about language only visual for lazy reasons
                # similarly we don't care about eval or train mode
                eval=True, debug=True, cues=None, no_visual=True, loc=True)
        )
    return datasets


class DiscreteFrequencyPrior():
    "Compute the frequency prior of segments in dataset"

    def __init__(self):
        self.table = {}
        self.num_segments = None
        self.frequency = None
        self.segments = None

    def update(self, gt_segments, pred_segments, duration=None):
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

    def predict(self, pred_segments=None, duration=None):
        return self.segments


class KDEFrequencyPrior():
    "Compute the frequency prior of segments in dataset"

    def __init__(self):
        self.table = []
        self.num_segments = None
        self.model = None

    def update(self, gt_segments, pred_segments=None, duration=None):
        "Count how often a given segment appears"
        assert duration is not None
        self.table.append(gt_segments / duration)

    def fit(self):
        "Sort table"
        all_segments = np.row_stack(self.table).T
        self.model = gaussian_kde(all_segments)

    def predict(self, pred_segments=None, duration=None):
        assert self.model is not None
        normalized_segments = (pred_segments / duration).T
        return self.model.pdf(normalized_segments)


if __name__ == '__main__':
    main(args)
