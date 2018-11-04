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
from np_segments_ops import non_maxima_suppresion
from utils import setup_logging
from utils import Multimeter

# TODO(tier-2;release): mirror this constants from an init file
TOPK, IOU_THRESHOLDS = (1, 5), (0.5, 0.7)
METRICS = [f'r@{k},{iou}' for iou, k in product(IOU_THRESHOLDS, TOPK)]

parser = argparse.ArgumentParser(description='Frequency prior')
parser.add_argument('--train-list', type=Path, default='non-existent',
                    help='JSON-file with training instances')
parser.add_argument('--test-list', type=Path, default='non-existent',
                    help='JSON-file with training instances')
# Freq prior parameters
parser.add_argument('--kde', action='store_true',
                    help='Perform continous analysis')
parser.add_argument('--bins', type=int, default=10,
                    help='Number of bins for discretization')
# Hyper-parameters to explore search space (inference)
parser.add_argument('--min-length', type=float, default=3,
                    help='Minimum length of slidding windows (seconds)')
parser.add_argument('--num-scales', type=int, default=8,
                    help='Number of scales in a multi-scale linear slidding '
                         'window')
parser.add_argument('--stride', type=float, default=3,
                    help='stride of the slidding window (seconds)')
parser.add_argument('--nms-threshold', type=float, default=0.6)
# Logging
parser.add_argument('--logfile', type=Path, default='', help='Logging file')

args = parser.parse_args()


def main(args):
    "Glue all the pieces to estimate the moment frequency prior"
    global TOPK
    TOPK = torch.tensor(TOPK)
    setup_logging(args)
    logging.info('Moment frequency prior')
    logging.info(args)
    logging.info('Setting-up datasets')
    train_dataset, test_dataset = setup_dataset(args)
    if args.kde:
        moment_freq_prior = KDEFrequencyPrior()
    else:
        moment_freq_prior = DiscretizedFrequencyPrior(args.bins)

    logging.info('Estimating prior')
    for i, data in enumerate(train_dataset):
        gt_moments = data[-2]
        duration_i = video_duration(train_dataset, i)
        moment_freq_prior.update(gt_moments, duration_i)
    logging.info('Model fitting')
    moment_freq_prior.fit()

    logging.info(f'* Evaluation')
    meters = Multimeter(keys=METRICS)
    for i, data in enumerate(test_dataset):
        duration_i = video_duration(train_dataset, i)
        gt_moments = torch.from_numpy(data[-2])
        proposals_i = data[-1]
        prob = moment_freq_prior.predict(proposals_i, duration_i)
        if args.nms_threshold < 1:
            ind = non_maxima_suppresion(
                proposals_i, prob, args.nms_threshold)
        else:
            ind = prob.argsort()[::-1]
        sorted_proposals = proposals_i[ind, :]
        sorted_proposals = torch.from_numpy(sorted_proposals)
        hit_k_iou = single_moment_retrieval(
            gt_moments, sorted_proposals, topk=TOPK)
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
    lastname = 'kde' if args.kde else 'discrete'
    cues = {f'mfp-{lastname}': None}
    subset_files = [('train', args.train_list), ('test', args.test_list)]
    datasets = []
    for i, (subset, filename) in enumerate(subset_files):
        datasets.append(
            dataset_untrimmed.UntrimmedMCN(
                filename, proposals_interface=proposal_generator,
                # we don't care about language only visual and there is not
                # distintion btw eval or train mode
                eval=True, debug=True, cues=cues, no_visual=True, loc=True)
        )
    return datasets


class BaseFrequencyPrior():
    "Compute the frequency prior of segments in dataset"

    def __init__(self):
        self.table = []
        self.model = None

    def update(self, gt_segments, duration):
        "Count how often a given segment appears"
        self.table.append(gt_segments / duration)


class DiscretizedFrequencyPrior(BaseFrequencyPrior):
    "Estimate frequency prior of segments in dataset via discretization"

    def __init__(self, bins):
        super(DiscretizedFrequencyPrior, self).__init__()
        self.bins = bins
        self._x_edges, self._y_edges = None, None

    def fit(self):
        "Make 2D histogram"
        table_np = np.row_stack(self.table)
        self.model, self._x_edges, self._y_edges = np.histogram2d(
            table_np[:, 0], table_np[:, 1], bins=self.bins, normed=True)

    def predict(self, pred_segments=None, duration=None):
        "Return prob that a proposal belongs to the dataset"
        assert self.model is not None
        normalized_proposals = pred_segments / duration
        ind_x = self._search(self._x_edges, normalized_proposals[:, 0])
        ind_y = self._search(self._y_edges, normalized_proposals[:, 1])
        return self.model[ind_x, ind_y]

    def _search(self, x, y):
        ind = np.searchsorted(x, y) - 1
        return np.clip(ind, 0, len(x) - 2)


class KDEFrequencyPrior(BaseFrequencyPrior):
    "Estimate frequency prior of segments in dataset via KDE"

    def fit(self):
        "Fit PDF with KDE and default bandwidth selection rule"
        all_segments = np.row_stack(self.table).T
        self.model = gaussian_kde(all_segments)

    def predict(self, pred_segments=None, duration=None):
        "Return prob that a proposal belongs to the dataset"
        assert self.model is not None
        normalized_proposal = (pred_segments / duration).T
        return self.model.pdf(normalized_proposal)


if __name__ == '__main__':
    main(args)
