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
from evaluation import single_moment_retrieval, didemo_evaluation
from np_segments_ops import non_maxima_suppresion
from utils import setup_logging
from utils import Multimeter

# TODO(tier-2;release): mirror this constants from an init file
TOPK, IOU_THRESHOLDS = (1, 5), (0.5, 0.7)
METRICS = [f'r@{k},{iou}' for iou, k in product(IOU_THRESHOLDS, TOPK)]
METRICS_OLD = ['iou', 'r@1', 'r@5']

parser = argparse.ArgumentParser(description='Frequency prior')
parser.add_argument('--train-list', type=Path, required=True,
                    help='JSON-file with training instances')
parser.add_argument('--test-list', type=Path, required=True,
                    help='JSON-file with training instances')
# Freq prior parameters
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--kde', action='store_true',
                   help='Perform continous analysis')
group.add_argument('--bins', type=int,
                   help=('Number of bins for discretization. Please provide '
                         'something for DiDeMo, but would be ignored in favor'
                         'of the ICCV-2017 bin edges.')
                  )
group_edges = group.add_argument_group(description='Bins for discretization')
group_edges.add_argument('--ts-edges', type=float, nargs='+',
                         help='Bin edges for t-start')
group_edges.add_argument('--te-edges', type=float, nargs='+',
                         help='Bin edges for t-end')
# Hyper-parameters to explore search space (inference)
parser.add_argument('--proposal-interface', default='SlidingWindowMSFS',
                    choices=proposals.PROPOSAL_SCHEMES,
                    help='Type of proposals spanning search space')
parser.add_argument('--min-length', type=float, default=3,
                    help='Minimum length of slidding windows (seconds)')
parser.add_argument('--num-scales', type=int, default=8,
                    help='Number of scales in a multi-scale linear slidding '
                         'window')
parser.add_argument('--stride', type=float, default=3,
                    help='stride of the slidding window (seconds)')
parser.add_argument('--nms-threshold', type=float, default=0.6,
                    help=('Threshold used to remove overlapped predictions'
                          'We use 1.0 in DiDeMo for fair comparsion.'
                          'Moreover the evaluation code also assumes that.')
                    )
# Logging
parser.add_argument('--logfile', type=Path, default='', help='Logging file')

args = parser.parse_args()


def main(args):
    "Glue all the pieces to estimate the moment frequency prior"
    global TOPK
    TOPK = torch.tensor(TOPK)
    TOPK_ = TOPK.float()
    setup_logging(args)
    logging.info('Moment frequency prior')
    logging.info(args)
    logging.info('Setting-up datasets')
    train_dataset, test_dataset = setup_dataset(args)
    if args.kde:
        moment_freq_prior = KDEFrequencyPrior()
    else:
        if args.proposal_interface == 'DidemoICCV17SS':
            logging.info('Ignoring bins argument')
            args.ts_edges = np.arange(0, 31, 5.0) / 30
            args.te_edges = np.arange(5, 36, 5.0) / 30
        if args.ts_edges is not None:
            args.bins = [args.ts_edges, args.te_edges]
        moment_freq_prior = DiscretizedFrequencyPrior(args.bins)

    logging.info('Estimating prior')
    for i, data in enumerate(train_dataset):
        gt_moments = data[-2]
        duration_i = video_duration(train_dataset, i)
        moment_freq_prior.update(gt_moments, duration_i)
    logging.info('Model fitting')
    moment_freq_prior.fit()

    logging.info(f'* Evaluation')
    meters, meters_old = Multimeter(keys=METRICS), None
    if args.proposal_interface == 'DidemoICCV17SS':
        meters_old = Multimeter(keys=METRICS_OLD)
        meters_old_ = Multimeter(keys=METRICS_OLD)
        # Details are provided in help
        args.nms_threshold = 1.0
    for i, data in enumerate(test_dataset):
        duration_i = video_duration(test_dataset, i)
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
        if meters_old:
            iou_r_at_ks = didemo_evaluation(
                gt_moments, sorted_proposals, TOPK_)
            meters_old.update([i.item() for i in iou_r_at_ks])

    logging.info(f'{meters.report()}')
    if meters_old:
        logging.info(f'{meters_old.report()}')


def video_duration(dataset, moment_index):
    "Return duration of video of a given moment"
    video_id = dataset.metadata[moment_index]['video']
    return dataset.metadata_per_video[video_id]['duration']


def setup_dataset(args):
    "Setup dataset and loader"
    proposal_generator = proposals.__dict__[args.proposal_interface](
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
        "Edges have priority over bins"
        super(DiscretizedFrequencyPrior, self).__init__()
        self._x_edges, self._y_edges = None, None
        self.bins = bins

    def fit(self):
        "Make 2D histogram"
        table_np = np.row_stack(self.table)
        self.model, self._x_edges, self._y_edges = np.histogram2d(
            table_np[:, 0], table_np[:, 1], bins=self.bins)

    def predict(self, pred_segments=None, duration=None):
        "Return prob that a proposal belongs to the dataset"
        assert self.model is not None
        normalized_proposals = pred_segments / duration        
        ind_x = np.digitize(normalized_proposals[:, 0], self._x_edges, True)
        ind_y = np.digitize(normalized_proposals[:, 1], self._y_edges, True)
        ind_x = np.clip(ind_x, 0, self.model.shape[0] - 1)
        ind_y = np.clip(ind_y, 0, self.model.shape[1] - 1)
        return self.model[ind_x, ind_y]


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
