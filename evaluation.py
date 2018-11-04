"Moment retrieval evaluation in a single video or a video corpus"
import json
from itertools import product

import numpy as np
import torch

from corpus import Corpus, CorpusAsDistanceMatrix
from dataset import Queries
from np_segments_ops import iou as numpy_iou
from np_segments_ops import torch_iou

IOU_THRESHOLDS = (0.5, 0.7)
TOPK = (1, 5)
DEFAULT_TOPK_AND_IOUTHRESHOLDS = tuple(product(TOPK, IOU_THRESHOLDS))


def iou(gt, pred):
    "Taken from @LisaAnne/LocalizingMoments project"
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return intersection / union


def rank(gt, pred):
    "Taken from @LisaAnne/LocalizingMoments project"
    if isinstance(pred[0], tuple) and not isinstance(gt, tuple):
        gt = tuple(gt)
    return pred.index(gt) + 1


def video_evaluation(gt, predictions, k=(1, 5)):
    "Single video moment retrieval evaluation with Python data-structures"
    top_segment = predictions[0]
    ious = [iou(top_segment, s) for s in gt]
    average_iou = np.mean(np.sort(ious)[-3:])
    ranks = [rank(s, predictions) for s in gt]
    average_ranks = np.mean(np.sort(ranks)[:3])
    r_at_k = [average_ranks <= i for i in k]
    return [average_iou] + r_at_k


def didemo_evaluation(true_segments, pred_segments, topk):
    """DiDeMo single video moment retrieval evaluation in torch

    Args:
        true_segments (tensor): of shape [N, 2]
        pred_segments (tensor): of shape [M, 2] with sorted predictions
        topk (float tensor): of shape [k] with all the ranks to compute.
    """
    result = torch.empty(1 + len(topk), dtype=true_segments.dtype,
                         device=true_segments.device)
    iou_matrix = torch_iou(pred_segments, true_segments)
    average_iou = iou_matrix[0, :].sort()[0][-3:].mean()
    ranks = ((iou_matrix == 1).nonzero()[:, 0] + 1).float()
    average_rank = ranks.sort()[0][:3].mean().to(topk)
    rank_at_k = average_rank <= topk

    result[0] = average_iou
    result[1:] = rank_at_k
    return result


def single_moment_retrieval(true_segments, pred_segments,
                            iou_thresholds=IOU_THRESHOLDS,
                            topk=torch.tensor(TOPK)):
    """Compute if a given segment is retrieved in top-k for a given IOU

    Args:
        true_segments (torch tensor) : shape [N, 2] holding 1 segments.
        pred_segments (torch tensor) : shape [M, 2] holding M segments sorted
            by their scores. pred_segment[0, :] is the most confident segment
            retrieved.
        iou_thresholds (sequence) : of iou thresholds of length [P] that
            determine if a prediction matched a true segment.
        topk (torch tensor) : shape [Q] holding ranks to consider.
    Returns:
        torch tensor of shape [P * Q] indicate if the true segment was found
        for a given pair of top-k,iou. The first Q elements results correspond
        to all the topk for the iou[0] and so on.
    """
    concensus_among_annotators = 1 if len(true_segments) == 1 else 2
    P, Q = len(iou_thresholds), len(topk)
    iou_matrix = torch_iou(pred_segments, true_segments)
    # TODO: check type
    hit_k_iou = torch.empty(P * Q, dtype=torch.uint8,
                            device=iou_matrix.device)
    for i, threshold in enumerate(iou_thresholds):
        hit_iou = ((iou_matrix >= threshold).sum(dim=1) >=
                    concensus_among_annotators)
        rank_iou = (hit_iou != 0).nonzero()
        if len(rank_iou) == 0:
            hit_k_iou[i * Q:(i + 1) * Q] = 0
        else:
            # 0-indexed -> +1
            hit_k_iou[i * Q:(i + 1) * Q] = topk >= (rank_iou[0] + 1)
    return hit_k_iou


class CorpusVideoMomentRetrievalEval():
    """Evaluation of moments retrieval from a video corpus

    Note:
        - method:evaluate change the type of `_rank_iou` and `_hit_iou_k` to
          block the addition of new instances. This also ease retrieval of per
          query metrics.
    """

    def __init__(self, topk=(1, 5), iou_thresholds=IOU_THRESHOLDS):
        self.topk = topk
        # 0-indexed rank
        self.topk_ = torch.tensor(topk) - 1
        self.iou_thresholds = iou_thresholds
        self.map_query = {}
        self._rank_iou = [[] for i in iou_thresholds]
        self._hit_iou_k = [[] for i in iou_thresholds]
        self.medrank_iou = [None] * len(iou_thresholds)
        self.stdrank_iou = [None] * len(iou_thresholds)
        self.recall_iou_k = [None] * len(iou_thresholds)
        self.performance = None

    def add_single_predicted_moment_info(self, query_info, video_indices,
                                         pred_segments):
        """Compute rank@IOU and hit@IOU,k per query

        Args:
            query_info (dict) : metadata about the moment. Mandatory
                key:values are a unique hashable `annotation_id`; an integer
                of the `video_index`; numpy array of shape [N x 2] with the
                ground-truth `times`.
            video_indices (torch tensor) : ranked vector of shape [M]
                representing videos associated with query_info.
            pred_segments (torch tensor) : ranked segments of shape [M x 2]
                representing segments inside the videos associated with
                query_info.
        """
        query_id = query_info.get('annotation_id')
        if query_id in self.map_query:
            raise ValueError('query was already added to the list')
        ind = len(self._rank_iou)
        true_video = query_info.get('video_index')
        true_segments = query_info.get('times')
        concensus_among_annotators = 1
        if true_segments.shape[0] > 1:
            concensus_among_annotators = 2

        true_segments = torch.from_numpy(true_segments)
        hit_video = video_indices == true_video
        iou_matrix = torch_iou(pred_segments, true_segments)
        for i, iou_threshold in enumerate(self.iou_thresholds):
            hit_segment = ((iou_matrix >= iou_threshold).sum(dim=1) >=
                           concensus_among_annotators)
            hit_iou = hit_segment & hit_video
            rank_iou = (hit_iou != 0).nonzero()
            if len(rank_iou) == 0:
                # shall we use nan? study localization vs retrieval errors
                rank_iou = torch.tensor(
                    [len(pred_segments)], device=video_indices.device)
            else:
                rank_iou = rank_iou[0]
            self._rank_iou[i].append(rank_iou)
            self._hit_iou_k[i].append(self.topk_ >= rank_iou)
        # Lock query-id and record index for debugging
        self.map_query[query_id] = ind

    def evaluate(self):
        "Compute MedRank@IOU and R@IOU,k accross the dataset"
        if self.performance is not None:
            return self.performance
        num_queries = len(self.map_query)
        for i, _ in enumerate(self.iou_thresholds):
            self._rank_iou[i] = torch.cat(self._rank_iou[i])
            # report results as 1-indexed ranks for humans
            ranks_i = self._rank_iou[i] + 1
            self.medrank_iou[i] = torch.median(ranks_i)
            self.stdrank_iou[i] = torch.std(ranks_i.float())
            self._hit_iou_k[i] = torch.stack(self._hit_iou_k[i])
            recall_i = self._hit_iou_k[i].sum(dim=0).float() / num_queries
            self.recall_iou_k[i] = recall_i
        self._rank_iou = torch.stack(self._rank_iou).transpose_(0, 1)
        self._hit_iou_k = torch.stack(self._hit_iou_k).transpose_(0, 1)
        self._consolidate_results()
        return self.performance

    def _consolidate_results(self):
        "Create dict with all the metrics"
        self.performance = {}
        for i, iou in enumerate(self.iou_thresholds):
            self.performance[f'MedRank@{iou}'] = self.medrank_iou[i]
            self.performance[f'StdRand@{iou}'] = self.stdrank_iou[i]
            for j, topk in enumerate(self.topk):
                self.performance[f'Recall@{topk},{iou}'] = (
                    self.recall_iou_k[i][j])


class RetrievalEvaluation():
    """First implementation used to evaluate corpus video moment retrieval

    DEPRECATED

    TODO: count search for a given query_id
    """
    # to run the evaluation again
    # _judge.reset()

    def __init__(self, corpus_h5, groundtruth_json,
                 k=(1,), iou_threshold=0.75,
                 nms_threshold=1.0, topk=None):
        self.corpus = Corpus(corpus_h5, nms_threshold=nms_threshold, topk=topk)
        videos = self.corpus.videos.tolist()
        segments = list(map(tuple, self.corpus.segments.tolist()))
        self._precompute_iou()
        self.gt_queries = Queries(groundtruth_json,
                                  videos=videos, segments=segments)
        self.k = k
        self.iou_threshold = iou_threshold
        self.reset()
        self._k = np.array(self.k)

    def eval(self, full=False):
        recall_k = [sum(i) / len(i) for i in self.hit_k]
        recall_k_iou = [sum(i) / len(i) for i in self.hit_k_iou]
        miou = sum(self.miou) / len(self.miou)
        mrank = np.mean(self.rank)
        self.avg_rank = np.array(self.avg_rank)
        recall_k_didemo = [np.sum(self.avg_rank <= k) / len(self.avg_rank)
                           for k in self.k]
        if full:
            return recall_k, recall_k_iou, recall_k_didemo, miou, mrank
        return recall_k, mrank

    def eval_single_query(self, query, query_id):
        # todo encode language vector
        raise NotImplementedError

    def eval_single_vector(self, vector, query_id):
        if query_id not in self.gt_queries.data:
            # log-this
            pass
        # corpus indices are used for original didemo evaluation
        (pred_video_indices, pred_segment_indices,
         sorted_dist) = self.corpus.index(vector)
        pred_corpus_indices = self.corpus.repo_to_ind(
            pred_video_indices, pred_segment_indices)
        true_video = self.gt_queries[query_id]['video_index']
        true_segments = self.gt_queries[query_id]['segment_indices']
        true_corpus_indices = self.corpus.repo_to_ind(
            true_video, true_segments)


        # Q&D -> tp_fp_segments
        # Note: I keep the entire corpus to compute rank
        # Note: np.in1d won't generalize for other criteria
        # TODO-p: check if bottleneck. In theory, using a loop
        #         and break for fp from other videos sounds efficient.
        tp_fp_videos = pred_video_indices == true_video
        tp_fp_segments = np.in1d(pred_segment_indices, true_segments)
        tp_fp_labels = np.logical_and(tp_fp_videos, tp_fp_segments)
        for i, k in enumerate(self.k):
            self.hit_k[i].append(tp_fp_labels[:k].sum(dtype=bool))
        self.rank.append(np.where(tp_fp_labels)[0].min())

        # Q&D -> mIOU
        self.miou.append(0)
        if tp_fp_labels[0]:
            ious = self.iou_matrix[true_segments, pred_segment_indices[0]]
            self.miou[-1] = np.mean(np.sort(ious)[-3:])

        # Q&D -> R@k,tIOU
        max_k = max(self.k)
        topk_tp_fp_videos = tp_fp_videos[:max_k]
        topk_pred_segment_indices = pred_segment_indices[:max_k]
        i, j = np.meshgrid(true_segments, topk_pred_segment_indices)
        iou = np.max(topk_tp_fp_videos[:, None] * self.iou_matrix[i, j],
                     axis=1)
        topk_tp_fp_labels = iou >= self.iou_threshold
        cum_tp_fp_labels = topk_tp_fp_labels.cumsum()
        for i, k in enumerate(self.k):
            self.hit_k_iou[i].append(cum_tp_fp_labels[k - 1] > 0)
            # debugging
            # if self.iou_threshold == 1:
            #     if self.hit_k[i][-1] != self.hit_k_iou[i][-1]:
            #         raise

        # Q&D -> R@k (Original didemo)
        # no joda, I could find the stupid function for this
        pred_rank = [np.argwhere(pred_corpus_indices == i)
                     for i in true_corpus_indices]
        self.avg_rank.append(np.mean(np.sort(pred_rank)[:3]))
        # debugging
        # assert np.all(pred_corpus_indices < len(self.corpus))
        # assert np.all(true_corpus_indices < len(self.corpus))

    def reset(self):
        self.hit_k = [[] for i in self.k]
        self.hit_k_iou = [[] for i in self.k]
        self.miou = []
        self.rank = []
        self.avg_rank = []

    def _precompute_iou(self):
        segments = self.corpus.segments * 5
        segments[:, 1] += 5
        self.iou_matrix = numpy_iou(segments, segments)


class CorpusVideoMomentRetrievalEvalFromMatrix():
    """Helper for Corpus video moment retrieval from distance matrix

    DEPRECATED. Replaced by CorpusVideoMomentRetrievalEval.

    TODO: refactor to avoid code-duplication
    """

    def __init__(self, groundtruth_json, matrix_h5,
                 k=(1,), iou_threshold=0.75,
                 nms_threshold=1.0, topk=None):
        self.corpus = CorpusAsDistanceMatrix(
            matrix_h5, nms_threshold=nms_threshold, topk=topk)
        self._load_groundtruth(groundtruth_json)
        self._precompute_iou()
        self.k = k
        self.iou_threshold = iou_threshold
        self.reset()
        self._k = np.array(self.k)

    def _load_groundtruth(self, filename):
        with open(filename, 'r') as f:
            self.ground_truth = {}
            for moment in json.load(f):
                query_id = moment['annotation_id']
                moment['video_index'] = self.corpus.video_to_id(
                    moment['video'])
                moment['segment_indices'] = np.array(
                    [self.corpus.segment_to_ind(i) for i in moment['times']])
                self.ground_truth[query_id] = moment

    def eval(self, full=False):
        for query_id in self.ground_truth:
            self.eval_single_description(query_id)
        return self.summary(full)

    def summary(self, full=False):
        recall_k = [sum(i) / len(i) for i in self.hit_k]
        recall_k_iou = [sum(i) / len(i) for i in self.hit_k_iou]
        miou = sum(self.miou) / len(self.miou)
        mrank = np.mean(self.rank)
        self.avg_rank = np.array(self.avg_rank)
        recall_k_didemo = [np.sum(self.avg_rank <= k) / len(self.avg_rank)
                           for k in self.k]
        if full:
            return recall_k, recall_k_iou, recall_k_didemo, miou, mrank
        return recall_k, mrank

    def eval_single_description(self, query_id):
        if query_id not in self.corpus.moments_id2ind:
            raise
        # corpus indices are used for original didemo evaluation
        (pred_video_indices, pred_segment_indices,
         sorted_dist) = self.corpus.index(query_id)
        pred_corpus_indices = self.corpus.repo_to_ind(
            pred_video_indices, pred_segment_indices)
        true_video = self.ground_truth[query_id]['video_index']
        true_segments = self.ground_truth[query_id]['segment_indices']
        true_corpus_indices = self.corpus.repo_to_ind(
            true_video, true_segments)

        # Q&D -> tp_fp_segments
        # Note: I keep the entire corpus to compute rank
        # Note: np.in1d won't generalize for other criteria
        # TODO-p: check if bottleneck. In theory, using a loop
        #         and break for fp from other videos sounds efficient.
        tp_fp_videos = pred_video_indices == true_video
        tp_fp_segments = np.in1d(pred_segment_indices, true_segments)
        tp_fp_labels = np.logical_and(tp_fp_videos, tp_fp_segments)
        for i, k in enumerate(self.k):
            self.hit_k[i].append(tp_fp_labels[:k].sum(dtype=bool))
        self.rank.append(np.where(tp_fp_labels)[0].min())

        # Q&D -> mIOU
        self.miou.append(0)
        if tp_fp_labels[0]:
            ious = self.iou_matrix[true_segments, pred_segment_indices[0]]
            self.miou[-1] = np.mean(np.sort(ious)[-3:])

        # Q&D -> R@k,tIOU
        max_k = max(self.k)
        topk_tp_fp_videos = tp_fp_videos[:max_k]
        topk_pred_segment_indices = pred_segment_indices[:max_k]
        i, j = np.meshgrid(true_segments, topk_pred_segment_indices)
        iou = np.max(topk_tp_fp_videos[:, None] * self.iou_matrix[i, j],
                     axis=1)
        topk_tp_fp_labels = iou >= self.iou_threshold
        cum_tp_fp_labels = topk_tp_fp_labels.cumsum()
        for i, k in enumerate(self.k):
            self.hit_k_iou[i].append(cum_tp_fp_labels[k - 1] > 0)

        # Q&D -> R@k (Original didemo)
        # no joda, I could find the stupid function for this
        pred_rank = [np.argwhere(pred_corpus_indices == i)
                     for i in true_corpus_indices]
        self.avg_rank.append(np.mean(np.sort(pred_rank)[:3]))

    def reset(self):
        self.hit_k = [[] for i in self.k]
        self.hit_k_iou = [[] for i in self.k]
        self.miou = []
        self.rank = []
        self.avg_rank = []

    def _precompute_iou(self):
        segments = self.corpus.segments_time
        self.iou_matrix = numpy_iou(segments, segments)


if __name__ == '__main__':
    import argparse
    import logging
    from datetime import datetime
    from pathlib import Path

    from tqdm import tqdm

    import corpus
    import dataset_untrimmed
    import model
    import proposals
    from utils import setup_logging

    parser = argparse.ArgumentParser(
        description='Corpus Retrieval Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    parser.add_argument('--test-list', type=Path, required=True,
                        help='JSON-file with corpus instances')
    parser.add_argument('--h5-path', type=Path, required=True,
                        help='HDF5-file with features')
    # Architecture
    parser.add_argument('--snapshot', type=Path, required=True,
                        help='pht.tar file with model parameters')
    parser.add_argument('--snapshot-args', type=Path, required=True,
                        help='JSON-file file with model parameters')
    # Evaluation parameters
    parser.add_argument('--topk', nargs='+', type=int,
                        default=[1, 10, 100, 1000, 10000],
                        help='top-k values to compute')
    # Extras
    parser.add_argument('--arch', choices=model.MOMENT_RETRIEVAL_MODELS,
                        default='MCN',
                        help='model architecture, only for old JSON files')
    parser.add_argument('--dataset', choices=model.MOMENT_RETRIEVAL_MODELS,
                        default='UntrimmedMCN',
                        help='model architecture, only for old JSON files')
    parser.add_argument('--engine', choices=model.MOMENT_RETRIEVAL_MODELS,
                        default='MomentRetrievalFromProposalsTable',
                        help='Type of engine')
    parser.add_argument('--proposal-interface', default='SlidingWindowMSFS',
                        choices=proposals.PROPOSAL_SCHEMES,
                        help='Type of proposals spanning search space')
    # Dump results and logs
    parser.add_argument('--dump', action='store_true',
                        help='Save log in text file and json')
    # Debug
    parser.add_argument('--debug', action='store_true',
                    help=('yield incorrect results! to verify we are gluing '
                          'things (dataset, model, eval) correctly'))
    args = parser.parse_args()
    args.logfile = Path('')
    args.disable_tqdm = False
    if args.dump:
        args.disable_tqdm = True
        args.logfile = args.snapshot_args.with_suffix('').with_name(
            args.snapshot_args.stem +
            f'_corpus-eval_{args.test_list.stem}.log')
    setup_logging(args)

    logging.info('Corpus Retrieval Evaluation for MCN')
    logging.info(args)
    logging.info('Parsing JSON file with hyper-parameters')
    with open(args.snapshot_args, 'r') as fid:
        model_hp = json.load(fid)
        if model_hp.get('arch') is None:
            logging.warning(f'Old JSON-file. Using `arch`: {args.arch}')
            model_hp['arch'] = args.arch
        args.arch = model_hp['arch']

    if args.arch == 'MCN':
        args.dataset = 'UntrimmedMCN'
        args.engine = 'MomentRetrievalFromProposalsTable'
    elif args.arch == 'SMCN':
        args.dataset = 'UntrimmedSMCN'
        args.engine = 'MomentRetrievalFromClipBasedProposalsTable'
    else:
        logging.warning('Using `dataset` and `engine` classes given by user')

    logging.info('Loading dataset')
    dataset_setup = dict(
        json_file=args.test_list,
        cues={model_hp['feat']: {'file': args.h5_path}},
        loc=model_hp['loc'],
        context=model_hp['context'],
        debug=args.debug,
        eval=True,
        proposals_interface=proposals.__dict__[args.proposal_interface](
            length=model_hp.get('min_length'),
            num_scales=model_hp.get('num_scales'),
            stride=model_hp.get('stride'),
            unique=True
        )
    )
    dataset = dataset_untrimmed.__dict__[args.dataset](**dataset_setup)
    if args.arch == 'SMCN':
        dataset.set_padding(False)

    logging.info('Setting up model')
    if model_hp.get('lang_hidden') is None:
        # Fallback option when we weren't recording arch hyper-parameters
        # TODO(tier-2;cleanup;release): move this to model or utils
        logging.warning('Inferring model hyper-parameters from snapshot')
        weights = torch.load(args.snapshot)['state_dict']
        model_hp['embedding_size'] = weights['lang_encoder.bias'].shape[0]
        model_hp['visual_hidden'] = weights['visual_encoder.0.bias'].shape[0]
        # Assumed that we were using LSTM
        model_hp['lang_hidden'] = weights[
            'sentence_encoder.weight_ih_l0'].shape[0] // 4
        num_layers = sum([1 for layer_name in weights.keys()
                          if layer_name.startswith('visual_encoder')]) // 2
        model_hp['visual_layers'] = num_layers - 1
    arch_setup = dict(
        visual_size=dataset.visual_size['rgb'],
        lang_size=dataset.language_size,
        max_length=dataset.max_words,
        embedding_size=model_hp['embedding_size'],
        visual_hidden=model_hp['visual_hidden'],
        lang_hidden=model_hp['lang_hidden'],
        visual_layers=model_hp['visual_layers'],
    )
    models_dict = {model_hp['feat']: model.__dict__[args.arch](**arch_setup)}
    models_dict[model_hp['feat']].load_state_dict(
        torch.load(args.snapshot,
                   map_location=lambda storage, loc: storage)['state_dict']
    )
    models_dict[model_hp['feat']].eval()

    logging.info('Creating database alas indexing corpus')
    engine = corpus.__dict__[args.engine](dataset, models_dict)
    engine.indexing()

    logging.info('Launch evaluation...')
    # log-scale up to the end of the database
    if len(args.topk) == 1 and args.topk[0] == 0:
        exp = int(np.floor(np.log10(engine.num_moments)))
        args.topk = [10**i for i in range(0, exp + 1)]
        args.topk.append(engine.num_moments)
    judge = CorpusVideoMomentRetrievalEval(topk=args.topk)
    for query_metadata in tqdm(dataset.metadata, disable=args.disable_tqdm):
        vid_indices, segments = engine.query(query_metadata['language_input'])
        judge.add_single_predicted_moment_info(
            query_metadata, vid_indices, segments)

    logging.info('Summarizing results')
    logging.info(f'Number of queries: {len(judge.map_query)}')
    logging.info(f'Number of proposals: {engine.num_moments}')
    result = judge.evaluate()
    _ = [logging.info(f'{k}: {v}') for k, v in result.items()]
    if args.dump:
        filename = args.logfile.with_suffix('.json')
        logging.info(f'Dumping results into: {filename}')
        with open(filename, 'x') as fid:
            for key, value in result.items():
                result[key] = float(value)
            result['date'] = datetime.now().isoformat()
            json.dump(result, fid)
