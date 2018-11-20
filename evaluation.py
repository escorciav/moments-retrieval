"Moment retrieval evaluation in a single & corpus video(s)"
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
    assert topk.dtype == torch.float
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

    def add_single_predicted_moment_info(
            self, query_info, video_indices, pred_segments, max_rank=None):
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
        if max_rank is None:
            max_rank = len(pred_segments)
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
                    [max_rank], device=video_indices.device)
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
