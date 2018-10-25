"DiDeMo evaluation"
import json
from itertools import product

import numpy as np
import torch

from corpus import Corpus, CorpusAsDistanceMatrix
from dataset import Queries
from np_segments_ops import iou as numpy_iou
from np_segments_ops import torch_iou

DEFAULT_TOPK_AND_IOUTHRESHOLDS = tuple(product((1, 5), (0.5, 0.7)))


def iou(gt, pred):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return intersection / union


def rank(gt, pred):
    return pred.index(tuple(gt)) + 1


def video_evaluation(gt, predictions, k=(1, 5)):
    top_segment = predictions[0]
    ious = [iou(top_segment, s) for s in gt]
    average_iou = np.mean(np.sort(ious)[-3:])
    ranks = [rank(s, predictions) for s in gt]
    average_ranks = np.mean(np.sort(ranks)[:3])
    r_at_k = [average_ranks <= i for i in k]
    return [average_iou] + r_at_k


def single_moment_retrieval(true_segments, pred_segments,
                            k_iou=DEFAULT_TOPK_AND_IOUTHRESHOLDS):
    """Compute if a given segment is retrieved in top-k for a given IOU

    Args:
        true_segments (torch tensor) : shape [1, 2] holding 1 segments.
        pred_segments (torch tensor) : shape [M, 2] holding M segments sorted
            by their scores. pred_segment[0, :] is the most confident segment
            retrieved.
        k_iou (sequence of pairs) : a pair represents the top-k and iou
            threshold used for evaluation. It must be sorted in ascending
            order based on top-k i.e. k_iou[-1][0] is the largest k.
    Returns:
        list of torch tensors indicating if the true segment was found for a
        given iou_threshold
    """
    iou_matrix = torch_iou(pred_segments, true_segments)
    max_k = k_iou[-1][0]
    if iou_matrix.shape[0] < max_k:
        n_times = round(max_k / iou_matrix.shape[0])
        iou_matrix = iou_matrix.repeat(n_times, 1)
    hit_topk_iou = []
    for top_k, iou_threshold in k_iou:
        best_iou_topk, _ = iou_matrix[:top_k, :].max(dim=0)
        hit_topk_iou.append(best_iou_topk >= iou_threshold)
    return hit_topk_iou


class CorpusVideoMomentRetrievalEval():
    """

    Note:
        - method:evaluate change the type of `_rank_iou` and `_hit_iou_k` to
          block the addition of new instances. This also ease retrieval of per
          query metrics.
    """

    def __init__(self, topk=(1, 5, 10), iou_thresholds=[0.5, 0.7]):
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
        # TODO: define attributes

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
        true_segments = torch.from_numpy(true_segments)
        hit_video = video_indices == true_video
        iou_matrix = torch_iou(pred_segments, true_segments)
        for i, iou_threshold in enumerate(self.iou_thresholds):
            # TODO(tier-1): threshold over annotators. We need a fix >= for
            # DiDeMo
            hit_segment = (iou_matrix >= iou_threshold).sum(dim=1) >= 1
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
        if self.medrank_iou[0] is not None:
            return
        num_queries = len(self.map_query)
        for i, _ in enumerate(self.iou_thresholds):
            self._rank_iou[i] = torch.cat(self._rank_iou[i])
            # report results as 1-indexed ranks for humans
            ranks_i = self._rank_iou[i] + 1
            self.medrank_iou[i] = torch.median(ranks_i)
            self.stdrank_iou[i] = torch.std(ranks_i.float())
            self._hit_iou_k[i] = torch.stack(self._hit_iou_k[i])
            recall_i = self._hit_iou_k[i].sum(dim=0) / num_queries
            self.recall_iou_k[i] = recall_i
        self._rank_iou = torch.stack(self._rank_iou).transpose_(0, 1)
        self._hit_iou_k = torch.stack(self._hit_iou_k).transpose_(0, 1)


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
    args = None
    if args is None:
        return
    args.corpus_h5 = None  # path to HDF5 with feature matrix of corpus
    args.queries_h5 = None  # path HDF5 with feature matrix of queries
    args.distance_h5 = None  # path to HDF5-file with distance matrix
    args.json_file = None  # Path to JSON with DiDeMo format
    args.full = None # HDF5 with distance matrix
    if args.distance_h5.exists():
        _judge = CorpusVideoMomentRetrievalEvalFromMatrix(
            args.json_file, args.distance_h5, (1, 5, 10), 0.1)
        _performance = _judge.eval()
        print('R@{0:}={1:};\nmRank={2:.2f};'
              .format(_judge.k, *_performance))
        output_filename = 'corpus_moment_retrieval_from_matrix.results'
        with open(f'{args.dirname}/{output_filename}', 'x') as f:
            json.dump({'k': _judge.k, 'performance': _performance}, f)
        exit()

    _judge = RetrievalEvaluation(args.corpus_h5, args.json_file,
                                 (1, 5, 10), 0.1)
    with h5py.File(args.queries_h5, 'r') as fid:
        import time
        start = time.time()
        for _sample_key, h5ds in fid.items():
            _query_id = int(_sample_key)
            _query_vector = h5ds[:]
            _judge.eval_single_vector(_query_vector, _query_id)
        _performace = _judge.eval(full=args.full)
        print('Elapsed time:', time.time() - start)
        if args.full:
            print('R@{0:}={2:};\nR@{0:},{1:}={3:};\nR@{0:},didemo={4:};\n'
                  'mIOU={5:.4f};\nmRank={6:.2f};'
                  .format(_judge.k, _judge.iou_threshold,
                          *_performace))
        else:
            print('R@{0:}={1:};\nmRank={2:.2f};'
                  .format(_judge.k, *_performace))
        output_filename = 'corpus_moment_retrieval.results'
        with open(f'{args.dirname}/{output_filename}', 'x') as f:
            json.dump({'k': _judge.k, 'performance': _performace}, f)