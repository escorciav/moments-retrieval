"DiDeMo evaluation"
import json

import numpy as np

from corpus import Corpus, CorpusAsDistanceMatrix
from dataset import Queries
from np_segments_ops import iou as segments_iou


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


class RetrievalEvaluation():
    "TODO: count search for a given query_id"
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
        self.iou_matrix = segments_iou(segments, segments)


class CorpusVideoMomentRetrievalEvalFromMatrix():
    """Helper for Corpus video moment retrieval from distance matrix

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
        self.iou_matrix = segments_iou(segments, segments)


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    import h5py
    parser = argparse.ArgumentParser(description='pseudo unit-test')
    parser.add_argument('--corpus-h5',
                        default='data/interim/mcn/corpus_val_rgb.hdf5')
    parser.add_argument('--queries-h5',
                        default='data/interim/mcn/queries_val_rgb.hdf5')
    parser.add_argument('--json-file', default='data/raw/val_data_wwa.json')
    parser.add_argument('--distance-h5', type=Path, default='non-existent')
    parser.add_argument('--full', action='store_true', help='more metrics')
    args = parser.parse_args()
    if args.distance_h5.exists():
        _judge = CorpusVideoMomentRetrievalEvalFromMatrix(
            args.json_file, args.distance_h5, (1, 5, 10), 0.1)
        _performance = _judge.eval()
        print('R@{0:}={1:};\nmRank={2:.2f};'
              .format(_judge.k, *_performance))
        with open('test_output/corpus_moment_retrieval_from_matrix.results',
                  'x') as f:
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
        with open('test_output/corpus_moment_retrieval.results', 'x') as f:
            json.dump({'k': _judge.k, 'performance': _performace}, f)