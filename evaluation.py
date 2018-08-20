"DiDeMo evaluation"
import numpy as np

from corpus import Corpus
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


if __name__ == '__main__':
    import h5py
    _filename1 = 'data/interim/mcn/corpus_val_rgb.hdf5'
    _filename2 = 'data/raw/val_data.json'
    _judge = RetrievalEvaluation(_filename1, _filename2, (1, 5, 10), 0.1)
    _filename = 'data/interim/mcn/queries_val_rgb.hdf5'

    with h5py.File(_filename, 'r') as fid:
        # sample_key = list(fid.keys())[0]
        # sample_value = fid[sample_key][:]
        # query_id = int(sample_key)
        # val_judge.eval_single_vector(sample_value, query_id)
        import time
        start = time.time()
        for _sample_key, h5ds in fid.items():
            _query_id = int(_sample_key)
            _query_vector = h5ds[:]
            _judge.eval_single_vector(_query_vector, _query_id)
        _performace = _judge.eval(full=True)
        print('R@{0:}={2:};\nR@{0:},{1:}={3:};\nR@{0:},didemo={4:};\n'
              'mIOU={5:.4f};\nmRank={6:.2f};'
              .format(_judge.k, _judge.iou_threshold,
                      *_performace))
        print('Elapsed time:', time.time() - start)