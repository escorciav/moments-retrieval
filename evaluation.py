"Moment retrieval evaluation in a single & corpus video(s)"
from itertools import product

import numpy as np
import torch

from np_segments_ops import torch_iou
import sys
import math

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
        self.number_of_reranked_clips_per_query = []

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

    def oracle_concept_reranking(self, query_info, vid_indices, pred_segments, metadata, map):
        '''
        The function rerankes the moments such that we have at the top the moments 
        related to the concepts present in the query
        '''
        push_up_indices, other_indices = [], []
        query_tokens = query_info["concepts"]
        for i,v_id in enumerate(vid_indices):
            video_concepts = map[int(v_id)]
            if any(item in query_tokens for item in video_concepts):
                push_up_indices.append(i)
            else:
                other_indices.append(i)
        indices = push_up_indices + other_indices
        # Rerank
        new_vid_indices   = torch.stack([vid_indices[i] for i in indices])
        new_pred_segments = torch.stack([pred_segments[i] for i in indices])

        self.number_of_reranked_clips_per_query.append(len(other_indices))
        return new_vid_indices, new_pred_segments

    def oracle_object_reranking(self, query_info, vid_indices, pred_segments, scores,
                                    metadata, map_concepts_to_obj_class, clip_length,
                                    use_concepts_and_obj_predictions):
        '''
        The function rerankes the moments such that we have at the top the moments 
        related to the concepts present in the query
        '''
        reranking_indices = [[],[]]
        maps_keys = None
        maps_keys = list(map_concepts_to_obj_class.keys())
        
        list_of_video_id = list(metadata.keys())
        # get the coco id of the concepts that are in the sentence
        obj_class_of_query_concepts = [0]
        if use_concepts_and_obj_predictions:
            obj_class_of_query_concepts = [map_concepts_to_obj_class[t] for t in query_info['concepts'] if t in maps_keys]
            num_classes = len(list(set([map_concepts_to_obj_class[k] for k in maps_keys])))
            reranking_indices = [[] for elem in range(num_classes+1)]
        # If they have been detected we push the clip to the top
        for i,v_id in enumerate(vid_indices):
            #get predicted object for video
            predicted_obj_per_clip_in_video = metadata[list_of_video_id[int(v_id)]]['detected_objs_per_clip']
            segment = pred_segments[i].tolist()             # get predited segments
            start_idx = int(segment[0]/clip_length)         # get initial index of window
            end_idx = int(math.ceil(segment[1]/clip_length))# get final index of window
            last_window_idx = int(list(predicted_obj_per_clip_in_video.keys())[-1])
            end_idx = min(end_idx, last_window_idx)
            if start_idx == end_idx: 
                end_idx += 1
            detected_obj_class_in_moment = []                 # list of objects detected in moment
            for ii in range(start_idx,end_idx):
                detected_obj_class_in_moment.extend(predicted_obj_per_clip_in_video[str(ii)])
            # DETERMINE CONDITION TO USE:
            matching_objects = list(set([obj for obj in obj_class_of_query_concepts if obj in detected_obj_class_in_moment]))
            reranking_indices[len(matching_objects)].append(i)
        indices = []
        for index_list in reversed(reranking_indices):
            indices.extend(index_list)
            
        # Rerank
        new_vid_indices   = torch.stack([vid_indices[i] for i in indices])
        new_pred_segments = torch.stack([pred_segments[i] for i in indices])
        new_scores        = torch.stack([scores[i] for i in indices])
        
        #compute number of reranked moments
        self.number_of_reranked_clips_per_query.append(len(vid_indices)-len(reranking_indices[0]))
        return new_vid_indices, new_pred_segments, new_scores

    def merge_rankings(self, vid_indices, segments, scores,rerank_vid_indices, 
                    rerank_segments, rerank_scores, k1, k2, reordering):
        new_scores      = torch.cat((rerank_scores[:k2],scores[:k1]),0).numpy()
        new_vid_indices = torch.cat((rerank_vid_indices[:k2],vid_indices[:k1]),0).numpy()
        new_segments    = torch.cat((rerank_segments[:k2],segments[:k1]),0).numpy()

        if reordering:
            concat          = np.asarray(list(set(zip(new_scores,new_segments[:,0],new_segments[:,1],new_vid_indices))))
            # new_scores      = torch.tensor(concat[:,0])
            new_vid_indices = torch.tensor(concat[:,3],dtype=vid_indices.dtype)
            new_segments    = torch.tensor(concat[:,1:3],dtype=segments.dtype)
            # new_segments    = torch.cat([[concat[i,1],concat[i,2]] for i in range(concat.shape[0])])
            a = 0

        else:
            concat          = np.concatenate((np.expand_dims(new_scores,axis=1),
                                            np.expand_dims(new_segments[:,0],axis=1),
                                            np.expand_dims(new_segments[:,1],axis=1),
                                            np.expand_dims(new_vid_indices,axis=1)),axis=1)
            unique, ind     = np.unique(concat,return_index=True,axis=0)
            # new_scores      = torch.tensor([new_scores[i] for i in ind])
            new_vid_indices = torch.tensor([new_vid_indices[i] for i in ind],dtype=vid_indices.dtype)
            new_segments    = torch.tensor([new_segments[i] for i in ind],dtype=segments.dtype)
        return new_vid_indices, new_segments

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


class CorpusConceptVideoMomentRetrievalEval(CorpusVideoMomentRetrievalEval):
    
    def __init__(self, topk=(1, 5), iou_thresholds=IOU_THRESHOLDS):
        super(CorpusConceptVideoMomentRetrievalEval, self).__init__(
            topk=topk,iou_thresholds=iou_thresholds) 

    def add_single_predicted_moment_info(
            self, query_info, ordered_video_indices, pred_segments, max_rank=None):
        """Compute rank@IOU and hit@IOU,k per query
            TBD
        """
        if max_rank is None:
            max_rank = len(pred_segments)
        query_id = query_info.get('annotation_id')
        if query_id in self.map_query:
            raise ValueError('query was already added to the list')
        ind = len(self._rank_iou)
        true_videos_indices = query_info['video_index']
        true_segments = query_info['times']
        concensus_among_annotators = 1
        if len(true_segments[0]) > 1:
            concensus_among_annotators = 2
        _hit_iou_k = [[torch.zeros_like(self.topk_, dtype=torch.uint8)] for i in self.iou_thresholds]
        M = len(true_videos_indices)

        for rank, video_indice in enumerate(ordered_video_indices):
            if video_indice in true_videos_indices:
                idxs = self._check_duplicates(video_indice, true_videos_indices)
                hits = [self._determine_hit(true_segments, pred_segments[rank], idx) for idx in idxs]
                best = [sum(elem) for elem in hits]
                best_indice = best.index(max(best))
                boolean_hit = False
                
                for ii, _ in enumerate(self.iou_thresholds):
                    hit_segment = (hits[best_indice][ii] >= concensus_among_annotators)
                    if hit_segment:
                        self._rank_iou[ii].append(rank)
                        _hit_iou_k[ii].append(self.topk_ >= rank)
                        boolean_hit = True

                if boolean_hit:
                    del true_videos_indices[idxs[best_indice]]
                    del true_segments[idxs[best_indice]]

        for ii, _ in enumerate(self.iou_thresholds):
            temp = torch.stack(_hit_iou_k[ii]).type(torch.float).sum(dim=0)/M
            self._hit_iou_k[ii].append(temp)

        self.map_query[query_id] = ind


    def _determine_hit(self,true_segments, pred_segment, idx):
        hit_segment = [] 
        true_segments_ = torch.FloatTensor(true_segments[idx])
        pred_segment_ = pred_segment.unsqueeze(0)
        iou_vector = torch_iou(pred_segment_, true_segments_)
        for ii, iou_threshold in enumerate(self.iou_thresholds):
            hit_segment.append((iou_vector >= iou_threshold).sum(dim=1))
        return hit_segment


    def evaluate(self):
        "Compute MedRank@IOU and R@IOU,k accross the dataset"
        if self.performance is not None:
            return self.performance
        num_queries = len(self.map_query)
        for i, _ in enumerate(self.iou_thresholds):
            # report results as 1-indexed ranks for humans
            ranks_i = torch.FloatTensor(self._rank_iou[i])+1
            self.medrank_iou[i] = torch.median(ranks_i)
            self.stdrank_iou[i] = torch.std(ranks_i.float())
            self._hit_iou_k[i] = torch.stack(self._hit_iou_k[i])
            recall_i = self._hit_iou_k[i].sum(dim=0).float() / num_queries
            self.recall_iou_k[i] = recall_i
        # self._rank_iou = torch.stack(self._rank_iou).transpose_(0, 1)
        # self._hit_iou_k = torch.stack(self._hit_iou_k).transpose_(0, 1)
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


    def _check_duplicates(self, video_indice, true_videos_indices):
        cnt = true_videos_indices.count(video_indice)
        if cnt == 1:
            return [true_videos_indices.index(video_indice)]
        else:
            indices = [i for i,value in enumerate(true_videos_indices) 
                        if value==video_indice]
            return indices
