"""Operations for [N, 4] numpy arrays representing bounding boxes.
Example box operations that are supported:
  * Areas: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
"""
import numpy as np


def intersection(segments1, segments2):
    """Compute pairwise intersection length between segments.

    Args:
        segments1 (numpy array): shape [N, 2] holding N segments
        segments2 (numpy array): shape [M, 2] holding M segments
    Returns:
        a numpy array with shape [N, M] representing pairwise intersection length
    """
    [t_min1, t_max1] = np.split(segments1, 2, axis=1)
    [t_min2, t_max2] = np.split(segments2, 2, axis=1)

    all_pairs_min_tmax = np.minimum(t_max1, np.transpose(t_max2))
    all_pairs_max_tmin = np.maximum(t_min1, np.transpose(t_min2))
    intersect_length = np.maximum(
        np.zeros(all_pairs_max_tmin.shape),
        all_pairs_min_tmax - all_pairs_max_tmin)
    return intersect_length


def length(segments):
    """Computes length of segments.

    Args:
        segments (numpy array): shape [N, 2] holding N segments
    Returns:
        a numpy array with shape [N] representing segment length
    Note:
        it works with time, it would be off if using frames.
    """
    return segments[:, 1] - segments[:, 0]


def iou(segments1, segments2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        segments1 (numpy array): shape [N, 2] holding N segments
        segments2 (numpy array): shape [M, 4] holding N boxes.
    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(segments1, segments2)
    length1 = length(segments1)
    length2 = length(segments2)
    union = np.expand_dims(length1, axis=1) + np.expand_dims(
        length2, axis=0) - intersect
    return intersect / union


def non_maxima_suppresion(segments, scores, nms_threshold):
    """non-maxima suppresion over segments

    Args:
        segments (numpy array): shape [N, 2] holding N segments
        scores (numpy array): shape [N] holding score of each segment.
    Returns:
        a numpy array with shape [M] representing indexes to pick after nms.
    """
    t1, t2 = np.split(segments, 2, axis=1)
    area = t2 - t1
    idx = np.argsort(scores)
    ind_pick = []
    for i in range(len(idx)):
        if len(idx) == 0:
            break
        p = idx[len(idx) - 1]
        ind_pick.append(p)

        tt1 = np.maximum(t1[p], t1[idx])
        tt2 = np.minimum(t2[p], t2[idx])
        wh = np.maximum(0, tt2 - tt1)
        o = wh / (area[p] + area[idx] - wh)

        ind_rm_i = np.where(o >= nms_threshold)[0]
        idx = np.delete(idx, ind_rm_i)
    ind_pick = np.array(ind_pick)
    return ind_pick


if __name__ == '__main__':
    x = np.random.rand(4, 2)
    y = np.random.rand(3, 2)
    intersection(x, y)
    length(x)
    iou(x, y)
    
    scores = np.random.rand(4)
    nms_threshold = 0.75
    non_maxima_suppresion(x, scores, nms_threshold)