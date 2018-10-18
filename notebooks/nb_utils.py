import sys

import h5py
import numpy as np
import pandas as pd

# TODO (release) remove next line when we package the code
sys.path.append('..')
from np_segments_ops import iou as segment_iou

FPS = 5
IOU_THRESHOLDS = [0.5, 0.7]


def make_annotations_df(instances, file_h5):
    "Create data-frames to ease indexing"
    instances_df = pd.DataFrame([{**i, **{'t_start': i['times'][0][0],
                                          't_end': i['times'][0][1]}}
                                 for i in instances])
    videos_set = {i for i in instances_df['video'].unique()}
    instances_gbv = instances_df.groupby('video')
    with h5py.File(file_h5, 'r') as f:
        videos_info = []
        for video_id, dataset in f.items():
            if video_id not in videos_set:
                continue
            videos_info.append(
                {'video': video_id,
                 'num_frames': dataset.shape[0],
                 'num_instances': instances_gbv.get_group(
                     video_id).shape[0],
                }
            )
    videos_df = pd.DataFrame(videos_info)
    return videos_df, instances_df


# TODO: deprecate the next two functions in favor of proposals.py module
def generate_windows(length, scale, linear=True):
    "create multi-scale (duration) right aligned windows"
    if not linear:
        raise NotImplementedError('WIP')
    windows = np.zeros((scale, 2))
    windows[:, 1] += np.arange(1, scale + 1) * length
    return windows


def sliding_window(length, scale, stride, t_end,
                   t_start=0, linear=True):
    "Sliding windows for a given time interval"
    list_of_np_windows = []
    canonical_windows = generate_windows(length, scale, linear)
    for t in np.arange(t_start, t_end, stride):
        window_t = canonical_windows * 1
        # shift windows
        window_t += t
        list_of_np_windows.append(window_t)
    windows = np.vstack(list_of_np_windows)
    # only keep valid windows inside video
    # this way is clean but change numbers
    # windows = windows[windows[:, 1] <= t_end, :]

    # hacky := we end up with windows length != alpha * length
    # with alpha in Z+
    windows[windows[:, 1] > t_end, 1] = t_end
    return windows


def recall_bound_and_search_space(videos_df, instances_df,
                                  stride, length, scale, linear=True,
                                  slidding_window_fn=sliding_window,
                                  iou_thresholds=IOU_THRESHOLDS, fps=FPS):
    """Compute recall and search-space for a given stride

    Args:
        videos_df : DataFrame with video info, required `num_frames`.
        instances_df : DataFrame with instance info, required `t_start`
                       and `t_end'.
    """
    num_videos = len(videos_df)
    videos_gbv = videos_df.groupby('video')
    instances_gbv = instances_df.groupby('video')

    matched_gt_per_iou = [[] for i in range(len(iou_thresholds))]
    search_space_card_per_video = np.empty(num_videos)
    for i, (video_id, gt_instances) in enumerate(instances_gbv):
        # Get ground-truth segments
        instances_start = gt_instances.loc[:, 't_start'].values[:, None]
        instances_end = gt_instances.loc[:, 't_end'].values[:, None]
        gt_segments = np.hstack([instances_start, instances_end])

        # Estimate video duration
        num_frames = videos_gbv.get_group(
            video_id)['num_frames'].values[0]
        t_end = num_frames / FPS

        # sanitize
        # i) clamp moments inside video
        gt_segments[gt_segments[:, 0] > t_end, 0] = t_end
        gt_segments[gt_segments[:, 1] > t_end, 1] = t_end
        # ii) remove moments with duration <= 0
        duration = gt_segments[:, 1] - gt_segments[:, 0]
        ind = duration > 0
        if ind.sum() == 0:
            continue
        gt_segments = gt_segments[ind, :]

        # Generate search space
        windows = slidding_window_fn(length, scale, stride, t_end,
                                     linear=linear)
        search_space_card_per_video[i] = len(windows)

        # IOU between windows and gt_segments
        iou_pred_vs_gt = segment_iou(windows, gt_segments)
        # Computing upper-bound of recall, given that we don't have
        # more info to do assignments
        iou_per_gt_i = iou_pred_vs_gt.max(axis=0)

        # Compute matched_gt_per_iou for over multiple thresholds
        for j, iou_thr in enumerate(iou_thresholds):
            matched_gt_per_iou[j].append(iou_per_gt_i >= iou_thr)

    recall_ious = np.empty(len(iou_thresholds))
    for i, list_of_arrays in enumerate(matched_gt_per_iou):
        matched_gt_i = np.concatenate(list_of_arrays)
        recall_ious[i] = matched_gt_i.sum() / len(matched_gt_i)
    search_space_median_std = np.array(
        [np.median(search_space_card_per_video),
         np.std(search_space_card_per_video)]
    )

    return recall_ious, search_space_median_std