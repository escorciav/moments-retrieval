import json
import random
import sys
from copy import deepcopy

import h5py
import numpy as np
import pandas as pd

# TODO (release) remove next line when we package the code
sys.path.append('..')
from np_segments_ops import iou as segment_iou

FPS = 5
IOU_THRESHOLDS = [0.5, 0.7]


def extend_metadata(list_of_moments, videos_gbv, filename, offset=0, fps=FPS):
    """Augment list of moments' (in-place) metadata and create video metadata

    The augmentation add `annotation_id` and `time` to each moment in
    list_of_moments.

    Args:
        list_of_moments (list of dicts) : the output of any parsing function
            such as `function::parse_charades_sta` or
            `function::parse_activitynet_captions`.
        videos_gbv (DataFrame groupedby) : pandas DataFrame grouped by
            `video_id` field. It is mandatory that the DataFrame has a column
            `num_frames` such that we can estimate the duration of the video.
        filename (str) : path to HDF5 with all features of the dataset.
        offset (int, optional) : ensure annotation-id accross moments is
            unique accross the entire dataset.
    Returns:
        videos (dict) : key-value pairs formed by video-id and its
            corresponding metadata.
        clean_list_of_moments (list) : the same contern of list_of_moments
            limited to the moments with duration >= 0.

    Note:
        We use the row index of the original CSV as unique identifier for the
        moment. For JSON files, the behavior it's trickier but we double check
        it. This worked for the json package of python (==3.6 from anaconda)
        standard library and the `function::parse_activitynet_captions`.
        BTW, the annotation-id is 0-indexed.
    """
    with h5py.File(filename, 'r') as fid:
        videos = {}
        keep = []
        for i, moment in enumerate(list_of_moments):
            assert len(moment['times']) == 1
            video_id = moment['video']
            # Get estimated video duration
            num_frames = videos_gbv.get_group(
                video_id)['num_frames'].values[0]
            video_duration = num_frames / fps

            # TODO: sanitize by trimming moments up to video duration <= 0
            # Sanitize
            # 1) clamp moments inside video
            moment['times'][0][0] = min(moment['times'][0][0], video_duration)
            moment['times'][0][1] = min(moment['times'][0][1], video_duration)
            # 2) remove moments with duration <= 0
            if moment['times'][0][1] <= moment['times'][0][0]:
                continue

            keep.append(i)
            moment['time'] = moment['times'][0]
            moment['annotation_id'] = i + offset

            # Update dict with video info
            if video_id not in videos:
                num_clips = fid[video_id].shape[0]
                videos[video_id] = {'duration': video_duration,
                                    'num_clips': num_clips,
                                    'num_moments': 0}
            videos[video_id]['num_moments'] += 1

    clean_list_of_moments = []
    for i in keep:
        clean_list_of_moments.append(list_of_moments[i])
    return videos, clean_list_of_moments


def make_annotations_df(instances, file_h5):
    "Create data-frames to ease indexing"
    instances_df = pd.DataFrame([{**i, **{'t_start': i['times'][0][0],
                                          't_end': i['times'][0][1]}}
                                 for i in instances])
    videos_set = {i for i in instances_df['video'].unique()}
    instances_gbv = instances_df.groupby('video')
    videos_info = []
    with h5py.File(file_h5, 'r') as f:
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


def split_moments_dataset(filename, ratio=0.75):
    """Create two disjoint partitions of a Dataset of moments

    Args:
        filename : path to JSON-file with the moments dataset
        ratio : float indicating the partition ratio in terms of
            number of unique videos.

    Returns:
        split1 : dict of length ratio * len(videos)
        split2 : dict of length (1 - ratio) * len(videos)
    """
    with open(filename, 'r') as fid:
        data = json.load(fid)
        video2moment_ind = {}
        for i, moment in enumerate(data['moments']):
            video_id = moment['video']
            if video_id not in video2moment_ind:
                video2moment_ind[video_id] = []
            video2moment_ind[video_id].append(i)

    splits = []
    for i in range(2):
        splits.append(deepcopy(data))
        splits[-1]['videos'] = {}
        splits[-1]['moments'] = []

    videos = list(data['videos'].keys())
    cut = int(len(videos) * ratio)
    random.shuffle(videos)

    repo = splits[0]
    for i, video_id in enumerate(videos):
        if i > cut:
            repo = splits[1]
        repo['videos'][video_id] = data['videos'][video_id]
        for j in video2moment_ind[video_id]:
            repo['moments'].append(data['moments'][j])
    return (*splits,)