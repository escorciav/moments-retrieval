image_csv = '../data/interim/yfcc100m/001.csv'
didemo_jsons = ['../data/raw/train_data.json',
                '../data/raw/val_data.json']
nouns2video_json = '../data/interim/didemo/nouns_to_video.json'
image_h5 = '../data/interim/yfcc100m/resnet152/320x240_001.h5'
video_h5 = '../data/interim/didemo/resnet152/320x240_max.h5'
IMAGES_PER_TAG = 100
RELAX_FACTOR = 100      # increase -> more tags
MINIMORUM = 75          # decrease -> less tags
MODE = 0                # increase -> more tags
TOPK = 15
OUTPUT_FILE = f'../data/interim/yfcc100m/003-{RELAX_FACTOR}-{MODE}.csv'

import json
import random
import time
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


# TODO. unit-norm features
class MomentDescriptor():
    def __init__(self, filename):
        self.file = filename

    def __call__(self, video, time):
        start, end = time
        end += 1
        with h5py.File(self.file, 'r') as fid:
            feature = fid[video][:]
            # TODO: try max?
            descriptor = feature[start:end, :].mean(axis=0)
        return descriptor

    def get_features(self, videos, times):
        assert len(videos) == len(times)
        descriptors = []
        with h5py.File(self.file, 'r') as fid:
            for i, video_i in enumerate(videos):
                feature_i = fid[video_i][:]
                start, end = times[i]
                end += 1
                descriptors.append(feature_i[start:end, :].mean(axis=0, keepdims=True))
        descriptors = np.concatenate(descriptors, axis=0)
        return descriptors


def load_image_features(filename, id_list):
    feat_db_list = []
    end = time.time()
    with h5py.File(filename, 'r') as fid:
        for v in id_list:
            feat_db_list.append(fid[v][:])
    print(f'Loaded image features: {time.time() - end}')
    end = time.time()
    feat_db = np.stack(feat_db_list).squeeze()
    print(f'Stacking features: {time.time() - end}')
    return feat_db

# get videos in train-val
didemo_videos = set()
didemo_moments = {}
for filename in didemo_jsons:
    with open(filename, 'r') as fid:
        for moment in json.load(fid):
            didemo_videos.add(moment['video'])
            moment_id = moment['annotation_id']
            didemo_moments[moment_id] = moment

# mapping of NOUNs to didemo videos
with open(nouns2video_json, 'r') as fid:
    didemo_nouns2video = json.load(fid)

get_descriptor = MomentDescriptor(video_h5)

df_yfcc100m = pd.read_csv(image_csv)
# Only consider TOPK tags of an image
df_yfcc100m.loc[:, 'tags'] = df_yfcc100m.loc[:, 'tags'].apply(lambda x: ';'.join(x.split(';')[:TOPK]) + ';')
image_descriptors = load_image_features(image_h5, df_yfcc100m['h5_id'].tolist())

end = time.time()
image_tree = cKDTree(image_descriptors)
print(f'Building tree: {time.time() - end}')
end = time.time()

clean_idxs = set()
debug, checalebn = [], []
for tag, _ in tqdm(df_yfcc100m.groupby('topk_tags')):
    assert tag in didemo_nouns2video['nouns']
    moments_videos = didemo_nouns2video['videos'][tag]
    moments_time = didemo_nouns2video['time'][tag]
    # DEBUG: get description
    assert len(moments_videos) == len(moments_time)

    moment_idxs = [j for j, video_j in enumerate(moments_videos)
                   if video_j in didemo_videos]
    n_per_j = (IMAGES_PER_TAG * RELAX_FACTOR) // len(moment_idxs)

    clean_idxs_i = set()
    for j in moment_idxs:
        moment_j = get_descriptor(moments_videos[j], moments_time[j])
        distance_j, ind_j = image_tree.query(moment_j, k=n_per_j, n_jobs=-1)
        # filter by tag
        if MODE == 0:
            pick_j = df_yfcc100m.loc[ind_j, 'topk_tags'] == tag
        elif MODE == 1:
            pick_j = df_yfcc100m.loc[ind_j, 'tags'].apply(lambda x: tag in x)
        else:
            raise
        clean_idxs_i.update(ind_j[pick_j].tolist())

        if random.random() < 0.01 and pick_j.sum() > 0:
            debug.append((moments_videos[j],
                          moments_time[j],
                          tag,
                          df_yfcc100m.loc[ind_j[pick_j], 'url'].iloc[:min(5, pick_j.sum())].tolist(),
                          )
                        )
    if len(clean_idxs_i) >= MINIMORUM:
        clean_idxs_i = list(clean_idxs_i)
        clean_idxs.update(clean_idxs_i[:min(IMAGES_PER_TAG, len(clean_idxs_i))])
    checalebn.append(len(clean_idxs_i))

clean_df = df_yfcc100m.loc[clean_idxs, :]
with open(OUTPUT_FILE, 'x') as fid:
    clean_df.to_csv(fid, index=None)
with open(OUTPUT_FILE.replace('.csv', '.json'), 'x') as fid:
    json.dump({'len_per_tag': checalebn,
               'dataset_size': len(clean_idxs),
               'debug_instances': debug,
              },
              fid)
# damm there are so many degrees of freedom, definetily I can't reject the hypothesis
# only conclude that I'm unlucky and not smart