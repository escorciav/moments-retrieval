import itertools
import json
import random
import re
from enum import IntEnum, unique

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from glove import RecurrentEmbedding
from utils import timeit

POSSIBLE_SEGMENTS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
for i in itertools.combinations(range(len(POSSIBLE_SEGMENTS)), 2):
    POSSIBLE_SEGMENTS.append(i)
POSSIBLE_SEGMENTS_SET = set(POSSIBLE_SEGMENTS)


def word_tokenize(s):
    sent = s.lower()
    sent = re.sub('[^A-Za-z0-9\s]+', ' ', sent)
    return sent.split()


def sentences_to_words(sentences):
    words = []
    for s in sentences:
        words.extend(word_tokenize(str(s.lower())))
    return words


class Didemo(Dataset):
    "Base DiDeMo Dataset"

    def __init__(self, json_file, cues=None):
        self._setup_list(json_file)
        self._load_features(cues)

    @property
    def segments(self):
        return POSSIBLE_SEGMENTS

    def _setup_list(self, filename):
        "Read JSON file with all the moments"
        self.json_file = filename
        with open(filename, 'r') as f:
            self.metadata = json.load(f)
        self._preprocess_descriptions()

    def _preprocess_descriptions(self):
        "Tokenize descriptions into words"
        for d in self.metadata:
            d['language_input'] = sentences_to_words([d['description']])

    def _load_features(self, cues):
        """Read all features (coarse chunks) in memory

        TODO:
            Edit to only load features of videos in metadata
        """
        self.cues = cues
        self.features = {}
        for key, params in cues.items():
            with h5py.File(params['file'], 'r') as f:
                self.features[key] = {i: v[:] for i, v in f.items()}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        raise NotImplementedError


class DidemoMCN(Didemo):
    """Data feeder for MCN
    TODO:
        testing data
        enable multiple visual cues
        negative mining and batch negative sampling
        make sorting in collate optional
    """

    def __init__(self, json_file, cues=None, loc=True, max_words=50,
                 test=False):
        super(DidemoMCN, self).__init__(json_file, cues)
        self.visual_interface = VisualRepresentationMCN()
        self.lang_interface = LanguageRepresentationMCN(max_words)
        self.tef_interface = None
        if loc:
            self.tef_interface = TemporalEndpointFeature()
        self.eval = False
        if test:
            self.eval = True

    def _compute_visual_feature(self, video_id, time=None):
        "Pool visual features and append TEF for a given segment"
        feature_collection_video_t = {}
        for key, feat_db in self.features.items():
            feature_video = feat_db[video_id]
            feature_video_t = self.visual_interface(*time, feature_video)
            if self.tef_interface:
                feature_video_t = np.concatenate(
                    [feature_video_t, self.tef_interface(time)])
            feature_collection_video_t[key] = feature_video_t.astype(
                np.float32)
        return feature_collection_video_t

    def _compute_visual_feature_eval(self, video_id):
        "Pool visual features and append TEF for all segments in video"
        all_t = [self._compute_visual_feature(video_id, t)
                 for t in self.segments]
        # List of dicts to dict of list
        all_t_dict = dict(zip(all_t[0],
                              zip(*[d.values() for d in all_t])))
        for k, v in all_t_dict.items():
            n = len(v)
            all_t_dict[k] = np.concatenate(v).reshape((n, -1))
        return all_t_dict

    def _negative_intra_sampling(self, idx, p_time):
        """Sample visual feature inside the video
        TODO:
            negative mining. Weak supervision?
        """
        video_id = self.metadata[idx]['video']
        if not isinstance(p_time, tuple):
            p_time = tuple(p_time)
        possible_n = list(POSSIBLE_SEGMENTS_SET - {p_time})
        random.shuffle(possible_n)
        n_time = possible_n[0]
        return self._compute_visual_feature(video_id, n_time)

    def _negative_inter_sampling(self, idx, p_time):
        """Sample visual feature outside the video
        TODO:
            test other time intervals
        """
        video_id = self.metadata[idx]['video']
        other_video = video_id
        while other_video == video_id:
            idx = int(random.random()*len(self.metadata))
            other_video = self.metadata[idx]['video']
        return self._compute_visual_feature(other_video, p_time)

    def __getitem__(self, idx):
        moment_i = self.metadata[idx]
        video_id = moment_i['video']
        num_annotators = len(moment_i['times'])
        annot_i = random.randint(0, num_annotators - 1)
        time = moment_i['times'][annot_i]
        query = moment_i['language_input']

        # TODO: pack next two vars into a dict
        sentence_feature = self.lang_interface(query)
        len_query = len(query)
        if self.eval:
            pos_visual_feature = self._compute_visual_feature_eval(video_id)
            n_segments = len(self.segments)
            len_query = [len_query] * n_segments
            sentence_feature = np.tile(sentence_feature, (n_segments, 1, 1))
            neg_intra_visual_feature = None
            neg_inter_visual_feature = None
        else:
            pos_visual_feature = self._compute_visual_feature(video_id, time)
            # Sample negatives
            neg_intra_visual_feature = self._negative_intra_sampling(
                idx, time)
            neg_inter_visual_feature = self._negative_inter_sampling(
                idx, time)

        return (sentence_feature, len_query, pos_visual_feature,
                neg_intra_visual_feature, neg_inter_visual_feature)

    def collate_data(self, batch):
        all_tensors = default_collate(batch)
        # Sort due to LSTM dealing with variable length
        al_s, idx = all_tensors[1].sort(descending=True)
        a_s = all_tensors[0][idx, ...]
        a_s.requires_grad_()
        dicts_of_tensors = (
            {k: v[idx, ...].requires_grad_() for k, v in i.items()}
            for i in all_tensors[2:])
        return (a_s, al_s) + tuple(dicts_of_tensors)

    def collate_test_data(self, batch):
        # Note: we could do batching but taking care of the length of the
        # sentence was a mess.
        assert len(batch) == 1
        tensors = []
        for i, v in enumerate(batch[0]):
            if isinstance(v, np.ndarray):
                tensors.append(torch.from_numpy(v))
            elif i == 1:
                tensors.append(torch.tensor(v))
            elif i == 2:
                assert isinstance(v, dict)
                tensors.append({k: torch.from_numpy(t_np)
                                for k, t_np in v.items()})
            else:
                tensors.append(None)
        return tensors


class DidemoSMCN(DidemoMCN):
    "Data feeder for SMCM"

    def __init__(self, json_file, cues=None, loc=True, max_words=50,
                 test=False, context=True):
        super(DidemoSMCN, self).__init__(json_file, cues)
        self.visual_interface = VisualRepresentationSMCN(context=context)
        self.lang_interface = LanguageRepresentationMCN(max_words)
        self.tef_interface = None
        if loc:
            self.tef_interface = TemporalEndpointFeature()
        self.eval = False
        if test:
            self.eval = True

    def _compute_visual_feature(self, video_id, time):
        "Pool visual features and append TEF for a given segment"
        feature_collection_video_t = {}
        for key, feat_db in self.features.items():
            feature_video = feat_db[video_id]
            feature_video_t, mask = self.visual_interface(
                *time, feature_video)
            T = mask.sum()
            if self.tef_interface:
                N = len(feature_video_t)
                tef_feature = np.zeros((N, 2))
                tef_feature[:T, :] = self.tef_interface(time)
                feature_video_t = np.concatenate(
                    [feature_video_t, tef_feature], axis=1)
            feature_collection_video_t[key] = feature_video_t.astype(
                np.float32)
        # whatever masks is fine given that we don't consider time responsive
        # features yet?
        # TODO: check if we don't need to cast
        feature_collection_video_t['mask'] = mask.astype(np.float32)
        return feature_collection_video_t

    def _compute_visual_feature_eval(self, video_id):
        "Pool visual features and append TEF for all segments in video"
        all_t = [self._compute_visual_feature(video_id, t)
                 for t in self.segments]
        # List of dicts to dict of list
        all_t_dict = dict(zip(all_t[0],
                              zip(*[d.values() for d in all_t])))
        for k, v in all_t_dict.items():
            all_t_dict[k] = np.stack(v)
        return all_t_dict

@unique
class SourceID(IntEnum):
    VIDEO = 0
    IMAGE = 1


class DidemoSMCNHeterogeneous(DidemoSMCN):
    "Data feeder for SMCM with Heterogenous data"

    def __init__(self, json_file, cues=None, loc=False, max_words=50,
                 test=False, context=False, DEBUG=False):
        self.DEBUG = DEBUG
        super(DidemoSMCNHeterogeneous, self).__init__(json_file, cues)
        self.visual_interface = VisualRepresentationSMCN(context)
        self.lang_interface = LanguageRepresentationMCN(max_words)
        self.tef_interface = None
        if loc:
            self.tef_interface = TemporalEndpointFeature()
        self.eval = False
        if test:
            self.eval = True
        # TODO: decorator preserving parent method
        self._set_source_to_idxs()

    def _load_features(self, cues):
        "Read all features (coarse chunks) in memory"
        assert all([isinstance(v['file'], list) for _, v in cues.items()])
        self.cues = cues
        self.features = dict.fromkeys(cues.keys())

        keep = []
        for key, params in cues.items():
            self.features[key] = {}
            repo = [h5py.File(i, 'r') for i in params['file']]
            for idx, instance in enumerate(self.metadata):
                h5_id = instance['video']
                source_id = instance['source']

                if self.DEBUG:
                    if source_id == SourceID.VIDEO:
                        if random.random() > 0.15:
                            continue
                    else:
                        if random.random() > 0.01:
                            continue
                    keep.append(idx)

                self.features[key].update({h5_id: repo[source_id][h5_id][:]})

        if self.DEBUG:
            keep = set(keep)
            self.metadata = [value for i, value in enumerate(self.metadata)
                             if i in keep]
            self._set_source_to_idxs()

    def _set_source_to_idxs(self):
        "Create a dict grouping all indices in metadata with same source-id"
        self.source = {i: [] for i in list(SourceID)}
        for ind, instance in enumerate(self.metadata):
            source_id_i = instance.get('source')
            if source_id_i is not None:
                self.source[source_id_i].append(ind)

    def _negative_intra_sampling(self, p_ind, p_time):
        """Sample visual feature inside the video

        WIP: joint training is disabled
            We need to skip this sampling for SourceID.IMAGE instances
            [ ] define what skip means. zero-vector?
            [ ] zero-vector implies masking instances in loss.
            [ ] zero-vector also demands to know feat_dim.

        """
        return None

    def _negative_inter_sampling(self, p_ind, p_time):
        "Sample visual feature outside the video"
        source_id = self.metadata[p_ind]['video']
        source_idx = self.metadata[p_ind]['source']
        p_label = self.metadata[p_ind]['language_input']

        other_source_id = source_id
        while other_source_id == source_id:
            idx_ = int(random.random() * len(self.source[source_idx]))
            idx = self.source[source_idx][idx_]
            other_source_id = self.metadata[idx]['video']

            if (source_idx == SourceID.IMAGE and
                self.metadata[idx]['language_input'][0] == p_label):
                other_source_id = source_id
        return self._compute_visual_feature(other_source_id, p_time)


class LanguageRepresentationMCN(object):
    "Get representation of sentence"

    def __init__(self, max_words):
        self.max_words = max_words
        self.rec_embedding = RecurrentEmbedding()
        self.dim = self.rec_embedding.embedding.glove_dim

    def __call__(self, query):
        "Return padded sentence feature"
        feature = np.zeros((self.max_words, self.dim), dtype=np.float32)
        len_query = min(len(query), self.max_words)
        for i, word in enumerate(query[:len_query]):
            if word in self.rec_embedding.vocab_dict:
                feature[i, :] = self.rec_embedding.vocab_dict[word]
        return feature


class VisualRepresentationMCN(object):
    "Process visual features"

    def normalization(self, start, end, features):
        "Visual feature nomarlization"
        base_feature = np.mean(features[start:end+1, :], axis = 0)
        return base_feature / (np.linalg.norm(base_feature) + 0.00001)

    def __call__(self, start, end, features):
        "Compute visual representation of the clip (global | local)"
        assert features.shape[0] == 6
        feature_dim = features.shape[1]
        full_feature = np.zeros((feature_dim * 2,))
        if np.sum(features[5,:]) > 0:
            full_feature[:feature_dim] = self.normalization(0, 6, features)
        else:
            full_feature[:feature_dim] = self.normalization(0, 5, features)
        full_feature[feature_dim:feature_dim * 2] = self.normalization(
            start, end, features)
        return full_feature


class VisualRepresentationSMCN(object):
    "Process visual features"
    # Maximum temporal support, set based on DiDeMo
    N = 6
    EPS = 0.00001

    def __init__(self, context=True):
        self.context = context

    def __call__(self, start, end, features):
        "Compute visual representation of the clip (global | S-features)"
        # assert features.shape[0] == self.N
        assert end >= start
        feat_dim, feat_dim_mulptiplier = features.shape[1], 1
        loc_index_start, loc_index_end = 0, feat_dim
        if self.context:
            feat_dim_mulptiplier = 2
            loc_index_start, loc_index_end = feat_dim, feat_dim * 2
        full_feat_dim = feat_dim * feat_dim_mulptiplier
        # T := \mathcal{T} but in this case is the cardinality of the set
        T = end - start + 1

        padded_feature = np.zeros((self.N, full_feat_dim))
        mask = np.zeros(self.N, dtype=np.int64)

        if self.context:
            padded_feature[:T, :feat_dim] = self._global_feature(features)
        padded_feature[:T, loc_index_start:loc_index_end] = (
            self._local_feature(start, end, features))
        mask[:T] = 1
        return padded_feature, mask

    def _global_feature(self, x):
        "Compute global representation"
        if np.sum(x[5,:]) > 0:
            return normalization1d(0, 6, x)
        else:
            return normalization1d(0, 5, x)

    def _local_feature(self, start, end, x):
        "Compute local representation"
        x_ = x[start:end + 1, :]
        scaling_factor = np.linalg.norm(x_, axis=1, keepdims=True) + self.EPS
        return x_ / scaling_factor

class TemporalEndpointFeature(object):
    "Relative position in the video"

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, start_end):
        return np.array(start_end, dtype=self.dtype) / 6


def normalization1d(start, end, features):
    "1D mean-pooling + normalization for visual features"
    base_feature = np.mean(features[start:end + 1, :], axis=0)
    scaling_factor = np.linalg.norm(base_feature) + 0.00001
    return base_feature / scaling_factor


if __name__ == '__main__':
    import time
    # Unit-test DidemoMCN
    data = 'data/raw/train_data.json'
    rgb = 'data/raw/average_fc7.h5'
    flow = 'data/raw/average_global_flow.h5'
    cues = {'rgb': {'file': rgb}, 'flow': {'file': flow}}
    t_start = time.time()
    dataset = DidemoMCN(data, cues)
    print(f'Time loading {data}: ', time.time() - t_start)
    print(len(dataset))
    print(dataset.metadata[0])
    for i, v in enumerate(dataset[0]):
        if i== 0:
            assert isinstance(v, np.ndarray)
            print(i, v.shape)
        elif i == 1:
            print(i, v)
        elif i > 1:
            assert isinstance(v, dict)
            for k, v in v.items():
                print(i, k, v.shape)

    dataset.eval = True
    values = dataset[0]
    for i, v in enumerate(values):
        if i== 0:
            assert isinstance(v, np.ndarray)
            print(i, v.shape)
        elif i == 1:
            assert isinstance(v, list)
            print(i, len(v))
        elif i == 2:
            assert isinstance(v, dict)
            for k, v in v.items():
                print(i, k, v.shape)
        elif i > 2:
            assert v is None
        else:
            raise ValueError

    # Unit-test DidemoSMCN
    t_start = time.time()
    dataset = DidemoSMCN(data, cues)
    print(f'Time loading {data}: ', time.time() - t_start)
    print(len(dataset))
    print(dataset.metadata[0])
    for i, v in enumerate(dataset[0]):
        if i== 0:
            assert isinstance(v, np.ndarray)
            print(i, v.shape)
        elif i == 1:
            print(i, v)
        elif i > 1:
            assert isinstance(v, dict)
            for k, v in v.items():
                if isinstance(v, np.ndarray):
                    print(i, k, v.shape)
                else:
                    print(i, k, v)

    dataset.eval = True
    values = dataset[0]
    for i, v in enumerate(values):
        if i== 0:
            assert isinstance(v, np.ndarray)
            print(i, v.shape)
        elif i == 1:
            assert isinstance(v, list)
            print(i, len(v))
        elif i == 2:
            assert isinstance(v, dict)
            for k, v in v.items():
                print(i, k, v.shape)
        elif i > 2:
            assert v is None
        else:
            raise ValueError

    # Unit-test DidemoSMCNHeterogeneous
    filename = 'data/interim/didemo_yfcc100m/train_data.json'
    cues = {'rgb': {'file': ['data/interim/didemo/resnet152/320x240_max.h5',
                             'data/interim/yfcc100m/resnet152/320x240_001.h5']
                    }
            }
    t_start = time.time()
    dataset = DidemoSMCNHeterogeneous(json_file=filename, cues=cues)
    print(f'Time loading {filename}: ', time.time() - t_start)
    idxs = random.choices(dataset.source[SourceID.IMAGE], k=5)
    for i in idxs:
        data = dataset[i]
        if dataset.metadata[i]['source'] == SourceID.IMAGE:
            assert data[1] == 1
            np.testing.assert_array_almost_equal(data[2]['mask'],
                                                 [1] + [0] * 5)
            assert data[3] is None