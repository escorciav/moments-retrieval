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
SAMPLING_SCHEMES = ['skip', 'finetune', 'joint']


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
        pass

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
        self.features = dict.fromkeys(cues.keys())
        for key, params in cues.items():
            with h5py.File(params['file'], 'r') as f:
                self.features[key] = {i['video']: f[i['video']][:]
                                      for i in self.metadata}

    def _set_metadata_per_video(self):
        _tmp = {}
        for moment in self.metadata:
            if moment['video'] not in _tmp:
                _tmp[moment['video']] = [moment]
                continue
            _tmp[moment['video']].append(moment)
        self.metada_per_video = [(k, v) for k, v in _tmp.items()]

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
        self._setup_list(json_file)
        self._load_features(cues)
        self.visual_interface = VisualRepresentationMCN()
        self.lang_interface = LanguageRepresentationMCN(max_words)
        self.tef_interface = None
        if loc:
            self.tef_interface = TemporalEndpointFeature()
        self.eval = False
        if test:
            self.eval = True
        self._set_feat_dim()

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
        source_id = moment_i.get('source', float('nan'))

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

        # Return source_id to use it for weighting schemes on the loss funct
        # idx is unnecesary and can be removed after debugging
        return (idx, source_id, sentence_feature, len_query,
                pos_visual_feature, neg_intra_visual_feature,
                neg_inter_visual_feature)

    def collate_data(self, batch):
        idxs, source_ids, *all_tensors = default_collate(batch)
        # Sort due to LSTM dealing with variable length
        al_s, idx = all_tensors[1].sort(descending=True)
        a_s = all_tensors[0][idx, ...]
        a_s.requires_grad_()
        dicts_of_tensors = (
            {k: v[idx, ...].requires_grad_() for k, v in i.items()}
            for i in all_tensors[2:])
        return (idxs, source_ids, a_s, al_s) + tuple(dicts_of_tensors)

    def collate_test_data(self, batch):
        # Note: we could do batching but taking care of the length of the
        # sentence was a mess.
        assert len(batch) == 1
        tensors = list(batch[0][:2])
        for i, v in enumerate(batch[0][2:]):
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

    def _set_feat_dim(self):
        "Set visual and language size"
        if self.eval:
            pass
        instance_feature = self[0]
        self.feat_dim = {'language_size': instance_feature[2].shape[1]}
        status = [self.feat_dim.update({f'visual_size_{k}': v.shape[-1]})
                  for k, v in instance_feature[4].items() if 'mask' not in k]

    @property
    def language_size(self):
        return self.feat_dim['language_size']

    @property
    def visual_size(self):
        return {k[12:]: v for k, v in self.feat_dim.items()
                if 'visual_size' in k}

    @property
    def max_words(self):
        return self.lang_interface.max_words


class DidemoSMCN(DidemoMCN):
    "Data feeder for SMCM"

    def __init__(self, json_file, cues=None, loc=True, max_words=50,
                 test=False, context=True):
        self._setup_list(json_file)
        self._load_features(cues)
        self.visual_interface = VisualRepresentationSMCN(context=context)
        self.lang_interface = LanguageRepresentationMCN(max_words)
        self.tef_interface = None
        if loc:
            self.tef_interface = TemporalEndpointFeature()
        self.eval = False
        if test:
            self.eval = True
        self._set_feat_dim()

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

    @timeit
    def __init__(self, json_file, cues=None, loc=False, max_words=50,
                 test=False, context=False, sampling_scheme='skip',
                 sampler_kwargs={'epoch': 0}, DEBUG=False):
        self.DEBUG = DEBUG
        self._setup_list(json_file)
        self._load_features(cues)
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
        # TODO: add these to other Didemo* dataset classes
        self._set_feat_dim()
        assert sampling_scheme in SAMPLING_SCHEMES
        setattr(self, 'sampling_scheme',
                getattr(self, '_sampling_' + sampling_scheme))
        self.sampler_kwargs = sampler_kwargs

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
        "Sample visual feature inside the video"
        source_id = self.metadata[p_ind]['video']
        source_idx = self.metadata[p_ind]['source']
        if source_idx == SourceID.VIDEO:
            return super(DidemoSMCNHeterogeneous,
                         self)._negative_intra_sampling(p_ind, p_time)
        else:
            feat_dict = self._compute_visual_feature(source_id, p_time)
            # We do not need to make it zero, but it's convenient for
            # debugging purposes
            status = [v.fill(0) for k, v in feat_dict.items() if k != 'mask']
            return feat_dict

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

    def update_sampler(self, epoch, sampler):
        """Update way in which instances are sampled from this Dataset

        TODO: TLDR, I don't get paid by designing great abstractions.
            It sounds that we are mixing Dataset, Sampler and training loop
            logic here :S; but no one cares as long as it works.
        """
        self.sampling_scheme(epoch, sampler)

    def _sampling_finetune(self, epoch, sampler):
        "Start with SourceID.IMAGE and switch to SourceID.VIDEO later"
        # Aja! I realized that this should be on the training loop logic
        # a.k.a Engine in TNT. I don't have such thing, thus this is a hack :)
        if epoch == 0:
            sampler.set_indices(self.source[SourceID.IMAGE])
            return None
        elif epoch != self.sampler_kwargs['epoch']:
            return None
        sampler.set_indices(self.source[SourceID.VIDEO])

    def _sampling_joint(self, epoch, sampler):
        "Minibatches with SourceID.IMAGE and SourceID.VIDEO"
        ind_videos = self.source[SourceID.VIDEO]
        # TODO: this ratio can be a hyper-parameters
        cap = min(len(ind_videos), len(self.source[SourceID.IMAGE]))
        ind_images = random.sample(self.source[SourceID.IMAGE], k=cap)
        sampler.set_indices(ind_videos + ind_images)

    def _sampling_skip(self, *args):
        pass


@unique
class RetrievalMode(IntEnum):
    MOMENT_TO_DESCRIPTION = 0
    DESCRIPTION_TO_MOMENT = 1
    VIDEO_TO_DESCRIPTION = 2


class DidemoSMCNRetrieval(DidemoSMCN):

    def __init__(self, json_file, cues=None, loc=True, max_words=50,
                 test=False, context=True,
                 mode=RetrievalMode.MOMENT_TO_DESCRIPTION):
        self._setup_list(json_file)
        self._set_metadata_per_video()
        self._load_features(cues)
        self.visual_interface = VisualRepresentationSMCN(context=context)
        self.lang_interface = LanguageRepresentationMCN(max_words)
        self.tef_interface = None
        if loc:
            self.tef_interface = TemporalEndpointFeature()
        self.eval = False
        self.mode = mode
        # self._set_feat_dim()

    def __getitem__(self, idx):
        if self.mode == RetrievalMode.MOMENT_TO_DESCRIPTION:
            return self._get_moment(idx)
        elif self.mode == RetrievalMode.DESCRIPTION_TO_MOMENT:
            return self._get_phrase(idx)
        elif self.mode == RetrievalMode.VIDEO_TO_DESCRIPTION:
            return self._get_video(idx)
        else:
            raise

    def _get_moment(self, idx):
        moment_i = self.metadata[idx]
        video_id = moment_i['video']
        num_annotators = len(moment_i['times'])
        # TODO: make it flexible, but deterministic
        annot_i = 0
        time = moment_i['times'][annot_i]
        pos_visual_feature = self._compute_visual_feature(video_id, time)
        return idx, pos_visual_feature

    def _get_phrase(self, idx):
        moment_i = self.metadata[idx]
        query = moment_i['language_input']
        # TODO: pack next two vars into a dict
        sentence_feature = self.lang_interface(query)
        len_query = len(query)
        return idx, sentence_feature, len_query

    def _get_video(self, idx):
        video_id, _ = self.metada_per_video[idx]
        visual_feature = self._compute_visual_feature_eval(video_id)
        return idx, visual_feature

    def __len__(self):
        if self.mode == RetrievalMode.VIDEO_TO_DESCRIPTION:
            return len(self.metada_per_video)
        return super().__len__()


class FakeLanguageRepresentation():
    "Allow faster iteration while I learn to cache stuff ;P"

    def __init__(self, max_words=50, dim=300):
        self.max_words = max_words
        self.dim = dim

    def __call__(self, query):
        return np.random.rand(self.max_words, self.dim).astype(np.float32)


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
    dataset = DidemoSMCNHeterogeneous(json_file=filename, cues=cues,
                                      DEBUG=True)
    print(f'Time loading {filename}: ', time.time() - t_start)
    idxs = random.choices(dataset.source[SourceID.IMAGE], k=5)
    for i in idxs:
        data = dataset[i]
        assert data[0] == i
        assert data[1] == SourceID.IMAGE
        np.testing.assert_array_almost_equal(data[4]['mask'],
                                             [1] + [0] * 5)
        assert data[5]['rgb'].sum() == 0

    # Unit-test Didemo
    VAL_LIST_PATH = 'data/raw/val_data_wwa.json'
    RGB_FEAT_PATH = 'data/interim/didemo/resnet152/320x240_max.h5'
    args = dict(test=False, context=False, loc=False,
                cues=dict(rgb=dict(file=RGB_FEAT_PATH)))
    val_dataset = DidemoSMCNRetrieval(VAL_LIST_PATH, **args)
