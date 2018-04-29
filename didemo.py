import itertools
import json
import random
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from glove import RecurrentEmbedding

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
    """DiDeMo dataset
    TODO:
        testing data
        enable multiple visual cues
        negative mining and batch negative sampling
        make sorting in collate optional
    """

    def __init__(self, json_file, rgb_file=None, flow_file=None,
                 loc=True, max_words=50, test=False):
        self._setup_list(json_file)
        self._load_features(rgb=rgb_file, flow=flow_file)
        self.visual_interface = VisualRepresentationMCN()
        self.lang_interface = LanguageRepresentationMCN(max_words)
        self.tef_interface = None
        if loc:
            self.tef_interface = TemporalEndpointFeature()
        self.eval = False
        if test:
            self.eval = True
    
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

    def _load_features(self, rgb=None, flow=None):
        """Read all features (coarse chunks) in memory
        TODO:
            Edit to only load features of videos in metadata
        """
        self.rgb_file = rgb
        self.rgb_features = None
        with h5py.File(rgb, 'r') as f:
            self.rgb_features = {i: v[:] for i, v in f.items()}

        self.flow_file = flow
        self.flow_features = None
        with h5py.File(rgb, 'r') as f:
            self.flow_features = {i: v[:] for i, v in f.items()}

    def _compute_visual_feature(self, video_id, time=None):
        "Pool visual feature and append TEF"
        video_feature = None
        if self.rgb_file:
            video_rgb = self.rgb_features[video_id]
            video_feature = self.visual_interface(*time, video_rgb)
        if self.flow_file:
            assert video_feature is None
            video_flow = self.rgb_features[video_id]
            video_feature = self.visual_interface(*time, video_flow)
        if self.tef_interface:
            video_feature = np.concatenate(
                [video_feature, self.tef_interface(time)])
        return video_feature.astype(np.float32)

    def _negative_intra_sampling(self, video_id, p_time):
        """Sample visual feature inside the video
        TODO:
            negative mining. Weak supervision?
        """
        if not isinstance(p_time, tuple):
            p_time = tuple(p_time)
        possible_n = list(POSSIBLE_SEGMENTS_SET - {p_time})
        random.shuffle(possible_n)
        n_time = possible_n[0]
        return self._compute_visual_feature(video_id, n_time)

    def _negative_inter_sampling(self, video_id, p_time):
        """Sample visual feature outside the video
        TODO:
            test other time intervals
        """
        other_video = video_id
        while other_video == video_id:
            idx = int(random.random()*len(self.metadata))
            other_video = self.metadata[idx]['video']
        return self._compute_visual_feature(other_video, p_time)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        moment_i = self.metadata[idx]
        video_id = moment_i['video']
        num_annotators = len(moment_i['times'])
        annot_i = random.randint(0, num_annotators - 1)
        time = moment_i['times'][annot_i]
        query = moment_i['language_input']

        sentence_feature = self.lang_interface(query)
        len_query = len(query)
        if self.eval:
            pos_visual_feature = np.concatenate(
                [self._compute_visual_feature(video_id, t)[np.newaxis, :]
                 for t in self.segments], axis=0)
            n_segments = len(self.segments)
            len_query = [len_query] * n_segments
            sentence_feature = np.tile(sentence_feature, (n_segments, 1, 1))
            neg_intra_visual_feature = None
            neg_inter_visual_feature = None
        else:
            pos_visual_feature = self._compute_visual_feature(video_id, time)
            # Sample negatives
            neg_intra_visual_feature = self._negative_intra_sampling(
                video_id, time)
            neg_inter_visual_feature = self._negative_inter_sampling(
                video_id, time)

        return (sentence_feature, len_query, pos_visual_feature,
                neg_intra_visual_feature, neg_inter_visual_feature)

    def collate_data(self, batch):
        all_tensors = default_collate(batch)
        # Sort due to LSTM dealing with variable length
        al_s, idx = all_tensors[1].sort(descending=True)
        tensors_minus_length = []
        for i in all_tensors[0:1] + all_tensors[2:]:
            tensors_minus_length.append(i[idx, ...])
            tensors_minus_length[-1].requires_grad_()
        a_s, p_s, niv_s, nid_s = tensors_minus_length
        return a_s, al_s, p_s, niv_s, nid_s

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
            else:
                tensors.append(None)
        return tensors


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


class TemporalEndpointFeature(object):
    "Relative position in the video"

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, start_end):
        return np.array(start_end, dtype=self.dtype) / 6

if __name__ == '__main__':
    import time
    print('simple test')
    data = 'data/raw/train_data.json'
    rgb = 'data/raw/average_fc7.h5'
    flow = 'data/raw/average_global_flow.h5'
    t_start = time.time()
    dataset = Didemo(data, rgb)
    print(f'Time loading {data}: ', time.time() - t_start)
    print(len(dataset))
    print(dataset.metadata[0])
    for i in dataset[0]:
        if isinstance(i, np.ndarray):
            print(i.shape)

    dataset.eval = True
    values = dataset[0]
    for i, v in enumerate(values):
        if isinstance(v, np.ndarray):
            print(i, v.shape)
        elif isinstance(v, list):
            print(i, len(v))
        if i > 2:
            assert v is None