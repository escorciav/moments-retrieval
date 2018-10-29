import hashlib
import itertools
import json
from collections import OrderedDict

import h5py
import numpy as np
import torch

from np_segments_ops import non_maxima_suppresion

TIME_UNIT = 5
POSSIBLE_SEGMENTS = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)]
for i in itertools.combinations(range(6), 2):
    POSSIBLE_SEGMENTS.append(i)


class Corpus(object):
    """DEPRECATED. Corpus of videos with clips of interest to index

    TODO
        batch indexing
    """
    segments = np.array(POSSIBLE_SEGMENTS)
    segments_time = TIME_UNIT * np.array(POSSIBLE_SEGMENTS)
    segments_time[:, 1] += TIME_UNIT

    def __init__(self, filename, videos=None,
                 nms_threshold=1.0, topk=None):
        self.nms_threshold = nms_threshold
        self.topk = topk
        self._create_repo(filename, videos)
        self._create_feature_matrix()

    def index(self, x):
        distance = self.search(x)
        distance = self.postprocess(distance)
        distance_sorted_idx = np.argsort(distance)
        distance_sorted = distance[distance_sorted_idx]
        video_idx, segment_idx = self.ind_to_repo(distance_sorted_idx)
        return video_idx, segment_idx, distance_sorted

    def ind_to_repo(self, i):
        "retrieve video and segment index"
        # TODO: the name sucks
        # purpose: given index in matrix return video and segment index
        video_index = i // self.T
        segment_index = i % self.T
        return video_index, segment_index

    def nms_per_video(self, distance, copy):
        "nms accross clips of each video"
        if copy:
            distance = distance.copy()
        max_value = distance.max() + distance.min() / 100
        ind_possible_segments = np.arange(self.T)
        for i in range(self.num_videos):
            video_index = i * self.T
            d_i = distance[video_index:video_index + self.T]
            s_i = d_i.max() - d_i
            ind_pick = non_maxima_suppresion(
                self.segments_time, s_i, self.nms_threshold)
            ind_rm = np.setdiff1d(ind_possible_segments, ind_pick)
            ind_rm_corpus = video_index + ind_rm
            distance[ind_rm_corpus] = max_value
        return distance

    def postprocess(self, distance, copy=False):
        "apply postprocessing functions"
        # TODO generalize it for similarities
        if self.nms_threshold < 1.0:
            distance = self.nms_per_video(distance, copy)
        if self.topk is not None:
            distance = self.topk_per_video(distance, copy)
        return distance

    def repo_to_ind(self, video_index, segment_index):
        """return index in the corpus

        Args:
            video_index (int scalar or ndarray)
            segment_index (int scalar or ndarray)

        Note:
            if both are ndarray, they must have the same shape.
        """
        # TODO: the name sucks
        return video_index * self.T + segment_index

    def search(self, x):
        "compute similarity between query and elements in corpus"
        # TODO generalize it for similarities
        distance = ((self.features - x)**2).sum(axis=1)
        return distance

    def topk_per_video(self, distance, copy):
        "nms accross clips of each video"
        if copy:
            distance = distance.copy()
        max_value = distance.max() + distance.min() / 100
        distance__ = distance.reshape((-1, self.T))
        ind_rm = np.argsort(distance__, axis=1)[:, self.topk:]
        row_offset = (np.arange(self.num_videos).reshape((-1, 1)) *
                      self.T)
        ind_rm = (ind_rm + row_offset).ravel()
        distance[ind_rm] = max_value
        return distance

    def _create_repo(self, filename, videos):
        "read hdf5 and make corpus repo"
        self.container = OrderedDict()
        with h5py.File(filename, 'r') as f:
            if videos is None:
                videos = list(f.keys())
            else:
                is_1darray = (isinstance(videos, np.ndarray) and
                              videos.ndim == 1)
                assert isinstance(videos, list) or is_1darray
            for k in videos:
                self.container[k] = f[k][:]
        self.videos = np.array(videos)
        self.T, self.D = self._grab_sample_value().shape
        self.num_videos = len(self.videos)
        assert self.T == len(self.segments)

    def _create_feature_matrix(self):
        "make corpus matrix"
        # purpose: perform search without loop over repo
        dtype = self._grab_sample_value().dtype
        self.features = np.empty((self.num_videos * self.T, self.D),
                               dtype=dtype)
        for i, (_, v) in enumerate(self.container.items()):
            r_start = i * self.T
            r_end = r_start + self.T
            self.features[r_start:r_end, :] = v

    def _grab_sample_value(self, idx=0):
        sample_key = self.videos[idx]
        return self.container[sample_key]

    def __len__(self):
        return self.features.shape[0]


class CorpusAsDistanceMatrix():
    """DEPRECATED. Distance matrix of video corpus

    Retrieval by indexing columns of distance matrix pre-computed with
    moment_retrieval.py

    TODO: refactor to avoid code duplication
    """
    segments = np.array(POSSIBLE_SEGMENTS)
    segments2ind = {tuple(v): i for i, v in enumerate(segments)}
    segments_time = TIME_UNIT * np.array(POSSIBLE_SEGMENTS)
    segments_time[:, 1] += TIME_UNIT

    def __init__(self, filename, nms_threshold=1.0, topk=None):
        self.nms_threshold = nms_threshold
        self.topk = topk
        self._load_h5(filename)

    def index(self, description_id):
        ind = self.moments_id2ind[description_id]
        distance = self.d_matrix[ind, :]
        distance = self.postprocess(distance)
        distance_sorted_idx = np.argsort(distance)
        distance_sorted = distance[distance_sorted_idx]
        video_idx, segment_idx = self.ind_to_repo(distance_sorted_idx)
        return video_idx, segment_idx, distance_sorted

    def ind_to_repo(self, i):
        "retrieve video and segment index"
        # TODO: the name sucks
        # purpose: given index in matrix return video and segment index
        video_index = i // self.T
        segment_index = i % self.T
        return video_index, segment_index

    def nms_per_video(self, distance, copy):
        "nms accross clips of each video"
        if copy:
            distance = distance.copy()
        max_value = distance.max() + distance.min() / 100
        ind_possible_segments = np.arange(self.T)
        for i in range(self.num_videos):
            video_index = i * self.T
            d_i = distance[video_index:video_index + self.T]
            s_i = d_i.max() - d_i
            ind_pick = non_maxima_suppresion(
                self.segments_time, s_i, self.nms_threshold)
            ind_rm = np.setdiff1d(ind_possible_segments, ind_pick)
            ind_rm_corpus = video_index + ind_rm
            distance[ind_rm_corpus] = max_value
        return distance

    def postprocess(self, distance, copy=False):
        "apply postprocessing functions"
        # TODO generalize it for similarities
        if self.nms_threshold < 1.0:
            distance = self.nms_per_video(distance, copy)
        if self.topk is not None:
            distance = self.topk_per_video(distance, copy)
        return distance

    def repo_to_ind(self, video_index, segment_index):
        """return index in the corpus

        Args:
            video_index (int scalar or ndarray)
            segment_index (int scalar or ndarray)

        Note:
            if both are ndarray, they must have the same shape.
        """
        # TODO: the name sucks
        return video_index * self.T + segment_index

    def topk_per_video(self, distance, copy):
        "nms accross clips of each video"
        if copy:
            distance = distance.copy()
        max_value = distance.max() + distance.min() / 100
        distance__ = distance.reshape((-1, self.T))
        ind_rm = np.argsort(distance__, axis=1)[:, self.topk:]
        row_offset = (np.arange(self.num_videos).reshape((-1, 1)) *
                      self.T)
        ind_rm = (ind_rm + row_offset).ravel()
        distance[ind_rm] = max_value
        return distance

    def _load_h5(self, filename):
        self.filename = filename
        with h5py.File(filename, 'r') as f:
            self.d_matrix = f['prediction_matrix'][:]
            assert f['similarity'].value == False
            # data-structures for mappings
            self.videos_ind2id, self.videos_id2ind = {}, {}
            for ind, id in f['_video_index'][:]:
                self.videos_ind2id[ind] = id
                self.videos_id2ind[id] = ind
            self.moments_id2ind = {}
            for ind, id in f['_moments_index'][:]:
                self.moments_id2ind[id] = ind
            self.num_videos = len(self.videos_ind2id)
            self.T = self.d_matrix.shape[1] // self.num_videos

    def video_to_iid(self, video):
        # return video integer id
        return (int(hashlib.sha256(video.encode('utf-8')).hexdigest(), 16) %
                10**8)

    def video_to_id(self, video):
        # return video id
        return self.videos_id2ind[self.video_to_iid(video)]

    def segment_to_ind(self, segment):
        # return segment index
        return self.segments2ind[tuple(segment)]


class CorpusVideoMomentRetrievalBase():
    """Composite abstraction for scalable moment retrieval from video corpus

    For simplicity the database is held in memory, and we perform exhaustive
    search. However, the abstraction was conceived to throw all the history of
    efficient indexing, such as PQ-codes, into it.

    Notes:
        - Our scientific tables are torch tensors `video_indices`,
          `proposals`, `moments_tables`, `entries_per_video` amenable for
          indexing. There is a lot of repetition in many of them which could
          be exploited to make them compact.
        - `models` have a `search` method that returns a vector with the size
           of the table i.e. all the elements that we give to them. Such that
           we can do late fusion of models.
        - Works for models that learnt distance functions btw embedding.
          Extending to similarities should be trivial (I guess).
    """

    def __init__(self, dataset, dict_of_models, alpha=None):
        self.dataset = dataset
        self.models = dict_of_models
        if alpha is None:
            self.alpha = {key: 1 /len(dict_of_models)
                          for key in dict_of_models}
        assert dataset.cues.keys() == dict_of_models.keys()
        assert dict_of_models.keys() == self.alpha.keys()
        self.num_videos = None
        self.num_moments = None
        self.video_indices = None
        self.entries_per_video = None
        self.proposals = None
        self.moments_tables = None

    def indexing(self):
        "Create tables to retrieve videos, proposals and store codes"
        raise NotImplementedError('Subclass and implement your indexing')

    def preprocess_description(self, description):
        "Return tensors representing description as 1) vectors and 2) length"
        assert isinstance(description, list)
        # TODO (release): allow to tokenize description
        lang_feature_, len_query_ = self.dataset._compute_language_feature(
            description)
        # torchify
        lang_feature = torch.from_numpy(lang_feature_)
        lang_feature.unsqueeze_(0)
        len_query = torch.tensor([len_query_])
        return lang_feature, len_query

    def postprocess(self, distance):
        "apply postprocessing functions"
        raise NotImplementedError('WIP')

    def query(self, description):
        "Return videos and moments aligned with a text description"
        raise NotImplementedError('Subclass and implement your indexing')


class MomentRetrievalFromProposalsTable(CorpusVideoMomentRetrievalBase):
    """Retrieve Moments which aligns with pre-defined proposals

    This abstraction suits MCN kind of models that embed a whole segment into
    a common visual-text embedding space.
    """

    def __init__(self, *args, **kwargs):
        super(MomentRetrievalFromProposalsTable, self).__init__(
            *args, **kwargs)

    def indexing(self):
        "Create database of moments in videos"
        num_entries_per_video = []
        codes = {key: [] for key in self.models}
        all_proposals = []
        # TODO (tier-2;design): define method in dataset to do this?
        # batchify the fwd-pass
        for video_id in self.dataset.videos:
            representation_dict, proposals_ = (
                self.dataset._compute_visual_feature_eval(video_id))
            num_entries_per_video.append(len(proposals_))

            # torchify
            for key, value in representation_dict.items():
                representation_dict[key] = torch.from_numpy(value)
            proposals = torch.from_numpy(proposals_)

            # Append items to database
            all_proposals.append(proposals)
            for key in self.dataset.cues:
                segment_rep_k = representation_dict[key]
                # get codes of the proposals -> S_i x D matrix
                # S_i := num proposals in ith-video
                codes[key].append(
                    self.models[key].visual_encoder(segment_rep_k))
        # Form the S x D matrix.
        # M := number of videos, S = \sum_{i=1}^M S_i
        # We have as many tables as visual cues
        self.moments_tables = {key: torch.cat(value)
                               for key, value in codes.items()}
        # TODO (tier-2; design): organize this better
        self.num_videos = len(num_entries_per_video)
        self.entries_per_video = torch.tensor(num_entries_per_video)
        self.proposals = torch.cat(all_proposals)
        self.num_moments = int(self.proposals.shape[0])
        video_indices = np.repeat(
            np.arange(0, len(self.dataset.videos)),
            num_entries_per_video)
        self.video_indices = torch.from_numpy(video_indices)

    def query(self, description):
        "Search moments based on a text description given as list of words"
        lang_feature, len_query = self.preprocess_description(description)
        score_list, descending_list = [], []
        for key, model_k in self.models.items():
            lang_code = model_k.encode_query(lang_feature, len_query)
            scores_k, descending_k = model_k.search(
                lang_code, self.moments_tables[key])
            score_list.append(scores_k * self.alpha[key])
            descending_list.append(descending_k)
        scores = sum(score_list)
        # assert np.unique(descending_list).shape[0] == 1
        scores, ind = scores.sort(descending=descending_k)
        # TODO (tier-1): enable bell and whistles
        return self.video_indices[ind], self.proposals[ind, :]


class MomentRetrievalFromClipBasedProposalsTable(
        CorpusVideoMomentRetrievalBase):
    """Retrieve Moments using a clip based model

    This abstraction suits SMCN kind of models that the representation of the
    video clips into a common visual-text embedding space.

    Note:
        - Make sure to setup the dataset in a way that retrieves a 2D
          `numpy:ndarray` with the representation of all the proposals and a
          1D `numpy:ndarray` with the number of clips per segment as `mask`.
        - currently this implementation deals with the more general case of
          non-decomposable models. Note that decomposable models would admit
          smaller tables.
    """

    def __init__(self, *args, **kwargs):
        super(MomentRetrievalFromClipBasedProposalsTable, self).__init__(
            *args, **kwargs)
        self.clips_per_moment = None
        self.clips_per_moment_list = None

    def indexing(self):
        "Create database of moments in videos"
        num_entries_per_video = []
        clips_per_moment = []
        codes = {key: [] for key in self.models}
        all_proposals = []
        # TODO (tier-2;design): define method in dataset to do this?
        # batchify the fwd-pass
        for video_id in self.dataset.videos:
            representation_dict, proposals_ = (
                self.dataset._compute_visual_feature_eval(video_id))
            num_entries_per_video.append(len(proposals_))
            num_clips_i = representation_dict['mask']
            if num_clips_i.ndim != 1:
                raise ValueError('Dataset setup incorrectly. Disable padding')

            # torchify
            for key, value in representation_dict.items():
                if key == 'mask': continue
                # get representation of all proposals
                representation_dict[key] = torch.from_numpy(value)
            proposals = torch.from_numpy(proposals_)
            clips_per_moment.append(torch.from_numpy(num_clips_i))

            # Append items to database
            all_proposals.append(proposals)
            for key in self.dataset.cues:
                segment_rep_k = representation_dict[key]
                # get codes of the proposals -> C_i x D matrix
                # Given a video i with S_i number of prooposals
                # Each proposal S_i spans c_ij clips of the i-th video.
                # C_i = \sum_{j=1}^{S_i} c_ij := num clips over all S_i
                # proposals in the i-th video
                codes[key].append(
                    self.models[key].visual_encoder(segment_rep_k))
        # Form the C x D matrix
        # M := number of videos, C = \sum_{i=1}^M C_i
        # We have as many tables as visual cues
        self.moments_tables = {key: torch.cat(value)
                               for key, value in codes.items()}
        # TODO (tier-2; design): organize this better
        self.num_videos = len(num_entries_per_video)
        self.entries_per_video = torch.tensor(num_entries_per_video)
        self.proposals = torch.cat(all_proposals)
        self.num_moments = int(self.proposals.shape[0])
        video_indices = np.repeat(
            np.arange(0, len(self.dataset.videos)),
            num_entries_per_video)
        self.video_indices = torch.from_numpy(video_indices)
        self.clips_per_moment = torch.cat(clips_per_moment)
        self.clips_per_moment_list = self.clips_per_moment.tolist()
        self.clips_per_moment = self.clips_per_moment.float()

    def query(self, description):
        "Search moments based on a text description given as list of words"
        lang_feature, len_query = self.preprocess_description(description)
        score_list, descending_list = [], []
        for key, model_k in self.models.items():
            lang_code = model_k.encode_query(lang_feature, len_query)
            scores_k, descending_k = model_k.search(
                lang_code, self.moments_tables[key], self.clips_per_moment,
                self.clips_per_moment_list)
            score_list.append(scores_k * self.alpha[key])
            descending_list.append(descending_k)
        scores = sum(score_list)
        # assert np.unique(descending_list).shape[0] == 1
        scores, ind = scores.sort(descending=descending_k)
        # TODO (tier-1): enable bell and whistles
        return self.video_indices[ind], self.proposals[ind, :]


if __name__ == '__main__':
    _filename = 'data/interim/mcn/corpus_val_rgb.hdf5'
    _corpus = Corpus(_filename)
    # test video and segment mapping
    _index = np.random.randint(len(_corpus.features))
    print(_index, _corpus.ind_to_repo(_index))

    _filename = 'data/interim/mcn/queries_val_rgb.hdf5'
    with h5py.File(_filename, 'r') as fid:
        _sample_key = list(fid.keys())[0]
        _sample_value = fid[_sample_key][:]
    _corpus.index(_sample_value)

    # Unit-test
    from dataset_untrimmed import UntrimmedMCN
    from proposals import SlidingWindowMSFS
    from model import MCN
    np.random.seed(1701)

    charades_data = 'data/processed/charades-sta'
    dataset_setup = dict(
        json_file=f'{charades_data}/test.json',
        cues={'rgb': {'file': f'{charades_data}/rgb_resnet152_max_cs-3.h5'}},
        loc=True,
        context=True,
        debug=True,
        eval=True,
        proposals_interface=SlidingWindowMSFS(
            length=3,
            num_scales=8,
            stride=3,
            unique=True
        )
    )
    dataset = UntrimmedMCN(**dataset_setup)
    arch_setup = dict(
        visual_size=dataset.visual_size['rgb'],
        lang_size=dataset.language_size,
        max_length=dataset.max_words,
        embedding_size=100,
        visual_hidden=500,
        lang_hidden=1000,
        visual_layers=1
    )
    model = {'rgb': MCN(**arch_setup)}
    subject = MomentRetrievalFromProposalsTable(dataset, model)
    subject.indexing()
    ind = np.random.randint(0, len(dataset))
    subject.query(dataset.metadata[ind]['language_input'])