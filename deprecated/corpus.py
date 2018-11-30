import hashlib
import itertools
from collections import OrderedDict

import h5py
import numpy as np

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


