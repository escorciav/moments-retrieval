import itertools
import json
from collections import OrderedDict

import h5py
import numpy as np

POSSIBLE_SEGMENTS = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)]
for i in itertools.combinations(range(6), 2):
    POSSIBLE_SEGMENTS.append(i)


class Corpus(object):
    """Corpus of videos with clips of interest to index

    TODO
        batch indexing
    """

    def __init__(self, filename, videos=None, segments=POSSIBLE_SEGMENTS):
        self.segments = np.array(segments)
        self._create_repo(filename, videos)
        self._create_feature_matrix()

    def index(self, x):
        distance = self.search(x)
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