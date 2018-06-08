import json
from collections import OrderedDict

import numpy as np


class Queries():
    "Dataset of queries for ground-truth data"
    # TODO: pytorchify with Dataset style?

    def __init__(self, filename, videos, segments):
        self.data = OrderedDict()
        self.filename = filename
        self.diff_four = [0, 0]  # debugging
        self._sanitize(videos, segments)
        self._setup_from_file(filename)

    def _add_list_items(self, l):
        for i in l:
            query_id = i.pop('annotation_id')
            i['video_index'] = self.lookup_video[i['video']]
            i['segment_indices'] = self._get_segment_indexes(i['times'])
            self.data[query_id] = i
        self.query_ids = list(self.data.keys())

    def _get_segment_indexes(self, annotations):
        idxs = np.array([self.lookup_segment[tuple(i)] for i in annotations])
        # debugging
        if len(annotations) > 4:
            self.diff_four[0] += 1
        elif len(annotations) < 4:
            self.diff_four[1] += 1
        return idxs

    def _sanitize(self, videos, segments):
        if videos is None or segments is None:
            raise ValueError
        assert len(set(videos)) == len(videos)
        assert len(set(segments)) == len(segments)
        self.lookup_video = dict(zip(videos, range(len(videos))))
        self.lookup_segment = dict(zip(segments, range(len(segments))))

    def _setup_from_file(self, filename):
        with open(filename, 'r') as f:
            self._add_list_items(json.load(f))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from corpus import Corpus
    _filename = 'data/interim/mcn/corpus_val_rgb.hdf5'
    _corpus = Corpus(_filename)
    _filename = 'data/raw/val_data.json'
    _segments = list(map(tuple, _corpus.segments.tolist()))
    _val_gt = Queries(_filename, _corpus.videos.tolist(), _segments)
    _val_gt[0]