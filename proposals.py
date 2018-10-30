"Group multiple methods to generate salient temporal windows in a video"
import itertools

import numpy as np

PROPOSAL_SCHEMES = ['DidemoICCV17SS', 'SlidingWindowMSFS']


class TemporalProposalsBase():
    "Base class (signature) to generate temporal candidate in video"

    def __call__(self, video_id, metadata=None, feature_collection=None):
        raise NotImplementedError('Implement with the signature above')


class DidemoICCV17SS(TemporalProposalsBase):
    "Original search space of moments proposed in ICCV-2017"
    time_unit = 5

    def __init__(self, *args, dtype=np.float32, **kwargs):
        clips_indices = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        for i in itertools.combinations(range(len(clips_indices)), 2):
            clips_indices.append(i)
        self.proposals = np.array(clips_indices, dtype=dtype)
        self.proposals *= self.time_unit
        self.proposals[:, 1] += self.time_unit

    def __call__(self, *args, **kwargs):
        return self.proposals


class SlidingWindowMSFS(TemporalProposalsBase):
    """Multi-scale (linear) sliding window with fixed stride

    TODO: documentation
    """

    def __init__(self, length, num_scales, stride, unique=False,
                 dtype=np.float32):
        self.length = length
        self.num_scales = num_scales
        self.stride = stride
        self.unique = unique
        self.dtype = dtype
        self.canonical_windows = np.zeros((num_scales, 2), dtype=self.dtype)
        self.canonical_windows[:, 1] += (
            length * np.arange(1, num_scales + 1))

    def sliding_windows(self, t_end, t_start=0):
        "sliding canonical windows over a given time interval"
        t_zero = np.arange(t_start, t_end, self.stride, dtype=self.dtype)
        windows = (np.tile(self.canonical_windows, (len(t_zero), 1)) +
                   np.repeat(t_zero, len(self.canonical_windows))[:, None])
        # hacky way to make windows fit inside video
        # this means the lengths of the windows at the end are not in the set
        # spanned by length and num_scales
        windows[windows[:, 1] > t_end, 1] = t_end
        if self.unique:
            return np.unique(windows, axis=0)
        return windows

    def __call__(self, video_id, metadata=None, feature_collection=None):
        duration = metadata.get('duration')
        assert duration is not None
        return self.sliding_windows(duration)


if __name__ == '__main__':
    test_fns_args = [(SlidingWindowMSFS, (3, 5, 3)),
                     (DidemoICCV17SS, ())]
    for fn_i, args_i in test_fns_args:
        proposal_fn = fn_i(*args_i)
        x = proposal_fn('hola', {'duration': 15})
        if fn_i == DidemoICCV17SS:
            assert len(x) == 21
