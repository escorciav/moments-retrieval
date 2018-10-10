"Group multiple methods to generate salient temporal windows in a video"
import numpy as np


class TemporalProposalsBase():
    "Base class (signature) to generate temporal candidate in video"

    def __call__(self, video_id, metadata=None, feature_collection=None):
        raise NotImplementedError('Implement with the signature above')


class SlidingWindowMSFS(TemporalProposalsBase):
    """Multi-scale (linear) sliding window with fixed stride

    TODO: documentation
    """

    def __init__(self, length, num_scales, stride, unique=False):
        self.length = length
        self.num_scales = num_scales
        self.stride = stride
        self.unique = unique
        self.canonical_windows = np.zeros((num_scales, 2))
        self.canonical_windows[:, 1] += (
            length * np.arange(1, num_scales + 1))

    def sliding_windows(self, t_end, t_start=0):
        "sliding canonical windows over a given time interval"
        t_zero = np.arange(t_start, t_end, self.stride)
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
    proposal_fn = SlidingWindowMSFS(3, 5, 3)
    x = proposal_fn('hola', {'duration': 15})