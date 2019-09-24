"Group multiple methods to generate salient temporal windows in a video"
import itertools

import numpy as np
import json
import random

PROPOSAL_SCHEMES = ['DidemoICCV17SS', 'SlidingWindowMSRSS','RC3D']


class TemporalProposalsBase():
    "Base class (signature) to generate temporal candidate in video"

    def __call__(self, video_id, metadata=None, feature_collection=None):
        raise NotImplementedError('Implement with the signature above')


class DidemoICCV17SS(TemporalProposalsBase):
    """Original search space of moments proposed in ICCV-2017

    Attributes:
        clip_length_min (float) : minimum length, in seconds, of a video clip.
        proposals (numpy array) : of shape [21, 2] representing all the
            possible temporal segments of valid annotations of DiDeMo dataset.
            It represents the search space of a temporal localization
            algorithm.

    Reference: Hendricks et al. Localizing Moments in Video with Natural
        Language. ICCV 2017.
    """
    clip_length_min = 5.0

    def __init__(self, *args, dtype=np.float32, **kwargs):
        clips_indices = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        for i in itertools.combinations(range(len(clips_indices)), 2):
            clips_indices.append(i)
        self.proposals = np.array(clips_indices, dtype=dtype)
        self.proposals *= self.clip_length_min
        self.proposals[:, 1] += self.clip_length_min

    def __call__(self, *args, **kwargs):
        return self.proposals


class SlidingWindowMSFS(TemporalProposalsBase):
    """Multi-scale (linear) sliding window with fixed stride

    TODO:
        - We are considering to deprecated this abstraction. Indeed, it's
          disabled from training.
        - documentation.
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


class SlidingWindowMSRSS(TemporalProposalsBase):
    """Multi-scale sliding window with relative stride within the same scale

    Attributes:
        length (float) : length of smallest window.
        scales (sequence of int) : duration of moments relative to
            `lenght`.
        stride (float) : relative stride between two windows with the same
            duration. We used different strides for each scale rounding it
            towards a multiple of `length`. Note that the minimum stride is
            `length` for any window will be the `length` itself.
        dtype (numpy.dtype) : TODO

    TODO: documentation
    """

    def __init__(self, length, scales, stride=0.5, dtype=np.float32):
        self.length = length
        self.scales = scales
        self.relative_stride = stride
        # pick strides per scale that are multiples of length
        self.strides = [max(round(i * stride), 1) * length for i in scales]
        self.dtype = dtype
        assert len(scales) > 0

    def sliding_windows(self, t_end, t_start=0):
        "sliding canonical windows over a given time interval"
        windows_ = []
        for i, stride in enumerate(self.strides):
            num_i = np.ceil((t_end - t_start)/ stride)
            windows_i = np.empty((int(num_i), 2), dtype=np.float32)
            windows_i[:, 0] = np.arange(t_start, t_end, stride)
            windows_i[:, 1] = windows_i[:, 0] + self.length * self.scales[i]
            windows_i[windows_i[:, 1] > t_end, 1] = t_end
            windows_.append(windows_i)
        windows = np.concatenate(windows_, axis=0)
        # Hacky way to make windows fit inside video
        # It implies windows at the end may not belong to the set spanned by
        # length and scales.
        return np.unique(windows, axis=0)

    def __call__(self, video_id, metadata=None, feature_collection=None):
        duration = metadata.get('duration')
        assert duration is not None
        return self.sliding_windows(duration)


class RC3D(TemporalProposalsBase):

    def __init__(self, length, scales, stride=0.5, dtype=np.float32):
        self.SW_proposals_gen = SlidingWindowMSRSS(length, scales)
        #Load R-C3D proposals
        self.RC3D_proposals = {}
        filename = './data/raw/RC3D_ranked_proposals_charades.json'
        with open(filename, 'r') as f:
            self.RC3D_proposals = json.load(f)
        self.complete_proposals = True #Enable or disable the generation of random proposals 
 
    def get_RC3D_proposals(self, video_id, duration, number_sliding_window_proposals):
        '''
        The function uses the precoumputed proposals using RC3D to generate a number of proposals 
        that is the same as the number of proposals we would have generated through a sliding window 
        procedure. 
        We match the number of proposals and the vale inside the proposals to be coherent with what we desider.
        '''
        proposals = self.RC3D_proposals[video_id]                   # Retrieve proposals for specific video
        missing = number_sliding_window_proposals - len(proposals)  # Check the number of proposals
        missing = missing * (missing>0)                             # Just apply relu
        # Check the proposals and reduce the time in the right interval
        proposals, removed = self._check_duration_vs_proposals(proposals, duration)
        # Generate data if missing
        if missing+removed > 0 and self.complete_proposals:
            proposals = self._generate_missing_proposals(missing+removed, proposals, duration)
        # cut the number of RC3D proposals to the number of proposals of the sliding windonw
        proposals = proposals[:number_sliding_window_proposals]    
        if self.complete_proposals: 
            assert number_sliding_window_proposals-len(proposals)>= 0 # Check that sizes are ok
        return np.asarray(proposals, dtype=np.float32)

    def _generate_missing_proposals(self, missing, proposals, duration):
        '''
        The fuction is used to generate random proposals such that the number 
        is the same as the sliding window procedure
        '''
        for _ in range(missing):
            t_start, t_end = 0,0
            while not t_end-t_start > 0:
                t_start=random.randint(0,duration*10)/10
                t_end=random.randint(0,duration*10)/10
            proposals.append([t_start,t_end])
        return proposals

    def _check_duration_vs_proposals(self, proposals, duration):
        '''
        The function check is the values of the proposals are in the range [0, duration]
        if both the values are outside the range the proposal get discarded.
        If only the second value is outside the range the we cut it down to duration.
        '''
        new_proposals = []
        for p in proposals:
            if p[0] < duration and p[1] < duration:
                new_proposals.append(p)
            elif p[0] > duration and p[1] > duration:
                pass
            elif p[0] < duration and p[1] > duration:
                new_proposals.append([p[0], duration])
        return new_proposals, len(proposals)-len(new_proposals)

    def _devel(self, video_id, metadata, feature_collection, duration):
        '''
            DEPRECATED
        '''
        #Check that all keys are in there:
        RC3D_keys = list(self.RC3D_proposals.keys())
        feature_keys = list(feature_collection["rgb"].keys())
        num_sliding, num_RC3D = 0, 0
        for k in feature_keys:
            num_sliding += len(feature_collection['rgb'][video_id])
            if not k in RC3D_keys:
                print('Key {} is missing'.format(k))
            num_RC3D += len(self.get_RC3D_proposals(k, duration, len(self.SW_proposals_gen(
                video_id, metadata=metadata, feature_collection=feature_collection))))
        print('Tot number of sliding window proposals: {}'.format(num_sliding))
        print('Tot number of RC3D proposals: {}'.format(num_RC3D))

    def __call__(self, video_id, metadata=None, feature_collection=None):
        duration = metadata.get('duration')
        assert duration is not None
        sliding_window_proposals = self.SW_proposals_gen(
            video_id, metadata=metadata, feature_collection=feature_collection)
        number_sliding_window_proposals = len(sliding_window_proposals)
        # self._devel(self, video_id, metadata, feature_collection, duration)   DEPRECATED
        return self.get_RC3D_proposals(video_id, duration, number_sliding_window_proposals)


if __name__ == '__main__':
    test_fns_args = [(SlidingWindowMSFS, (3, 5, 3)),
                     (DidemoICCV17SS, (),),
                     (SlidingWindowMSRSS, (1.5, [2, 4, 6, 12]))]
    for fn_i, args_i in test_fns_args:
        proposal_fn = fn_i(*args_i)
        x = proposal_fn('hola', {'duration': 15})
        if fn_i == DidemoICCV17SS:
            assert len(x) == 21
