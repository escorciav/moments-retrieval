"Dataset abstractions to deal with data of various length"
import json
import random

import h5py
import numpy as np
from torch.utils.data import Dataset

from didemo import LanguageRepresentationMCN, FakeLanguageRepresentation
from didemo import sentences_to_words, normalization1d
from utils import dict_of_lists
from utils import timeit

TIME_UNIT = 3  # 3 seconds due to psychological evidence


class UntrimmedBase(Dataset):
    "Base Dataset for pairs moment-description in videos of various length"

    def __init__(self):
        self.json_file = None
        self.cues = None
        self.features = None
        self.max_clips = None
        self.metadata = None
        self.metadata_per_video = None
        self._video_list = None
        self.time_unit = TIME_UNIT

    def max_number_of_clips(self):
        "Return maximum number of clips/chunks over all videos in dataset"
        if self.max_clips is None:
            if self.metadata_per_video is None:
                raise ValueError('Dataset is empty. Run setup first')
            max_clips = 0
            for vid_metadata in self.metadata_per_video.values():
                max_clips = max(max_clips, vid_metadata.get('num_clips', 0))
            self.max_clips = max_clips
        return self.max_clips

    @property
    def videos(self):
        "Iterator over videos in Dataset"
        if self._video_list is None:
            self._video_list = list(self.metadata_per_video.keys())
        return self._video_list

    def _load_features(self, cues):
        """Load visual features (coarse chunks) in memory

        TODO:
            - refactor duplicated code with Didemo
        """
        self.cues = cues
        if cues is None:
            return
        self.features = dict.fromkeys(cues.keys())
        for key, params in cues.items():
            with h5py.File(params.get('file', 'NO-filename'), 'r') as fid:
                self.features[key] = {}
                for video_id in self.metadata_per_video:
                    self.features[key][video_id] = fid[video_id][:]

    def _preprocess_descriptions(self):
        "Tokenize descriptions into words"
        for moment_i in self.metadata:
            # TODO: update to use spacy or allennlp
            moment_i['language_input'] = sentences_to_words(
                [moment_i['description']])

    def _setup_list(self, filename):
        "Read JSON file with all moments i.e. segment and description"
        self.json_file = filename
        with open(filename, 'r') as fid:
            data = json.load(fid)
            self.metadata = data['moments']
            self.metadata_per_video = data['videos']
            self.time_unit = data['time_unit']
            self._update_metadata_per_video()
            self._update_metadata()
        self._preprocess_descriptions()

    def _shrink_dataset(self):
        "Make single video dataset to debug video corpus moment retrieval"
        # TODO(tier-2;release): log if someone triggers this
        ind = random.randint(0, len(self.videos) - 1)
        self._video_list = [self._video_list[ind]]
        video_id = self._video_list[0]
        moment_indices = self.metadata_per_video[video_id]['moment_indices']
        self.metadata = [self.metadata[i] for i in moment_indices]
        self.metadata_per_video[video_id]['moment_indices'] = list(
            range(len(self.metadata)))

    def _update_metadata(self):
        """Add keys to items in attribute:metadata plus extra update of videos

        `video_index` field corresponds to the unique identifier of the video
        that contains the moment.
        Transforms `times` into numpy array for training.
        """
        for i, moment in enumerate(self.metadata):
            video_id = self.metadata[i]['video']
            self.metadata[i]['times'] = np.array(
                moment['times'], dtype=np.float32)
            self.metadata[i]['video_index'] = (
                self.metadata_per_video[video_id]['index'])
            self.metadata_per_video[video_id]['moment_indices'].append(i)

    def _update_metadata_per_video(self):
        """Add keys to items in attribute:metadata_per_video

        `index` field corresponds to a unique identifier for the video in the
        dataset to evaluate moment retrieval from a corpus.
        `moment_indices` field corresponds to the indices of the moments that
        comes from a particular video.
        """
        for i, key in enumerate(self.metadata_per_video):
            self.metadata_per_video[key]['index'] = i
            # This field is populated by _update_metadata
            self.metadata_per_video[key]['moment_indices'] = []

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        raise NotImplementedError


class UntrimmedBasedMCNStyle(UntrimmedBase):
    """Untrimmed abstract Dataset for MCN kind models

    TODO:
        batch during evaluation
        negative mining and batch negative sampling
    """
    MAGIC_TIOU = 0.3

    def __init__(self, json_file, cues=None, loc=True, max_words=50,
                 eval=False, context=True, proposals_interface=None,
                 no_visual=False, debug=False):
        super(UntrimmedBasedMCNStyle, self).__init__()
        self._setup_list(json_file)
        self._load_features(cues)
        self.eval = eval
        self.loc = loc
        self.context = context
        self.debug = debug
        self.no_visual = no_visual
        self.visual_interface = None
        self.tef_interface = None
        self.proposals_interface = None
        # clean this, glove of original MCN is really slow, it kills fast
        # iteration during debugging :) (yes, I could cache but dahh)
        self.lang_interface = FakeLanguageRepresentation(
            max_words=max_words)
        if not debug:
            self.lang_interface = LanguageRepresentationMCN(max_words)
        if self.loc:
            self.tef_interface = TemporalEndpointFeature()
        else:
            # we need features to align with language
            assert not no_visual
        if self.eval:
            self.eval = True
            self.proposals_interface = proposals_interface
            assert self.proposals_interface is not None

    @property
    def language_size(self):
        "dimension of word embeddings"
        return self.feat_dim['language_size']

    @property
    def visual_size(self):
        "dimension of visual features"
        return {k[12:]: v for k, v in self.feat_dim.items()
                if 'visual_size' in k}

    @property
    def max_words(self):
        "max number of words per description"
        return self.lang_interface.max_words

    def _compute_language_feature(self, query):
        "Get language representation of words in query"
        # TODO: pack next two vars into a dict
        feature = self.lang_interface(query)
        len_query = min(len(query), self.max_words)
        return feature, len_query

    def _compute_visual_feature(self, video_id, moment_loc, video_duration):
        raise NotImplementedError

    def _compute_visual_feature_eval(self, video_id):
        raise NotImplementedError

    def _eval_item(self, idx):
        "Return anchor, positive, None*2, gt_segments, candidate_segments"
        moment_i = self.metadata[idx]
        gt_segments = moment_i['times']
        query = moment_i['language_input']
        video_id = moment_i['video']

        pos_visual_feature, segments = self._compute_visual_feature_eval(
            video_id)
        neg_intra_visual_feature = None
        neg_inter_visual_feature = None
        words_feature, len_query = self._compute_language_feature(query)
        num_segments = len(segments)
        len_query = [len_query] * num_segments
        words_feature = np.tile(words_feature, (num_segments, 1, 1))

        # TODO: return a dict to avoid messing around with indices
        argout = (words_feature, len_query, pos_visual_feature,
                  neg_intra_visual_feature, neg_inter_visual_feature,
                  gt_segments, segments)
        if self.debug:
            # TODO: deprecate source_id
            source_id = moment_i.get('source', float('nan'))
            return (idx, source_id) + argout
        return argout

    def __getitem__(self, idx):
        if self.eval:
            return self._eval_item(idx)
        return self._train_item(idx)

    def _negative_intra_sampling(self, idx, moment_loc):
        "Sample another moment inside the video"
        moment_i = self.metadata[idx]
        video_id = moment_i['video']
        video_duration = self.metadata_per_video[video_id]['duration']
        tiou = 1
        while tiou >= self.MAGIC_TIOU:
            # sample segment
            sampled_loc = [random.random() * video_duration,
                           random.random() * video_duration]
            sampled_loc = [min(sampled_loc), max(sampled_loc)]

            # compute tIOU
            i_end = min(sampled_loc[1], moment_loc[1])
            i_start = max(sampled_loc[0], moment_loc[0])
            intersection = max(0, i_end - i_start)

            u_end = max(sampled_loc[1], moment_loc[1])
            u_start = min(sampled_loc[0], moment_loc[0])
            union = u_end - u_start

            tiou = intersection / union
        return self._compute_visual_feature(
            video_id, sampled_loc, video_duration)

    def _negative_inter_sampling(self, idx, moment_loc):
        "Sample another moment from other video as in original MCN paper"
        moment_i = self.metadata[idx]
        video_id = moment_i['video']
        video_duration = self.metadata_per_video[video_id]['duration']
        other_video = video_id
        while other_video == video_id:
            idx = int(random.random()*len(self.metadata))
            other_video = self.metadata[idx]['video']
        # MCN-strategy as close as possible
        other_video_duration = self.metadata_per_video[other_video]['duration']
        sampled_loc = moment_loc
        if other_video_duration < video_duration:
            sampled_loc = [random.random() * other_video_duration,
                           random.random() * other_video_duration]
            sampled_loc = [min(sampled_loc), max(sampled_loc)]
        return self._compute_visual_feature(
            other_video, sampled_loc, other_video_duration)

    def _train_item(self, idx):
        "Return anchor, positive, negatives"
        moment_i = self.metadata[idx]
        time = moment_i['time']
        query = moment_i['language_input']
        video_id = moment_i['video']
        video_i = self.metadata_per_video[video_id]
        video_duration = video_i['duration']

        pos_visual_feature = self._compute_visual_feature(
            video_id, time, video_duration)
        # Sample negatives
        neg_intra_visual_feature = self._negative_intra_sampling(idx, time)
        neg_inter_visual_feature = self._negative_inter_sampling(idx, time)
        words_feature, len_query = self._compute_language_feature(query)

        argout = (words_feature, len_query, pos_visual_feature,
                  neg_intra_visual_feature, neg_inter_visual_feature)
        if self.debug:
            # TODO: deprecate source_id
            source_id = moment_i.get('source', float('nan'))
            return (idx, source_id) + argout
        return argout

    def _set_feat_dim(self):
        "Set visual and language size"
        ind = 2 if self.debug else 0
        instance = self[0]
        if self.eval:
            instance = ((instance[0 + ind][0, ...],) +
                        instance[1 + ind:3 + ind])
            ind = 0
        self.feat_dim = {'language_size': instance[0 + ind].shape[1]}
        _ = [self.feat_dim.update({f'visual_size_{k}': v.shape[-1]})
             for k, v in instance[2 + ind].items() if 'mask' not in k]


class UntrimmedMCN(UntrimmedBasedMCNStyle):
    "Data feeder for MCN"

    def __init__(self, *args, **kwargs):
        super(UntrimmedMCN, self).__init__(*args, **kwargs)
        self.visual_interface = VisualRepresentationMCN(context=self.context)
        self._set_feat_dim()

    def _compute_visual_feature(self, video_id, moment_loc, video_duration):
        "Return visual features plus TEF for a given segment in the video"
        if self.no_visual:
            return self._only_tef(video_id, moment_loc, video_duration)
        feature_collection = {}
        for key, feat_db in self.features.items():
            feature_video = feat_db[video_id]
            moment_feat_k = self.visual_interface(
                feature_video, moment_loc, time_unit=self.time_unit)
            if self.tef_interface:
                moment_feat_k = np.concatenate(
                    [moment_feat_k,
                     self.tef_interface(moment_loc, video_duration)]
                )

            feature_collection[key] = moment_feat_k.astype(
                np.float32, copy=False)
        return feature_collection

    def _compute_visual_feature_eval(self, video_id):
        "Return visual features plus TEF for candidate segments in video"
        # We care about corpus video moment retrieval thus our
        # `proposals_interface` does not receive language queries.
        metadata = self.metadata_per_video[video_id]
        candidates = self.proposals_interface(
            video_id, metadata=metadata, feature_collection=self.features)
        candidates_rep = dict_of_lists(
            [self._compute_visual_feature(video_id, t, metadata['duration'])
             for t in candidates]
        )
        num_segments = len(candidates)
        for k, v in candidates_rep.items():
            candidates_rep[k] = np.concatenate(v).reshape((num_segments, -1))
        return candidates_rep, candidates

    def _only_tef(self, video_id, moment_loc, video_duration):
        feature_collection = {
            'tef': self.tef_interface(moment_loc, video_duration)
        }
        feature_collection['tef'] = feature_collection['tef'].astype(
            np.float32, copy=False)
        return feature_collection


class UntrimmedSMCN(UntrimmedBasedMCNStyle):
    """Data feeder for SMCN

    Attributes
        padding (bool): if True the representation is padded with zeros.
    """

    def __init__(self, *args, max_clips=None, padding=True, **kwargs):
        super(UntrimmedSMCN, self).__init__(*args, **kwargs)
        self.padding = padding
        if not self.eval:
            max_clips = self.max_number_of_clips()
        self.visual_interface = VisualRepresentationSMCN(
            context=self.context, max_clips=max_clips)
        self._set_feat_dim()

    def _compute_visual_feature(self, video_id, moment_loc, video_duration):
        """Return visual features plus TEF for a given segment in the video

        Note:
            This implementation deals with non-decomposable features such
            as TEF. In practice, if you can decompose your model/features
            it's more efficient to re-write the final pooling.
        """
        feature_collection = {}
        for key, feat_db in self.features.items():
            feature_video = feat_db[video_id]
            moment_feat_k, mask = self.visual_interface(
                feature_video, moment_loc, time_unit=self.time_unit)
            if self.tef_interface:
                T, N = mask.sum().astype(np.int), len(moment_feat_k)
                tef_feature = np.zeros((N, 2), dtype=self.tef_interface.dtype)
                tef_feature[:T, :] = self.tef_interface(
                    moment_loc, video_duration)
                moment_feat_k = np.concatenate(
                    [moment_feat_k, tef_feature], axis=1)

            feature_collection[key] = moment_feat_k.astype(
                np.float32, copy=False)
        # whatever masks is fine given that we don't consider time responsive
        # features yet?
        dtype = np.float32 if self.padding else np.int64
        feature_collection['mask'] = mask.astype(dtype, copy=False)
        return feature_collection

    def _compute_visual_feature_eval(self, video_id):
        "Return visual features plus TEF for all candidate segments in video"
        metadata = self.metadata_per_video[video_id]
        # We care about corpus video moment retrieval thus our
        # `proposals_interface` does not receive language queries.
        candidates = self.proposals_interface(
            video_id, metadata=metadata, feature_collection=self.features)
        candidates_rep = dict_of_lists(
            [self._compute_visual_feature(video_id, t, metadata['duration'])
             for t in candidates]
        )
        for k, v in candidates_rep.items():
            if self.padding:
                candidates_rep[k] = np.stack(v)
            else:
                candidates_rep[k] = np.concatenate(v, axis=0)
        return candidates_rep, candidates

    def set_padding(self, padding):
        "Change padding mode"
        self.padding = padding
        self.visual_interface.padding = padding


class TemporalEndpointFeature():
    """Compute TEF for MCN model

    TODO:
        documentation
        force input to be numpy darray
    """

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, moment_loc, duration):
        return np.array(moment_loc, dtype=self.dtype) / duration


class VisualRepresentationMCN():
    """Compute visual features for MCN model

    Redefine MCN feature extraction to deal with varied length 1D video
    features. This class extracts features as in the original MCN formulation,
    which did not consider varied length videos.
    We apply KIS and explicit principle by keeping moments in seconds instead
    of moving into a frame unit. In this way any issue is constrained to the
    feature extraction and pre-processing stage without affecting the
    evaluation which must be done in seconds as it's the unit that humans
    understand and in which the data is provided.
    Given that moments are in second and features are packed with a time unit
    fixed the programmer, the only difference with
    `class::didemo.VisualRepresentationMCN` is the use of rounding to
    determine the features to pool.

    TODO:
        documentation
        force inputs to be numpy darray
        consider linear interpolation to extract feature instead of rounding.
    """

    def __init__(self, context=True, dtype=np.float32, eps=1e-6):
        self.context = context
        self.size_factor = context + 1
        self.dtype = dtype
        self.eps = eps

    def __call__(self, features, moment_loc, time_unit=TIME_UNIT):
        f_dim = features.shape[1]
        data = np.empty(f_dim * self.size_factor, dtype=self.dtype)
        # From time to units of time
        # we substract a small amount of t_end to ensure that it's close to
        # the unit of time in case t_end == time_unit
        # The end index is inclusive, check `function:normalization1d`.
        im_start = int(moment_loc[0] // time_unit)
        im_end = int((moment_loc[1] - self.eps) // time_unit)
        data[0:f_dim] = normalization1d(im_start, im_end, features)
        if self.context:
            ic_start, ic_end = 0, len(features) - 1
            data[f_dim:2 * f_dim] = normalization1d(
                ic_start, ic_end, features)
        return data


class VisualRepresentationSMCN():
    """Compute visual features for SMCN model

    The SMCN model does not pool the feature vector inside a moment, instead
    it returns a numpy array of shape [N, D] and a numpy array of shape [N]
    denoting a masked visual representation and the binary mask associated
    with a given moment, respectively. N corresponds to the legnth of the
    video or the max_clips when it's not `None`. For more details e.g.
    what's D?, take a look at the details in`class:VisualRepresentationMCN`.

    TODO:
        - create base class to remove duplicated code
    """

    def __init__(self, context=True, dtype=np.float32, eps=1e-6,
                 max_clips=None, padding=True):
        self.context = context
        self.size_factor = context + 1
        self.dtype = dtype
        self.eps = eps
        self.max_clips = max_clips
        self.padding = padding

    def __call__(self, features, moment_loc, time_unit=TIME_UNIT):
        n_feat, f_dim = features.shape
        if self.max_clips is not None:
            n_feat = self.max_clips
        # From time to units of time
        # we substract a small amount of t_end to ensure that it's close to
        # the unit of time in case t_end == time_unit
        # The end index is inclusive, check `function:normalization1d`.
        im_start = int(moment_loc[0] // time_unit)
        im_end = int((moment_loc[1] - self.eps) // time_unit)
        # T := \mathcal{T} but in this case is the cardinality of the set
        T = im_end - im_start + 1
        if not self.padding:
            n_feat = T
        padded_data = np.zeros((n_feat, f_dim * self.size_factor),
                               dtype=self.dtype)
        # mask is numpy array of type self.dtype to avoid upstream casting
        mask = np.zeros(n_feat, dtype=self.dtype)

        padded_data[:T, 0:f_dim] = self._local_feature(
            im_start, im_end, features)
        mask[:T] = 1
        if self.context:
            ic_start, ic_end = 0, len(features) - 1
            padded_data[:T, f_dim:2 * f_dim] = normalization1d(
                ic_start, ic_end, features)
        if self.padding:
            return padded_data, mask
        return padded_data, np.array([T])

    def _local_feature(self, start, end, x):
        "Return normalized representation of each clip/chunk"
        y = x[start:end + 1, :]
        scaling_factor = np.linalg.norm(y, axis=1, keepdims=True) + self.eps
        return y / scaling_factor


if __name__ == '__main__':
    import time
    from proposals import SlidingWindowMSFS
    # Kinda Unit-test
    print('UntrimmedMCN\n\t* Train')
    json_data = 'data/processed/charades-sta/train.json'
    h5_file = 'data/processed/charades-sta/rgb_resnet152_max_cs-3.h5'
    cues = {'rgb': {'file': h5_file}}
    t_start = time.time()
    dataset = UntrimmedMCN(json_data, cues, debug=True)
    ind = int(random.random() * len(dataset))
    print(f'\tTime loading dataset: {time.time() - t_start}')
    print('\tNumber of moments: ', len(dataset))
    print(f'\tSample moment ({ind}): ', dataset.metadata[ind])
    item = dataset[ind]
    assert len(item) == 7
    for i, v in enumerate(item):
        if i < 2:
            # ignore index and source
            continue
        elif i == 2:
            assert isinstance(v, np.ndarray)
            print('\tlanguage feature of a moment', v.shape)
        elif i == 3:
            assert isinstance(v, int)
            print('\tNumber of words', v)
        elif i > 3:
            print(f'\t{i} Visual feature of a moment')
            assert isinstance(v, dict)
            for k, v in v.items():
                print('\tMode:', k, v.shape)
        else:
            raise ValueError('Fix UntrimmedMCN train!')
    dataset.debug = False
    item = dataset[ind]
    assert len(item) == 5

    print('\t* Eval')
    dataset.eval = True
    dataset.debug = True
    dataset.proposals_interface = SlidingWindowMSFS(3, 5, 3)
    item = dataset[ind]
    assert len(item) == 9
    for i, v in enumerate(item):
        if i < 2:
            # ignore index and source
            continue
        elif i == 2:
            assert isinstance(v, np.ndarray)
            print('\tlanguage feature of a moment', v.shape)
        elif i == 3:
            assert isinstance(v, list)
            print('\tNumber of words', np.unique(v))
        elif i == 4:
            print(f'\t{i} Visual feature of a moment')
            assert isinstance(v, dict)
            for k, v in v.items():
                print('\tMode:', k, v.shape)
        elif 4 < i < 7:
            assert v is None
        elif i >= 7:
            assert isinstance(v, np.ndarray)
            print('\tsegments', v.shape)
        else:
            raise ValueError('Fix UntrimmedMCN eval!')
    dataset.debug = False
    item = dataset[ind]
    assert len(item) == 7

    # Kinda Unit-test for UntrimmedSMCN
    # (json_data and cues come from previous unit-test
    print('UntrimmedSMCN\n\ttrain')
    t_start = time.time()
    dataset = UntrimmedSMCN(json_data, cues, debug=True)
    ind = int(random.random() * len(dataset))
    print(f'\tTime loading dataset: ', time.time() - t_start)
    print('\tNumber of moments: ', len(dataset))
    print(f'\tSample moment ({ind}): ', dataset.metadata[ind])
    item = dataset[ind]
    assert len(item) == 7
    for i, v in enumerate(item):
        if i < 2:
            # ignore index and source
            continue
        elif i == 2:
            assert isinstance(v, np.ndarray)
            print('\tlanguage feature of a moment', v.shape)
        elif i == 3:
            assert isinstance(v, int)
            print('\tNumber of words', v)
        elif i > 3:
            print(f'\t{i}, Visual feature of a moment')
            assert isinstance(v, dict)
            for k, v in v.items():
                print(f'\tMode:', k, v.shape)
        else:
            raise ValueError('Fix UntrimmedMCN train!')
    dataset.debug = False
    item = dataset[ind]
    assert len(item) == 5

    print('\teval')
    dataset.eval = True
    dataset.debug = True
    dataset.proposals_interface = SlidingWindowMSFS(3, 5, 3)
    item = dataset[ind]
    assert len(item) == 9
    for i, v in enumerate(item):
        if i < 2:
            # ignore index and source
            continue
        elif i == 2:
            assert isinstance(v, np.ndarray)
            print('\tlanguage feature of a moment', v.shape)
        elif i == 3:
            assert isinstance(v, list)
            print('\tNumber of words', np.unique(v))
        elif i == 4:
            print(f'\t{i}, Visual feature of a moment')
            assert isinstance(v, dict)
            for k, v in v.items():
                print(f'\tMode:', k, v.shape)
        elif 4 < i < 7:
            assert v is None
        elif i >= 7:
            assert isinstance(v, np.ndarray)
            print('\tsegments', v.shape)
        else:
            raise ValueError('Fix UntrimmedMCN eval!')
    dataset.debug = False
    item = dataset[ind]
    assert len(item) == 7
    print('\tpadding')
    dataset.set_padding(False)
    for k, v in dataset[ind][2].items():
        print(f'\tMode:', k, v.shape)