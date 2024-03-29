"Dataset abstractions to deal with data of various length"
import json
import random
import re
from enum import IntEnum, unique

import h5py
import numpy as np
from scipy.signal import convolve
from torch.utils.data import Dataset

from glove import RecurrentEmbedding
from np_segments_ops import iou as segment_iou
from utils import dict_of_lists, unique2d_perserve_order


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


@unique
class TemporalFeatures(IntEnum):
    NONE = 0
    TEMPORAL_ENDPOINT = 1
    TEMPORALLY_AWARE = 2
    START_POINT = 3

    @staticmethod
    def from_string(s):
        # credits to https://bit.ly/2Mvu9bz
        try:
            return TemporalFeatures[s]
        except KeyError:
            raise ValueError()


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
        self.clip_length = None

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
    def num_videos(self):
        "Number of videos in the Dataset"
        return len(self.videos)

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
        self.features = dict.fromkeys(cues.keys())
        time_units = []
        for key, params in cues.items():
            if params is None:
                continue
            with h5py.File(params.get('file', 'NO-filename'), 'r') as fid:
                self.features[key] = {}
                for video_id in self.metadata_per_video:
                    self.features[key][video_id] = fid[video_id][:]
                    # HDF5 contains info about the video that is not practical
                    # to include in the JSON
                    self.metadata_per_video[video_id]['num_clips'] = len(
                        fid[video_id])
                time_unit = fid.get('metadata/time_unit')
                if time_unit is not None:
                    time_unit = time_unit.value
                time_units.append(time_unit)

        # Update time_unit of the dataset
        if len(time_units) == 0:
            return
        elif len(set(time_units)) > 1:
            raise ValueError(
                'Handling multiple features with different time_unit is '
                'tricky. Do it at your own discretion.')
        else:
            self.clip_length = time_units[0]

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
            self._update_metadata_per_video()
            self._update_metadata()
        self._preprocess_descriptions()

    def _shrink_dataset(self):
        "Make single video dataset to debug video corpus moment retrieval"
        # TODO(tier-2;release): log if someone triggers this
        ind = random.randint(0, self.num_videos - 1)
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
            # `time` field may get deprecated
            self.metadata[i]['time'] = None

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

    def _video_duration(self, video_id):
        "Return duration of a given video"
        return self.metadata_per_video[video_id]['duration']

    def _video_num_clips(self, video_id):
        "Return number of clips of a given video"
        return self.metadata_per_video[video_id]['num_clips']

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        raise NotImplementedError


class UntrimmedBasedMCNStyle(UntrimmedBase):
    """Untrimmed abstract Dataset for MCN kind models

    TODO:
        batch during evaluation
        negative mining and batch negative sampling

    Attributes:
        _prob_querytovideo (2D numpy-array): matrix used to sample negative
            videos for each query.
    """

    def __init__(self, json_file, cues=None,
                 loc=TemporalFeatures.TEMPORAL_ENDPOINT,
                 max_words=50, eval=False, context=True,
                 proposals_interface=None, no_visual=False, sampling_iou=0.35,
                 ground_truth_rate=1, prob_nproposal_nextto=-1,
                 clip_length=None, h5_nis=None, nis_k=None, debug=False):
        super(UntrimmedBasedMCNStyle, self).__init__()
        self._setup_list(json_file)
        self._load_features(cues)
        self.eval = eval
        self.loc = loc
        self.context = context
        self.debug = debug
        self.no_visual = no_visual
        self.visual_interface = None
        self._set_tef_interface(loc)
        self._sampling_iou = sampling_iou
        self.proposals_interface = proposals_interface
        self._ground_truth_rate = ground_truth_rate
        self._prob_neg_proposal_next_to = prob_nproposal_nextto
        self.h5_nis = h5_nis
        self.nis_k = nis_k
        self._prob_querytovideo = None
        # clean this, glove of original MCN is really slow, it kills fast
        # iteration during debugging :) (yes, I could cache but dahh)
        self.lang_interface = FakeLanguageRepresentation(
            max_words=max_words)
        if not debug:
            self.lang_interface = LanguageRepresentationMCN(max_words)
        if self.eval:
            self.eval = True
            assert self.proposals_interface is not None
        # UntrimmedBase was designed to hold visual features, thus ignoring
        # visual information is held outside it
        if self.no_visual:
            if clip_length is None:
                raise ValueError(
                    'Please provide the clip length (seconds) as this is a'
                    'property grabbed from the HDF5. Missing in this case.')
            self.clip_length = clip_length
        self._setup_neg_importance_sampling()

    @property
    def decomposable(self):
        "If True -> model can be decomposed into clips"
        raise NotImplementedError('Class property')

    @property
    def language_size(self):
        "dimension of word embeddings"
        return self.feat_dim['language_size']

    @property
    def max_words(self):
        "max number of words per description"
        return self.lang_interface.max_words

    @property
    def sampling_iou(self):
        "IoU value used to sample negative during training"
        # TODO: add setter to implement mining scheme?
        return self._sampling_iou

    @property
    def visual_size(self):
        "dimension of visual features"
        return {k[12:]: v for k, v in self.feat_dim.items()
                if 'visual_size' in k}

    def video_item(self, idx):
        "Return visual description of all possible moments in a given video"
        video_id = self.videos[idx]
        pos_visual_feature, segments = self._compute_visual_feature_eval(
            video_id)
        return pos_visual_feature, segments

    def video_proposals(self, idx):
        "Return proposals (candidate moments) for a given video"
        video_id = self.videos[idx]
        metadata = self.metadata_per_video[video_id]
        proposals = self.proposals_interface(video_id, metadata)
        return proposals

    def _compute_language_feature(self, query):
        "Get language representation of words in query"
        # TODO: pack next two vars into a dict
        feature = self.lang_interface(query)
        len_query = min(len(query), self.max_words)
        return feature, len_query

    def _compute_visual_feature(self, video_id, moment_loc):
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

    def _get_tef_feature(self, moment_loc, video_id,):
        "Return TEF feature for a given instance"
        video_duration = self._video_duration(video_id)
        tef_feature = self.tef_interface(
            moment_loc, video_duration, clip_length=self.clip_length)
        # Disable TEF-dropout. This was an experimental thing which we may
        # remove later.
        # tef_feature = np.zeros((2,), dtype=np.float32)
        # Dropout is implemented as inputting random values betweem 0-1
        # tef_feature = np.random.uniform(0,1,(2,))
        # if random.random() >= self.dropout_tef:
        #     tef_feature = self.tef_interface(
        #     moment_loc, video_duration, clip_length=self.clip_length)
        # if self.eval and self.dropout_tef > 0.0:
        #     tef_feature=np.asarray([0.5,0.5])

        return tef_feature

    def _negative_intra_sampling(self, idx, moment_loc):
        "Sample another moment inside the video"
        moment_i = self.metadata[idx]
        video_id = moment_i['video']
        if random.random() <= self._prob_neg_proposal_next_to:
            sampled_loc = self._proposal_next_to_moment(idx, moment_loc)
        elif self.proposals_interface is None:
            video_duration = self._video_duration(video_id)
            sampled_loc = self._random_proposal_sampling(
                video_duration, moment_loc)
        else:
            metadata = self.metadata_per_video[video_id]
            proposals = self.proposals_interface(video_id, metadata)
            iou_matrix = segment_iou(proposals, moment_loc[None, :])
            indices = (iou_matrix < self.sampling_iou).nonzero()[0]
            if len(indices) > 0:
                ind = indices[random.randint(0, len(indices) - 1)]
                sampled_loc = proposals[ind, :]
            else:
                video_duration = self._video_duration(video_id)
                sampled_loc = self._random_proposal_sampling(
                    video_duration, moment_loc)
        return self._compute_visual_feature(video_id, sampled_loc)

    def _proposal_next_to_moment(self, idx, moment_loc):
        "Return a proposal next to moment_loc of length <= moment length"
        if not hasattr(self.proposals_interface, 'stride'):
            raise ValueError('Unsupported proposal interface')
        t_start, t_end = moment_loc[1], moment_loc[1]
        clip_length = self.clip_length
        moment_i = self.metadata[idx]
        video_id = moment_i['video']
        duration = self._video_duration(video_id)
        sampled_loc = np.empty_like(moment_loc)

        # sample duration at random
        moment_num_clips = max(int((t_end - t_start) // clip_length), 1)
        sample_num_clips = random.randint(1, moment_num_clips)
        sample_length = sample_num_clips * clip_length
        if random.random() >= 0.5:
            if t_start - sample_length >= 0:
                sampled_loc[1] = t_start
                sampled_loc[0] = t_start - sample_length
                return sampled_loc
        else:
            if t_end + sample_length <= duration:
                sampled_loc[0] = t_end
                sampled_loc[1] = t_end + sample_length
                return sampled_loc
        return self._random_proposal_sampling(duration, moment_loc)

    def _negative_inter_sampling(self, idx, moment_loc):
        "Sample another moment from other video as in original MCN paper"
        prob_videos = self._prob_querytovideo[idx, :]
        # tech taming humam: Bug in numpy
        # https://github.com/numpy/numpy/issues/8317
        prob_videos = prob_videos.astype(float, copy=False)
        prob_videos /= prob_videos.sum()
        neg_video_ind = np.random.multinomial(1, prob_videos).nonzero()[0][0]
        other_video = self.videos[neg_video_ind]
        video_id = self.metadata[idx]['video']

        # MCN-ICCV2017 strategy as close as possible
        video_duration = self._video_duration(video_id)
        other_video_duration = self._video_duration(other_video)
        sampled_loc = moment_loc
        if other_video_duration < video_duration:
            sampled_loc = self._random_proposal_sampling(other_video_duration)
        return self._compute_visual_feature(other_video, sampled_loc)

    def _proposal_augmentation(self, moment_loc, video_id):
        "positive data augmentation"
        if random.random() > self._ground_truth_rate:
            metadata = self.metadata_per_video[video_id]
            proposals = self.proposals_interface(video_id, metadata)
            iou_matrix = segment_iou(proposals, moment_loc[None, :])
            # 0.6 is the mean btw 0.5 and 0.7 :sweat_smile:
            indices = (iou_matrix > 0.6).nonzero()[0]
            if len(indices) > 0:
                ind = indices[random.randint(0, len(indices) - 1)]
                sampled_loc = proposals[ind, :]
                return sampled_loc
        return moment_loc

    def _random_proposal_sampling(self, video_duration, moment_loc=None):
        "Sample a proposal with random start and end point"
        tiou = 1
        while tiou >= self.sampling_iou:
            # sample segment
            sampled_loc = [random.random() * video_duration,
                           random.random() * video_duration]
            sampled_loc = [min(sampled_loc), max(sampled_loc)]

            if moment_loc is None:
                break
            i_end = min(sampled_loc[1], moment_loc[1])
            i_start = max(sampled_loc[0], moment_loc[0])
            intersection = max(0, i_end - i_start)

            u_end = max(sampled_loc[1], moment_loc[1])
            u_start = min(sampled_loc[0], moment_loc[0])
            union = u_end - u_start

            tiou = intersection / union
        return sampled_loc

    def _set_feat_dim(self):
        "Set visual and language size"
        ind = 2 if self.debug else 0
        instance = self[0]
        if self.eval:
            instance = ((instance[0 + ind][0, ...],) +
                        instance[1 + ind:3 + ind])
            ind = 0
        self.feat_dim = {'language_size': instance[0 + ind].shape[1]}
        for key in self.features:
            self.feat_dim.update(
                {f'visual_size_{key}': instance[2 + ind][key].shape[-1]}
            )

    def _setup_neg_importance_sampling(self):
        "Define sampling prob for videos and moments"
        # 1. Init neg sampling to uniform dist
        num_queries = len(self)
        num_videos = self.num_videos
        prob_querytovideoid = np.empty(
            (num_queries, num_videos), dtype=np.float32)
        prob_querytovideoid[:None, :] = 1 / num_videos
        # 1.1 zero-out prob of sampling the same video
        for query_ind, query_data in enumerate(self.metadata):
            video_ind = query_data['video_index']
            prob_querytovideoid[query_ind, video_ind] = 0
        # 1.2 re-normalize probability
        prob_querytovideoid /= prob_querytovideoid.sum(axis=1, keepdims=True)

        # TODO: prob_querytomoment
        # max_num_proposals = None
        # self._prior_querytomomentid = np.zeros(
        #     (num_queries, num_videos, max_num_proposals), dtype=np.float32)
        # raise NotImplementedError('WIP')

        # No importance sampling
        if self.h5_nis is None:
            self._prob_querytovideo = prob_querytovideoid
            return

        # 2. Importance sampling can be casted as updating P(video) based on
        #   P(query | video).
        # 2.1 Generate P(query | video) from video ranking from each query
        #   Here. we will use the a pdf derived from 1 / x i.e. videos
        #   retrieved first will be sampled often but this will decay rapidly
        with h5py.File(self.h5_nis, 'r') as fid:
            # Num-queries x Num-moments matrix, i-th columns correspond to
            # rank-ith
            ranked_video_indices_per_query = fid['vid_indices'][:]
        if ranked_video_indices_per_query.shape[1] > self.num_videos:
            ranked_video_indices_per_query = unique2d_perserve_order(
                ranked_video_indices_per_query)
        if self.nis_k:
            self.nis_k = min(self.nis_k, self.num_videos)
            ranked_video_indices_per_query = ranked_video_indices_per_query[
                :, :self.nis_k].reshape(-1)
            ind = np.repeat(np.arange(num_queries), self.nis_k)
            prob_query_given_video = np.zeros(
                (num_queries, num_videos), dtype=np.float32)
            prob_query_given_video[
                ind, ranked_video_indices_per_query] = 1 / self.nis_k

        else:
            rank_prob = 1 / np.arange(1, num_videos + 1, dtype=np.float32)
            rank_prob /= sum(rank_prob)
            prob_query_given_video = np.empty(
                (num_queries, num_videos), dtype=np.float32)
            # TODO: vectorize this
            for i in range(num_queries):
                prob_query_given_video[
                    i, ranked_video_indices_per_query[i]] = rank_prob
        # 2.2 Update P(video) with P(video) * P(query | video)
        #   Shall we wrap this inside a for loop?
        prob_querytovideoid = prob_query_given_video * prob_querytovideoid
        prob_querytovideoid /= prob_querytovideoid.sum(axis=1, keepdims=True)
        self._prob_querytovideo = prob_querytovideoid

    def _set_tef_interface(self, loc):
        "Setup interface to get moment location feature"
        # TODO(tier-2;enhacement). Is there a nicer way to do this?
        if loc == TemporalFeatures.NONE:
            self.tef_interface = None
            assert not self.no_visual
        elif loc == TemporalFeatures.TEMPORAL_ENDPOINT:
            self.tef_interface = TemporalEndpointFeature()
        elif loc == TemporalFeatures.TEMPORALLY_AWARE:
            self.tef_interface = TemporallyAwareFeature()
        else:
            raise ValueError('Unsupported')

    def _train_item(self, idx):
        "Return anchor, positive, negatives"
        moment_i = self.metadata[idx]
        query = moment_i['language_input']
        video_id = moment_i['video']
        # Sample a positive annotations if there are multiple
        t_start_end = moment_i['times'][0, :]
        if len(moment_i['times']) > 1:
            ind_t = random.randint(0, len(moment_i['times']) - 1)
            t_start_end = moment_i['times'][ind_t, :]

        t_start_end = self._proposal_augmentation(t_start_end, video_id)
        pos_visual_feature = self._compute_visual_feature(
            video_id, t_start_end)
        # Sample negatives
        neg_intra_visual_feature = self._negative_intra_sampling(
            idx, t_start_end)
        neg_inter_visual_feature = self._negative_inter_sampling(
            idx, t_start_end)
        words_feature, len_query = self._compute_language_feature(query)

        argout = (words_feature, len_query, pos_visual_feature,
                  neg_intra_visual_feature, neg_inter_visual_feature)
        if self.debug:
            # TODO: deprecate source_id
            source_id = moment_i.get('source', float('nan'))
            return (idx, source_id) + argout
        return argout


class UntrimmedMCN(UntrimmedBasedMCNStyle):
    "Data feeder for MCN"

    def __init__(self, *args, **kwargs):
        super(UntrimmedMCN, self).__init__(*args, **kwargs)
        self.visual_interface = VisualRepresentationMCN(context=self.context)
        self._set_feat_dim()

    @property
    def decomposable(self):
        return False

    def _compute_visual_feature(self, video_id, moment_loc):
        "Return visual features plus TEF for a given segment in the video"
        if self.no_visual:
            return self._only_tef(video_id, moment_loc)

        feature_collection = {}
        video_duration = self._video_duration(video_id)
        num_clips = self._video_num_clips(video_id)
        for key, feat_db in self.features.items():
            feature_video = feat_db[video_id]
            moment_feat_k = self.visual_interface(
                feature_video, moment_loc, clip_length=self.clip_length,
                num_clips=num_clips)
            if self.tef_interface:
                moment_feat_k = np.concatenate(
                    [moment_feat_k,
                     self.tef_interface(moment_loc, video_duration,
                                        clip_length=self.clip_length)]
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
            [self._compute_visual_feature(video_id, t) for t in candidates]
        )
        num_segments = len(candidates)
        for k, v in candidates_rep.items():
            candidates_rep[k] = np.concatenate(v).reshape((num_segments, -1))
        return candidates_rep, candidates

    def _only_tef(self, video_id, moment_loc):
        "Return the feature collection with only any of the temporal features"
        video_duration = self._video_duration(video_id)
        feature_collection = {}
        for i, key in enumerate(self.features):
            feature_collection[key] = self.tef_interface(
                moment_loc, video_duration, clip_length=self.clip_length)
        return feature_collection


class UntrimmedSMCN(UntrimmedBasedMCNStyle):
    """Data feeder for SMCN

    Attributes
        padding (bool): if True the representation is padded with zeros.
    """

    def __init__(self, *args, max_clips=None, padding=True, w_size=None,
                 clip_mask=False, **kwargs):
        super(UntrimmedSMCN, self).__init__(*args, **kwargs)
        self.padding = padding
        self.clip_mask = clip_mask
        if not self.eval:
            max_clips = self.max_number_of_clips()
        self.visual_interface = VisualRepresentationSMCN(
            context=self.context, max_clips=max_clips, w_size=w_size)
        self._set_feat_dim()

    @property
    def decomposable(self):
        decomposable_time_features = (
            self.loc == TemporalFeatures.NONE or
            self.loc == TemporalFeatures.TEMPORALLY_AWARE
        )
        if decomposable_time_features:
            return True
        return False

    def video_clip_representation(self, idx):
        "Return clip-based features of a given video as dict of numpy array"
        video_id = self.videos[idx]
        video_duration = self._video_duration(video_id)
        num_clips = self._video_num_clips(video_id)
        feature_collection = {}
        for key, feat_db in self.features.items():
            feature_video = feat_db[video_id]
            moment_feat_k, _ = self.visual_interface(
                feature_video, [0, video_duration], self.clip_length,
                num_clips)
            feature_collection[key] = moment_feat_k.astype(
                np.float32, copy=False)
        return feature_collection

    def _compute_visual_feature(self, video_id, moment_loc):
        """Return visual features plus TEF for a given segment in the video

        Note:
            This implementation deals with non-decomposable features such
            as TEF. In practice, if you can decompose your model/features
            it's more efficient to re-write the final pooling.
        """
        if self.no_visual:
            # return self._only_tef(video_id, moment_loc)
            raise NotImplementedError('only-TEF is temporally disabled')

        feature_collection = {}
        video_duration = self._video_duration(video_id)
        num_clips = self._video_num_clips(video_id)
        clip_length = self.clip_length
        for key, feat_db in self.features.items():
            feature_video = feat_db[video_id]
            moment_feat_k, mask = self.visual_interface(
                feature_video, moment_loc, clip_length=clip_length,
                num_clips=num_clips)
            if self.tef_interface:
                T, N = mask.sum().astype(np.int), len(moment_feat_k)
                tef_feature = np.zeros((N, 2), dtype=self.tef_interface.dtype)
                tef_feature[:T, :] = self.tef_interface(
                    moment_loc, video_duration, clip_length=clip_length)
                moment_feat_k = np.concatenate(
                    [moment_feat_k, tef_feature], axis=1)

            feature_collection[key] = moment_feat_k.astype(
                np.float32, copy=False)
        # whatever masks is fine given that we don't consider time responsive
        # features yet?
        dtype = np.float32 if self.padding else np.int64
        moment_mask = mask.astype(dtype, copy=False)
        feature_collection['mask'] = moment_mask
        if self.clip_mask:
            clip_mask = np.zeros_like(moment_mask)
            T = int(moment_mask.sum())
            clip_mask[random.randint(0, T - 1)] = 1
            feature_collection['mask_c'] = clip_mask
        return feature_collection

    def _compute_visual_feature_eval(self, video_id):
        "Return visual features plus TEF for all candidate segments in video"
        # We care about corpus video moment retrieval thus our
        # `proposals_interface` does not receive language queries.
        metadata = self.metadata_per_video[video_id]
        candidates = self.proposals_interface(
            video_id, metadata=metadata, feature_collection=self.features)
        candidates_rep = dict_of_lists(
            [self._compute_visual_feature(video_id, t) for t in candidates]
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

    def _only_tef(self, video_id, moment_loc):
        "Return the feature collection with only any of the temporal features"
        video_duration = self._video_duration(video_id)
        clip_length = self.clip_length
        num_clips = self._video_num_clips(video_id)
        if not self.eval:
            num_clips = self.max_number_of_clips()

        # Padding and info about extend of the moment on it
        padded_data = np.zeros((num_clips, 2), dtype=np.float32)
        mask = np.zeros(num_clips, dtype=np.float32)
        im_start = int(moment_loc[0] // clip_length)
        im_end = int((moment_loc[1] - 1e-6) // clip_length)
        T = im_end - im_start + 1

        # Actual features and mask
        padded_data[:T, :] = self.tef_interface(
            moment_loc, video_duration, clip_length=clip_length)
        mask[:T] = 1

        # Packing
        feature_collection = {}
        for i, key in enumerate(self.features):
            feature_collection[key] = padded_data
        feature_collection['mask'] = mask
        return feature_collection


class UntrimmedCALChamfer(UntrimmedBasedMCNStyle):
    """Data feeder for ModelB, ModelD, ModelE, ModelF

    Attributes
        padding (bool): if True the representation is padded with zeros.

    TODO:
        - Disable padding, and decomposable.
        - Consider refactoring using a base class for clip based models to
          reduce code duplication. The `_compute_visual_feature` and
          `_compute_visual_feature_eval` look very similar.
    """

    def __init__(self, *args, max_clips=None, padding=True, w_size=None,
                 **kwargs):
        super(UntrimmedCALChamfer, self).__init__(*args, **kwargs)
        self.padding = padding
        if not self.eval:
            max_clips = self.max_number_of_clips()
        self.visual_interface = VisualRepresentationCALChamfer(
            context=self.context, max_clips=max_clips, w_size=w_size)
        self._set_feat_dim()

    @property
    def decomposable(self):
        decomposable_time_features = (
            self.loc == TemporalFeatures.NONE or
            self.loc == TemporalFeatures.TEMPORALLY_AWARE
        )
        if decomposable_time_features:
            return True
        return False

    def _compute_visual_feature(self, video_id, moment_loc):
        """Return visual features plus TEF for a given segment in the video

        Note:
            This implementation deals with non-decomposable features such
            as TEF. In practice, if you can decompose your model/features
            it's more efficient to re-write the final pooling.
        """
        if self.no_visual:
            # return self._only_tef(video_id, moment_loc)
            raise NotImplementedError('only-TEF is temporally disabled')

        feature_collection = {}
        video_duration = self._video_duration(video_id)
        num_clips = self._video_num_clips(video_id)
        clip_length = self.clip_length
        for key, feat_db in self.features.items():
            feature_video = feat_db[video_id]
            moment_feat_k, mask, im_start = self.visual_interface(
                feature_video, moment_loc, clip_length=clip_length,
                num_clips=num_clips)
            if self.tef_interface:
                T, N = mask.sum().astype(np.int), len(moment_feat_k)
                # tef_feature = np.zeros((N, 2), dtype=self.tef_interface.dtype)
                # tef_feature[:T, :] = self.tef_interface(
                #     moment_loc, video_duration, clip_length=clip_length)
                tef_feature = np.zeros((N, 2), dtype=self.tef_interface.dtype)
                tef_feature[:T, :] = self._get_tef_feature(moment_loc, video_id)
                moment_feat_k = np.concatenate(
                    [moment_feat_k, tef_feature], axis=1)

            feature_collection[key] = moment_feat_k.astype(
                np.float32, copy=False)
        # whatever masks is fine given that we don't consider time responsive
        # features yet?
        dtype = np.float32 if self.padding else np.int64
        moment_mask = mask.astype(dtype, copy=False)
        feature_collection['mask'] = moment_mask
        # DEPRECATED since training is disabled
        # feature_collection['im_start'] = im_start
        return feature_collection

    def _compute_visual_feature_eval(self, video_id):
        "Return visual features plus TEF for all candidate segments in video"
        # We care about corpus video moment retrieval thus our
        # `proposals_interface` does not receive language queries.
        metadata = self.metadata_per_video[video_id]
        candidates = self.proposals_interface(
            video_id, metadata=metadata, feature_collection=self.features)
        candidates_rep = dict_of_lists(
            [self._compute_visual_feature(video_id, t) for t in candidates]
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

    def _only_tef(self, video_id, moment_loc):
        "Return the feature collection with only any of the temporal features"
        video_duration = self._video_duration(video_id)
        clip_length = self.clip_length
        num_clips = self._video_num_clips(video_id)
        if not self.eval:
            num_clips = self.max_number_of_clips()

        # Padding and info about extend of the moment on it
        padded_data = np.zeros((num_clips, 2), dtype=np.float32)
        mask = np.zeros(num_clips, dtype=np.float32)
        im_start = int(moment_loc[0] // clip_length)
        im_end = int((moment_loc[1] - 1e-6) // clip_length)
        T = im_end - im_start + 1

        # Actual features and mask
        padded_data[:T, :] = self.tef_interface(
            moment_loc, video_duration, clip_length=clip_length)
        mask[:T] = 1

        # Packing
        feature_collection = {}
        for i, key in enumerate(self.features):
            feature_collection[key] = padded_data
        feature_collection['mask'] = mask
        feature_collection['im_start'] = im_start
        return feature_collection


class TemporalEndpointFeature():
    """Encode the span of moment in terms of the relative start and end points

    Given that we deal with untrimmed video, we divide by the video duration.
    This class returns the TEF described by MCN paper @ ICCV-2017.

    TODO: force input to be numpy darray
    """

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, start_end, duration, **kwargs):
        return np.array(start_end, dtype=self.dtype) / duration


class TemporalStartpointFeature():
    """Similar to TEF but only take sinto account start point

    TODO:
        force input to be numpy darray
    """

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, start_end, duration, **kwargs):
        return np.array(start_end[0], dtype=self.dtype) / duration


class TemporallyAwareFeature():
    "Encodes the temporal range of a moment in the video"

    def __init__(self, dtype=np.float32, eps=1e-6):
        self.dtype = dtype
        self.eps = eps

    def __call__(self, start_end, duration, clip_length, **kwargs):
        im_start = int(start_end[0] // clip_length)
        im_end = int((start_end[1] - self.eps) // clip_length)
        T = im_end - im_start + 1
        feat = np.empty((T, 2), dtype=self.dtype)
        feat[:, 0] = np.arange(im_start * clip_length,
                               im_end * clip_length + self.eps,
                               clip_length)
        feat[:, 1] = feat[:, 0] + clip_length
        return feat / duration


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

    def __call__(self, features, moment_loc, clip_length, num_clips=None):
        f_dim = features.shape[1]
        data = np.empty(f_dim * self.size_factor, dtype=self.dtype)
        # From time to units of time
        # we substract a small amount of t_end to ensure that it's close to
        # the unit of time in case t_end == clip_length
        # The end index is inclusive, check `function:normalization1d`.
        im_start = int(moment_loc[0] // clip_length)
        im_end = int((moment_loc[1] - self.eps) // clip_length)
        data[0:f_dim] = normalization1d(im_start, im_end, features)
        if self.context:
            ic_start, ic_end = 0, len(features) - 1
            if num_clips is not None:
                ic_end = num_clips - 1
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
                 max_clips=None, padding=True, w_size=None):
        self.context = context
        self.size_factor = context + 1
        self.dtype = dtype
        self.eps = eps
        self.max_clips = max_clips
        self.padding = padding
        self.context_fn = global_context
        self._w_half = None
        self._box = None
        if w_size is not None:
            self.context_fn = self._local_context
            self._w_half = w_size // 2
            self._box = np.ones((w_size, 1), dtype=dtype) / w_size

    def __call__(self, features, moment_loc, clip_length, num_clips=None):
        n_feat, f_dim = features.shape
        if self.max_clips is not None:
            n_feat = self.max_clips
        # From time to units of time
        # we substract a small amount of t_end to ensure that it's close to
        # the unit of time in case t_end == clip_length
        # The end index is inclusive, check `function:normalization1d`.
        im_start = int(moment_loc[0] // clip_length)
        im_end = int((moment_loc[1] - self.eps) // clip_length)
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
            if self._w_half is None:
                context_info = self.context_fn(features, num_clips)
            else:
                context_info = self.context_fn(
                    im_start, im_end, features, num_clips)
            padded_data[:T, f_dim:2 * f_dim] = context_info
        if self.padding:
            return padded_data, mask
        return padded_data, np.array([T])

    def _local_context(self, start, end, x_t, num_clips=None):
        "Context around clips"
        if num_clips is None:
            num_clips = x_t.shape[0]
        ind_start = start - self._w_half
        pad_left = -1 * min(ind_start, 0)
        ind_start = max(ind_start, 0)
        ind_end = end + self._w_half + 1
        pad_right = max(ind_end - num_clips, 0)

        x_t = x_t[ind_start:ind_end, :]
        if pad_right > 0 or pad_left > 0:
            x_t = np.pad(x_t, ((pad_left, pad_right), (0, 0)), 'edge')
        y_t = convolve(x_t, self._box, 'valid')
        scaling_factor = np.linalg.norm(y_t, axis=1, keepdims=True) + self.eps
        return y_t / scaling_factor

    def _local_feature(self, start, end, x):
        "Return normalized representation of each clip/chunk"
        y = x[start:end + 1, :]
        scaling_factor = np.linalg.norm(y, axis=1, keepdims=True) + self.eps
        return y / scaling_factor


class VisualRepresentationCALChamfer():
    """Compute visual features for ModelB model

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
                 max_clips=None, padding=True, w_size=None):
        self.context = context
        self.size_factor = context + 1
        self.dtype = dtype
        self.eps = eps
        self.max_clips = max_clips
        self.padding = padding
        self.context_fn = global_context
        self._w_half = None
        self._box = None
        if w_size is not None:
            self.context_fn = self._local_context
            self._w_half = w_size // 2
            self._box = np.ones((w_size, 1), dtype=dtype) / w_size

    def __call__(self, features, moment_loc, clip_length, num_clips=None):
        n_feat, f_dim = features.shape
        if self.max_clips is not None:
            n_feat = self.max_clips
        # From time to units of time
        # we substract a small amount of t_end to ensure that it's close to
        # the unit of time in case t_end == clip_length
        # The end index is inclusive, check `function:normalization1d`.
        im_start = int(moment_loc[0] // clip_length)
        im_end = int((moment_loc[1] - self.eps) // clip_length)
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
            if self._w_half is None:
                context_info = self.context_fn(features, num_clips)
            else:
                context_info = self.context_fn(
                    im_start, im_end, features, num_clips)
            padded_data[:T, f_dim:2 * f_dim] = context_info
        if self.padding:
            return padded_data, mask, im_start
        return padded_data, np.array([T])

    def _local_context(self, start, end, x_t, num_clips=None):
        "Context around clips"
        if num_clips is None:
            num_clips = x_t.shape[0]
        ind_start = start - self._w_half
        pad_left = -1 * min(ind_start, 0)
        ind_start = max(ind_start, 0)
        ind_end = end + self._w_half + 1
        pad_right = max(ind_end - num_clips, 0)

        x_t = x_t[ind_start:ind_end, :]
        if pad_right > 0 or pad_left > 0:
            x_t = np.pad(x_t, ((pad_left, pad_right), (0, 0)), 'edge')
        y_t = convolve(x_t, self._box, 'valid')
        scaling_factor = np.linalg.norm(y_t, axis=1, keepdims=True) + self.eps
        return y_t / scaling_factor

    def _local_feature(self, start, end, x):
        "Return normalized representation of each clip/chunk"
        y = x[start:end + 1, :]
        scaling_factor = np.linalg.norm(y, axis=1, keepdims=True) + self.eps
        return y / scaling_factor


def global_context(x_t, num_clips=None):
    """Context over the entire video

    Args:
        x_t (numpy array): of shape [N, D]. D:= feature dimension
        num_clips (int): valid features. If `None` then `num_clips = N`.
    Returns:
        numpy array of shape [D].
    """
    ic_start, ic_end = 0, len(x_t) - 1
    if num_clips is not None:
        ic_end = num_clips - 1
    return normalization1d(ic_start, ic_end, x_t)


def normalization1d(start, end, features):
    "1D mean-pooling + normalization for visual features"
    base_feature = np.mean(features[start:end + 1, :], axis=0)
    scaling_factor = np.linalg.norm(base_feature) + 0.00001
    return base_feature / scaling_factor


def sentences_to_words(sentences):
    words = []
    for s in sentences:
        words.extend(word_tokenize(str(s.lower())))
    return words


def word_tokenize(s):
    sent = s.lower()
    sent = re.sub('[^A-Za-z0-9\s]+', ' ', sent)
    return sent.split()



if __name__ == '__main__':
    import os
    import time
    from proposals import SlidingWindowMSRSS, DidemoICCV17SS
    # Kinda Unit-test
    print('UntrimmedMCN\n\t* Train')
    t_start = time.time()
    json_data = 'data/processed/didemo/train-03.json'
    h5_file = 'data/processed/didemo/resnet152_rgb_max_cl-5.h5'
    dummy_h5_neg_sampling = 'data/interim/debug/dummy_1st_retrieval.h5'
    # TODO: update Charades-STA pre-processed data
    # json_data = 'data/processed/charades-sta/train-01.json'
    # h5_file = 'data/processed/charades-sta/rgb_resnet152_max_cl-3.h5'
    cues = {'rgb': {'file': h5_file}}
    dataset = UntrimmedMCN(json_data, cues, debug=True)
    num_queries = len(dataset)
    num_videos = len(dataset.videos)
    if 'didemo' in json_data:
        dataset.proposals_interface = DidemoICCV17SS()
    else:
        dataset.proposals_interface = SlidingWindowMSRSS()
    num_proposals = sum([len(dataset.video_item(i)[1])
                         for i in range(num_videos)])
    with h5py.File(dummy_h5_neg_sampling, 'w') as fid:
        video_ind = np.arange(num_videos)
        data = [np.random.permutation(video_ind) for i in range(num_queries)]
        data = np.repeat(data, (num_proposals // num_queries) + 1, 1)
        data = data[:, :num_proposals]
        fid.create_dataset('vid_indices', data=data)
    print(f'\tPreproc due to h5-nis: {time.time() - t_start}')
    t_start = time.time()
    dataset = UntrimmedMCN(
        json_data, cues, proposals_interface=dataset.proposals_interface,
        h5_nis=dummy_h5_neg_sampling, debug=True)
    ind = int(random.random() * len(dataset))
    print(f'\tTime loading dataset: {time.time() - t_start}')
    print('\tNumber of moments: ', num_queries)
    print('\tNumber of videos: ', num_videos)
    print('\tNumber of proposals: ', num_proposals)
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
    os.remove(dummy_h5_neg_sampling)

    # Kinda Unit-test for UntrimmedSMCN
    # (json_data and cues come from previous unit-test
    print('UntrimmedSMCN\n\ttrain')
    t_start = time.time()
    dataset = UntrimmedSMCN(json_data, cues, debug=True)
    num_queries = len(dataset)
    num_videos = len(dataset.videos)
    if 'didemo' in json_data:
        dataset.proposals_interface = DidemoICCV17SS()
    else:
        dataset.proposals_interface = SlidingWindowMSRSS()
    num_proposals = sum([len(dataset.video_item(i)[1])
                         for i in range(num_videos)])
    with h5py.File(dummy_h5_neg_sampling, 'w') as fid:
        video_ind = np.arange(num_videos)
        data = [np.random.permutation(video_ind) for i in range(num_queries)]
        data = np.repeat(data, (num_proposals // num_queries) + 1, 1)
        data = data[:, :num_proposals]
        fid.create_dataset('vid_indices', data=data)
    print(f'\tPreproc due to h5-nis: {time.time() - t_start}')
    t_start = time.time()
    dataset = UntrimmedSMCN(
        json_data, cues, proposals_interface=dataset.proposals_interface,
        h5_nis=dummy_h5_neg_sampling, debug=True)
    ind = int(random.random() * len(dataset))
    print(f'\tTime loading dataset: ', time.time() - t_start)
    print('\tNumber of moments: ', num_queries)
    print('\tNumber of videos: ', num_videos)
    print('\tNumber of proposals: ', num_proposals)
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
    os.remove(dummy_h5_neg_sampling)