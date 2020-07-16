import hashlib
import itertools
import json

import h5py
import numpy as np
import torch

from np_segments_ops import non_maxima_suppresion
from utils import unique2d_perserve_order


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

    def __init__(self, dataset, dict_of_models, alpha=None, device='cpu'):
        self.dataset = dataset
        self.models = dict_of_models
        self.device = device
        if alpha is None:
            self.alpha = {key: 1 /len(dict_of_models)
                          for key in dict_of_models}
        else:
            if alpha <0 or alpha > 1:
                raise('Alpha must be a value between 0 and 1')
            else:
                self.alpha = {key: round(1*i + (-1)**i*alpha,2)  
                          for i,key in enumerate(dict_of_models)}
        condition = (dataset.cues.keys() == dict_of_models.keys() or 
                    {'-'.join(dataset.cues.keys()):None}.keys() == dict_of_models.keys())
        assert condition == True
        assert dict_of_models.keys() == self.alpha.keys()
        self.dataset_cues = dataset.cues.keys()
        self.keys = dict_of_models.keys()
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
        # TODO (release): allow to tokenize description
        # assert isinstance(description, list)
        lang_feature_, len_query_ = self.dataset._compute_language_feature(description)
        # torchify
        lang_feature = torch.from_numpy(lang_feature_[0])
        lang_feature.unsqueeze_(0)
        len_query = torch.tensor([len_query_[0]])
        return lang_feature, len_query

    def postprocess(self, distance):
        "apply postprocessing functions"
        raise NotImplementedError('WIP')

    def query(self, description):
        "Return videos and moments aligned with a text description"
        raise NotImplementedError('Subclass and implement your indexing')


class ClipRetrieval(CorpusVideoMomentRetrievalBase):
    "Retrieve clips that aligns well with a particular lang query"

    def __init__(self, *args, **kwargs):
        super(ClipRetrieval, self).__init__(*args, **kwargs)
        self.clip2proposal_ind = None
        self.clip_tables = None
        self.num_clips = None
        if not self.dataset.decomposable:
            raise ValueError('Dataset represenation does not admit '
                             'clip-based retrieval')
    def indexing(self):
        "Create database of moments in videos"
        torch.set_grad_enabled(False)
        num_entries_per_video = []
        all_proposals = []
        clip2proposal_ind = {}
        codes = {key: [] for key in self.models}
        sample_key = list(self.models.keys())[0]
        proposal_runner, clip_runner = 0, 0
        clip_length = self.dataset.clip_length
        # TODO (tier-2;design): define method in dataset to do this?
        # batchify the fwd-pass
        for video_ind, video_id in enumerate(self.dataset.videos):
            proposals_ = self.dataset.video_proposals(video_ind)
            representation_dict = self.dataset.video_clip_representation(
                video_ind)
            num_clips_i = representation_dict[sample_key].shape[0]
            num_entries_per_video.append(num_clips_i)

            # Map clips to proposals
            for clip_ind in range(num_clips_i):
                clip_center_s = clip_length * (clip_ind + 0.5)
                clip_ge_start = clip_center_s >= proposals_[:, 0]
                clip_le_end = clip_center_s <= proposals_[:, 1]
                clip_in_proposal = clip_ge_start & clip_le_end
                proposal_inds = (
                    clip_in_proposal.nonzero()[0] + proposal_runner).tolist()

                clip_id = clip_ind + clip_runner
                clip2proposal_ind[clip_id] = proposal_inds
            proposal_runner += len(proposals_)
            clip_runner += num_clips_i

            # torchify
            for key, value in representation_dict.items():
                representation_dict[key] = torch.from_numpy(value)
            proposals = torch.from_numpy(proposals_)

            # Append items to database
            all_proposals.append(proposals)
            for key in self.dataset.cues:
                clip_rep_k = representation_dict[key].unsqueeze_(0)
                # get codes of the clips -> C_i x D matrix
                # Given a video i with C_i clips, we encode all the clips
                # through the visual encoder
                code_k = self.models[key].visual_encoder[key](clip_rep_k).squeeze_()
                codes[key].append(code_k)
        # Form the C x D matrix
        # M := number of videos, C = \sum_{i=1}^M C_i
        # We have as many tables as visual cues
        self.moments_tables = {key: torch.cat(value)
                               for key, value in codes.items()}
        # TODO (tier-2; design): organize this better
        self.num_videos = self.dataset.num_videos
        self.entries_per_video = torch.tensor(num_entries_per_video)
        self.num_clips = self.entries_per_video.sum().item()
        self.proposals = torch.cat(all_proposals)
        self.num_moments = int(self.proposals.shape[0])
        video_indices = np.repeat(np.arange(0, self.dataset.num_videos),
                                  num_entries_per_video)
        self.video_indices = torch.from_numpy(video_indices)
        self.clip2proposal_ind = clip2proposal_ind

    def query(self, description, batch_size=None):
        "Search clips based on a text description given as list of words"
        descending_k = False
        torch.set_grad_enabled(False)
        lang_feature, len_query = self.preprocess_description(description)
        score_list, descending_list = [], []
        for key, model_k in self.models.items():
            lang_code = model_k.encode_query([lang_feature], [len_query])
            scores_k = ((lang_code[key][0] - self.moments_tables[key])**2).sum(dim=-1)
            score_list.append(scores_k * self.alpha[key])
            descending_list.append(descending_k)
        scores = sum(score_list)
        # assert np.unique(descending_list).shape[0] == 1
        scores, sorted_clip_indices = scores.sort(descending=descending_k)
        return self.video_indices[sorted_clip_indices], sorted_clip_indices


class DummyMomentRetrievalFromProposalsTable(CorpusVideoMomentRetrievalBase):
    """Setup moments table
    """

    def __init__(self, *args, **kwargs):
        super(DummyMomentRetrievalFromProposalsTable, self).__init__(
            *args, **kwargs)

    def indexing(self):
        "Setup index for moments and videos"
        num_entries_per_video = []
        all_proposals = []
        for video_id in self.dataset.videos:
            _, proposals_ = self.dataset._compute_visual_feature_eval(
                video_id)
            num_entries_per_video.append(len(proposals_))

            # torchify
            proposals = torch.from_numpy(proposals_)

            # Append items to database
            all_proposals.append(proposals)
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
        raise ValueError('Not intended to be used to retrieve moments')


class LoopOverKBase():
    "TODO: description"

    def __init__(self, dataset, model, h5_1ststage, topk=100,
                 nms_threshold=1.0):
        self.dataset = dataset
        self.model = model
        self.h5_file = h5_1ststage
        self.topk = topk
        self.nms_threshold = nms_threshold
        self.proposals = None  # torch 2D-tensor
        self.query2videos_ind = None  # numpy  2D-array
        self.query2videos_ind_per_proposal = None  # torch 2D-tensor
        self.query2proposals_ind = None  # torch 2D-tensor
        self._setup()

    @property
    def num_moments(self):
        return self.proposals.shape[0]

    def preprocess_description(self, description):
        "Return tensors representing description as 1) vectors and 2) length"
        # TODO (refactor): duplicate snippet from
        # CorpusVideoMomentRetrievalBase. Factor it out as function or apply
        # inheritance.

        # TODO (release): allow to tokenize description
        # assert isinstance(description, list)
        lang_feature_, len_query_ = self.dataset._compute_language_feature(description)
        # torchify
        lang_feature = torch.from_numpy(lang_feature_[0])
        lang_feature.unsqueeze_(0)
        len_query = torch.tensor(len_query_)
        return lang_feature, len_query

    def query(self, description, description_ind):
        raise NotImplementedError('Subclass and implement')

    def _setup(self):
        "Misc stuff like load results from 1st retrieval stage"
        with h5py.File(self.h5_file, 'r') as fid:
            query2videos_ind = fid['vid_indices'][:]
            # Force us to examine a way to deal with approximate retrieval
            # approaches
            assert query2videos_ind.shape[1] >= self.dataset.num_videos
            assert (query2videos_ind >= 0).all()
            # Trigger post-processing in case we are dealing with retrieval
            # results from a moment-based approach
            if query2videos_ind.shape[1] > self.dataset.num_videos:
                self.query2videos_ind_per_proposal = torch.from_numpy(
                    query2videos_ind)
                query2videos_ind = unique2d_perserve_order(query2videos_ind)
            self.query2videos_ind = query2videos_ind

            # Note: self.proposals may be redudant and we could create a table
            # to save storage in practice
            if 'proposals' in fid:
                self.proposals = torch.from_numpy(fid['proposals'][:])
            else:
                proposals = []
                for video_ind in range(self.dataset.num_videos):
                    _, proposals_i = self.dataset.video_item(video_ind)
                    proposals.append(proposals_i)
                self.proposals = torch.from_numpy(
                    np.concatenate(proposals, axis=0))

            if 'proposals_ind' in fid:
                self.query2proposals_ind = fid['proposals_ind'][:]


class LoopOverKVideos(LoopOverKBase):
    """Rank moments contained on K-videos

    TODO: description
    """

    def __init__(self, *args, repeat_lang=False, **kwargs):
        self.repeat_lang = repeat_lang
        super(LoopOverKVideos, self).__init__(*args, **kwargs)

    def query(self, description, description_ind, batch_size=None):
        "Return videos and moments aligned with a text description"
        # TODO (tier-2): remove 2nd-stage results from 1st-stage to make them
        # exhaustive
        torch.set_grad_enabled(False)
        lang_feature, len_query = self.preprocess_description(description)

        video_indices_1ststage = self.query2videos_ind[description_ind, :]
        video_indices, proposals, scores = [], [], []
        for i in range(self.topk):
            video_ind = int(video_indices_1ststage[i])

            candidates_i_feat, proposals_i = self.dataset.video_item(video_ind)
            # torchify
            candidates_i_feat = {k: torch.from_numpy(v)
                                 for k, v in candidates_i_feat.items()}
            proposals_i = torch.from_numpy(proposals_i)

            if self.repeat_lang:
                lang_feature_ = lang_feature.repeat(len(proposals_i), 1, 1)
                len_query_ = len_query.repeat(len(proposals_i))
            else:
                lang_feature_, len_query_ = lang_feature, len_query

            scores_i, descending_i = self.model.predict(
                lang_feature_, len_query_, candidates_i_feat)

            # TODO: add post-processing such as NMS
            if self.nms_threshold < 1:
                idx = non_maxima_suppresion(
                        proposals_i.numpy(), -scores_i.numpy(),
                        self.nms_threshold)
                proposals_i = proposals_i[idx, :]
                scores_i = scores_i[idx]

            scores.append(scores_i)
            proposals.append(proposals_i)
            video_indices.append(
                video_ind * torch.ones(len(proposals_i), dtype=torch.int32))

        scores = torch.cat(scores)
        proposals = torch.cat(proposals, dim=0)
        video_indices = torch.cat(video_indices)
        scores, ind = scores.sort(descending=descending_i)
        return video_indices[ind], proposals[ind, :]


class LoopOverKMoments(LoopOverKBase):
    """Re-rank topk moments

    For text-to-video retrieval algorithms, we evaluate enough videos such
    that the number of retrieved moments is bounded.

    TODO: description
    """

    def __init__(self, *args, **kwargs):
        self.moment_based_reranking = False
        super(LoopOverKMoments, self).__init__(*args, **kwargs)

    def query(self, description, description_ind, batch_size=None):
        "Return videos and moments aligned with a text description"
        # TODO (tier-2): remove 2nd-stage results from 1st-stage to make them
        # exhaustive
        torch.set_grad_enabled(False)
        lang_feature, len_query = self.preprocess_description(description)

        video_ind_1ststage = self.query2videos_ind[description_ind, :]
        # Sorry for this dirty trick
        video_indices, proposals, scores = [], [], []
        if self.moment_based_reranking:
            proposals_ind = self.query2proposals_ind[
                description_ind, :self.topk]
            video_indices = self.query2videos_ind_per_proposal[
                description_ind, :self.topk]
            proposals = self.proposals[proposals_ind, :]

        proposals_counter = 0
        for i in range(self.topk):
            # branch according to 1st-stage
            if self.moment_based_reranking:
                video_id = self.dataset.videos[video_indices[i]]
                # There is only a single candidate in this case
                candidates_i_feat = self.dataset._compute_visual_feature(
                    video_id, proposals[i, :].numpy())
                for k, v in candidates_i_feat.items():
                    if isinstance(v, np.ndarray):
                        candidates_i_feat[k] = v[None, :]
                proposals_i = proposals[i, :].unsqueeze_(dim=0)
                proposals_counter += 1
            else:
                video_ind = int(video_ind_1ststage[i])
                candidates_i_feat, proposals_i = self.dataset.video_item(
                    video_ind)
                video_ind_i = video_ind * torch.ones(
                    len(proposals_i), dtype=torch.int32)
                proposals_counter += len(proposals_i)

            # torchify
            candidates_i_feat = {k: torch.from_numpy(v)
                                 for k, v in candidates_i_feat.items()}
            if isinstance(proposals_i, np.ndarray):
                proposals_i = torch.from_numpy(proposals_i)

            scores_i, descending_i = self.model.predict(
                lang_feature, len_query, candidates_i_feat)

            # TODO: add post-processing such as NMS
            if self.nms_threshold < 1:
                idx = non_maxima_suppresion(
                        proposals_i.numpy(), -scores_i.numpy(),
                        self.nms_threshold)
                proposals_i = proposals_i[idx, :]
                scores_i = scores_i[idx]

            scores.append(scores_i)
            if isinstance(proposals, list):
                proposals.append(proposals_i)
                video_indices.append(video_ind_i)

            if proposals_counter >= self.topk:
                break

        # Part of the dirty trick
        if isinstance(proposals, list):
            proposals = torch.cat(proposals, dim=0)
            video_indices = torch.cat(video_indices)
        scores = torch.cat(scores)
        scores, ind = scores.sort(descending=descending_i)
        return video_indices[ind], proposals[ind, :]

    def _setup(self):
        super(LoopOverKMoments, self)._setup()
        if self.query2videos_ind_per_proposal is not None:
            self.moment_based_reranking = True


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
        torch.set_grad_enabled(False)
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
                # segment_rep_k = representation_dict[key]
                # get codes of the proposals -> S_i x D matrix
                # S_i := num proposals in ith-video

                codes[key].append(
                    self.models[key].visual_encoder[key](
                                representation_dict[key]))
                # codes[key].append(
                #     self.models[key].visual_encoder(segment_rep_k))
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

    def query(self, description, return_indices=False, batch_size=None):
        "Search moments based on a text description given as list of words"
        torch.set_grad_enabled(False)
        lang_feature, len_query = self.preprocess_description(description)
        score_list, descending_list = [], []
        for key, model_k in self.models.items():
            lang_code = model_k.encode_query([lang_feature], [len_query])
            scores_k, descending_k = model_k.search(
                lang_code[key][0], self.moments_tables[key])
            score_list.append(scores_k * self.alpha[key])
            descending_list.append(descending_k)
        scores = sum(score_list)
        # assert np.unique(descending_list).shape[0] == 1
        scores, ind = scores.sort(descending=descending_k)
        # TODO (tier-1): enable bell and whistles
        if return_indices:
            return self.video_indices[ind], self.proposals[ind, :], ind, scores
        return self.video_indices[ind], self.proposals[ind, :], scores


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
        torch.set_grad_enabled(False)
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
                codes[key].append(self.models[key].visual_encoder[key](segment_rep_k))
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

    def query(self, description, return_indices=False, batch_size=None):
        "Search moments based on a text description given as list of words"
        torch.set_grad_enabled(False)
        lang_feature, len_query = self.preprocess_description(description)
        score_list, descending_list = [], []
        for key, model_k in self.models.items():
            lang_code = model_k.encode_query([lang_feature], [len_query])
            scores_k, descending_k = model_k.search(
                lang_code, self.moments_tables[key], self.clips_per_moment,
                self.clips_per_moment_list)
            score_list.append(scores_k * self.alpha[key])
            descending_list.append(descending_k)
        scores = sum(score_list)
        # assert np.unique(descending_list).shape[0] == 1
        scores, ind = scores.sort(descending=descending_k)
        # TODO (tier-1): enable bell and whistles
        if return_indices:
            return self.video_indices[ind], self.proposals[ind, :], ind, scores
        return self.video_indices[ind], self.proposals[ind, :], scores


class MomentRetrievalFromClipBasedProposalsTableNew(
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
        super(MomentRetrievalFromClipBasedProposalsTableNew, self).__init__(
            *args, **kwargs)
        self.clips_per_moment = None
        self.clips_per_moment_list = None
        self.masks = {}

    def indexing(self):
        "Create database of moments in videos"
        torch.set_grad_enabled(False)
        num_entries_per_video = []
        # clips_per_moment = []
        # obj_per_moment = []
        list_of_keys = []
        stream_keys = [k for k in self.models] 
        if len(stream_keys) == 1:
            for k in stream_keys[0].split('-'):
                mask_key = '-'.join(['mask',k])
                list_of_keys.append(k)
                list_of_keys.append(mask_key)
        else:
            for k in stream_keys:
                mask_key = '-'.join(['mask',k])
                list_of_keys.append(k)
                list_of_keys.append(mask_key)

        codes = {key: [] for key in list_of_keys}

        if len(self.models) == 1:
            self.model_key = stream_keys[0]
            if len(stream_keys) > 1 :
                self.model_key = '-'.join(stream_keys)

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
            if len(self.models) == 1:
                for k in representation_dict.keys():
                    if 'mask' not in k:
                        codes[k].append(self.models[self.model_key].visual_encoder[k](
                                        representation_dict[k].to(self.device)).cpu())        
                    else:
                        codes[k].append(representation_dict[k])
            else:
                for k in representation_dict.keys():
                    if 'mask' not in k:
                        codes[k].append(self.models[k].visual_encoder[k](
                                        representation_dict[k].to(self.device)).cpu())        
                    else:
                        codes[k].append(representation_dict[k])

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

    def query(self, description, return_indices=False, batch_size=10000):
        "Search moments based on a text description given as list of words"
        torch.set_grad_enabled(False)
        lang_feature, len_query = self.preprocess_description(description)
        score_list, descending_list = [], []
        if  len(self.models) == 1:
            for k, model_k in self.models.items():
                # [0] was added after introducing regularizer for the language, we should code a function to wrap the
                # operation instead of accessing the parts of the network we want.
                lang_code = model_k.encode_query([lang_feature.to(self.device)], [len_query])[0]
                for key in self.moments_tables.keys():
                    if 'mask' not in key:
                        mask_key = '-'.join(['mask',key])
                        scores_k, descending_k = model_k.search(
                                        lang_code[key], len_query, 
                                        self.moments_tables[key], 
                                        self.moments_tables[mask_key],
                                        batch_size)

                        score_list.append(scores_k)
                        descending_list.append(descending_k)
        else:
            for k, model_k in self.models.items():
                lang_code = model_k.encode_query([lang_feature], [len_query])
                mask_key = '-'.join(['mask',k])
                scores_k, descending_k = model_k.search(
                                lang_code[k], len_query, 
                                self.moments_tables[k], 
                                self.moments_tables[mask_key])
                score_list.append(scores_k * self.alpha[k])
                descending_list.append(descending_k)
        
        scores = sum(score_list)
        # assert np.unique(descending_list).shape[0] == 1
        scores, ind = scores.sort(descending=descending_k)
        # TODO (tier-1): enable bell and whistles
        if return_indices:
            return self.video_indices[ind], self.proposals[ind, :], ind, scores
        return self.video_indices[ind], self.proposals[ind, :], scores


class MomentRetrievalFromClipBasedProposalsTableEarlyFusion(
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
        super(MomentRetrievalFromClipBasedProposalsTableEarlyFusion, self).__init__(
            *args, **kwargs)
        self.clips_per_moment = None
        self.clips_per_moment_list = None
        self.masks = {}

    def indexing(self):
        "Create database of moments in videos"
        torch.set_grad_enabled(False)
        num_entries_per_video = []
        # clips_per_moment = []
        # obj_per_moment = []
        list_of_keys = []
        stream_keys = [k for k in self.models] 
        if len(stream_keys) == 1:
            for k in stream_keys[0].split('-'):
                mask_key = '-'.join(['mask',k])
                list_of_keys.append(k)
                list_of_keys.append(mask_key)

        else:
            for k in stream_keys:
                mask_key = '-'.join(['mask',k])
                list_of_keys.append(k)
                list_of_keys.append(mask_key)

        codes = {key: [] for key in list_of_keys}

        if len(self.models) == 1:
            self.model_key = stream_keys[0]
            if len(stream_keys) > 1 :
                self.model_key = '-'.join(stream_keys)

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
            if len(self.models) == 1:
                for k in representation_dict.keys():
                    if 'mask' not in k:
                        codes[k].append(self.models[self.model_key].visual_encoder[k](
                                                            representation_dict[k]))        
                    else:
                        codes[k].append(representation_dict[k])
            else:
                for k in representation_dict.keys():
                    if 'mask' not in k:
                        codes[k].append(self.models[k].visual_encoder[k](
                                                            representation_dict[k]))        
                    else:
                        codes[k].append(representation_dict[k])

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

    def query(self, description, return_indices=False, batch_size=None):
        "Search moments based on a text description given as list of words"
        torch.set_grad_enabled(False)
        lang_feature, len_query = self.preprocess_description(description)
        score_list, descending_list = [], []
        for k, model_k in self.models.items():
            lang_code = model_k.encode_query(lang_feature, len_query)
            scores_k, descending_k = model_k.search(lang_code, len_query, self.moments_tables)
            score_list.append(scores_k)
            descending_list.append(descending_k)
            
        scores = sum(score_list)
        # assert np.unique(descending_list).shape[0] == 1
        scores, ind = scores.sort(descending=descending_k)
        # TODO (tier-1): enable bell and whistles
        if return_indices:
            return self.video_indices[ind], self.proposals[ind, :], ind, scores
        return self.video_indices[ind], self.proposals[ind, :], scores


class GreedyMomentRetrievalFromClipBasedProposalsTable(
        CorpusVideoMomentRetrievalBase):
    "TODO: Retrieve Moments using a clip based model"

    def __init__(self, *args, topk=None, **kwargs):
        super(GreedyMomentRetrievalFromClipBasedProposalsTable,
              self).__init__(*args, **kwargs)
        self.clips_per_moment = None
        self.clips_per_moment_list = None
        self.video_clip2proposals = {}
        self.clips_tables = None
        self.clip_indices = None
        self.topk = topk
        assert self.dataset.decomposable

    def indexing(self):
        "Create database of moments in videos"
        torch.set_grad_enabled(False)
        num_entries_per_video, clips_per_moment = [], []
        clips_per_video, clip_indices = [], []
        all_proposals = []
        codes = {key: [] for key in self.models}
        clip_codes = {key: [] for key in self.models}
        sample_key = list(self.models.keys())[0]
        moment_ind_runner = 0
        # TODO (tier-2;design): define method in dataset to do this?
        # batchify the fwd-pass
        for video_index, video_id in enumerate(self.dataset.videos):
            representation_dict, proposals_ = (
                self.dataset._compute_visual_feature_eval(video_id))
            num_entries_per_video.append(len(proposals_))
            num_clips_i = representation_dict['mask']
            if num_clips_i.ndim != 1:
                raise ValueError('Dataset setup incorrectly. Disable padding')
            # TODO(tier-2;refactor): this could be cleaner.
            # We resort in our implementation of non-decomposable SMCN while
            # this could be implemented much efficiently. Given that the
            # features were packed like [Mi0; Mi1;...; MiS_i] where
            # Mij := c_ij x D tensor (lines above), it's meassy to select the
            # features of the unique clips. Thus, we prefer to request them
            # again for a proposal spanning the entire video duration.
            all_video_moment = np.array(
                [0, self.dataset._video_duration(video_id)])
            clips_representation_dict = self.dataset._compute_visual_feature(
                video_id, all_video_moment)

            clips_per_video.append(len(clips_representation_dict[sample_key]))
            clip_indices.append(
                torch.arange(0, clips_per_video[-1], 1, dtype=torch.long))

            # Update mapping from (video_index, clip_index_at_video) to
            # proposal_index_at_corpus
            clip_length = self.dataset.clip_length
            for proposal_ind_v, proposal_i in enumerate(proposals_):
                proposal_index = moment_ind_runner + proposal_ind_v
                c_start = int(proposal_i[0] // clip_length)
                c_end = int((proposal_i[1] - 1e-6) // clip_length)
                for c_index in range(c_start, c_end + 1):
                    video_clip_index = (video_index, c_index)
                    if video_clip_index not in self.video_clip2proposals:
                        self.video_clip2proposals[video_clip_index] = []
                    self.video_clip2proposals[video_clip_index].append(
                        proposal_index)
            moment_ind_runner += num_entries_per_video[-1]

            # torchify
            for key, value in representation_dict.items():
                if key == 'mask': continue
                # get representation of all proposals
                representation_dict[key] = torch.from_numpy(value)
                clips_rep_k = clips_representation_dict[key]
                clips_representation_dict[key] = torch.from_numpy(clips_rep_k)
            proposals = torch.from_numpy(proposals_)
            clips_per_moment.append(torch.from_numpy(num_clips_i))

            # Append items to database
            all_proposals.append(proposals)
            for key in self.dataset.cues:
                segment_rep_k = representation_dict[key]
                # get codes of the proposals -> C_i x D matrix
                # Given The i-th video with S_i number of proposals
                # Each proposal S_i spans c_ij clips of the i-th video.
                # C_i = \sum_{j=1}^{S_i} c_ij := num clips over all S_i
                # proposals in the i-th video
                codes[key].append(
                    self.models[key].visual_encoder(segment_rep_k))

                # similar to codes of proposals but of clips in video
                clips_rep_k = clips_representation_dict[key]
                clip_codes[key].append(
                    self.models[key].visual_encoder(clips_rep_k))
        # Form the C x D matrix
        # M := number of videos, C = \sum_{i=1}^M C_i
        # We have as many tables as visual cues
        self.moments_tables = {key: torch.cat(value)
                               for key, value in codes.items()}
        self.clips_tables = {key: torch.cat(value)
                             for key, value in clip_codes.items()}
        clip_table_entries = sum(clips_per_video)
        for key, value in self.clips_tables.items():
            assert value.shape[0] == clip_table_entries

        # TODO (tier-2; design): organize this better
        self.num_videos = len(num_entries_per_video)
        self.entries_per_video = torch.tensor(num_entries_per_video)
        self.proposals = torch.cat(all_proposals)
        self.num_moments = int(self.proposals.shape[0])
        video_indices = np.repeat(
            np.arange(0, len(self.dataset.videos)), num_entries_per_video)
        self.video_indices = torch.from_numpy(video_indices)
        self.clips_per_moment = torch.cat(clips_per_moment)
        self.clips_per_moment_list = self.clips_per_moment.tolist()
        self.clips_per_moment = self.clips_per_moment.float()
        self.cumsum_clips_per_moment_np = np.cumsum(
            self.clips_per_moment_list)
        self.clips_indices = torch.cat(clip_indices)
        video_indices_clip = np.repeat(
            np.arange(0, len(self.dataset.videos)), clips_per_video)
        self.video_indices_clip = torch.from_numpy(video_indices_clip)

    def query(self, description, return_indices=False, batch_size=None):
        "Search moments based on a text description given as list of words"
        torch.set_grad_enabled(False)
        lang_feature, len_query = self.preprocess_description(description)

        # Search over clips
        clip_score_list = []
        for key, model_k in self.models.items():
            lang_code = model_k.encode_query(lang_feature, len_query)
            clips_score_k = model_k.compare_emdeddings(
                lang_code, self.clips_tables[key])
            clip_score_list.append(clips_score_k * self.alpha[key])
        clips_score = sum(clip_score_list)
        # TODO (tier-2;release): not hard-code False
        _, ind_clips = clips_score.sort(descending=False)

        # TODO (tier-1): enable bell and whistles
        # TODO(tier-2;performance?): allocate tensor of size topk
        greedy_global_moment_indices = [
            self.video_clip2proposals.get(
                (self.video_indices_clip[i].item(),
                 self.clips_indices[i].item())
            )
            for i in ind_clips[:self.topk]
        ]
        greedy_global_moment_indices = sum(greedy_global_moment_indices, [])
        greedy_global_moment_indices = np.unique(greedy_global_moment_indices)

        # Search over moments, only over moments containing clips above
        clips_per_moment = self.clips_per_moment[greedy_global_moment_indices]
        video_indices = self.video_indices[greedy_global_moment_indices]
        proposals = self.proposals[greedy_global_moment_indices, :]
        clips_per_moment_list, indices_for_moments_table = [], []
        for i in greedy_global_moment_indices:
            clips_per_moment_list.append(self.clips_per_moment_list[i])
            # We need this hack because we are using the general
            # non-decomposable indexing of SMCN. This is a realistic test bed
            # to compare accuracy.
            ind_start_moment_i = 0
            if i > 0:
                ind_start_moment_i = self.cumsum_clips_per_moment_np[i - 1]
            ind_end_moment_i = self.cumsum_clips_per_moment_np[i]
            indices_for_moments_table.append(
                list(range(ind_start_moment_i, ind_end_moment_i)))
        indices_for_moments_table = sum(indices_for_moments_table, [])

        moment_score_list, descending_list = [], []
        for key, model_k in self.models.items():
            moments_table_k = self.moments_tables[key][
                indices_for_moments_table, :]
            scores_k, descending_k = model_k.search(
                lang_code, moments_table_k, clips_per_moment,
                clips_per_moment_list)
            moment_score_list.append(scores_k * self.alpha[key])
            descending_list.append(descending_k)
        assert descending_k == False
        moments_score = sum(moment_score_list)
        sorted_moments_score, ind_moments = moments_score.sort(
            descending=descending_k)
        if return_indices:
            # return self.video_indices[ind], self.proposals[ind, :], ind
            raise NotImplementedError('WIP')
        return video_indices[ind_moments], proposals[ind_moments, :]


class TwoStageClipPlusGeneric():
    "Two-Stage approach to retrieve moments from natural lang queries"

    def __init__(self, dataset, model, dataset_1stage, model_1ststage,
                 topk=100):
        self.dataset = dataset
        self.model = model
        model_dict_1ststage = {
            key: model_1ststage[i]
            for i, key in enumerate(dataset_1stage.cues)
        }
        self.stage1 = ClipRetrieval(dataset_1stage, model_dict_1ststage)
        self.topk = topk
        self._setup()

    @property
    def num_moments(self):
        return self.stage1.proposals.shape[0]

    def query(self, description, *args, **kwargs):
        "Return videos and moments aligned with a text description"
        # TODO (tier-2): remove 2nd-stage results from 1st-stage to make them
        # exhaustive
        torch.set_grad_enabled(False)
        lang_feature, len_query = preprocess_description(self.dataset, description)
        visited_proposals = set()

        # 1st-stage
        video_indices_1ststage, clip_indices = self.stage1.query(description)

        # 2nd-stage
        scores, video_indices, proposal_indices = [], [], []
        num_proposals = 0
        for i in range(len(clip_indices)):
            video_ind_i = video_indices_1ststage[i]
            video_id = self.dataset.videos[video_ind_i]
            moments_in_video_i = 0
            for j in self.stage1.clip2proposal_ind[clip_indices[i].item()]:
                if j in visited_proposals:
                    continue
                visited_proposals.add(j)

                # Grab features for 2nd-stage.
                # Make articifical batch and torchify.
                candidates_ij_feat = self.dataset._compute_visual_feature(
                            video_id, self.stage1.proposals[j, :].numpy())
                for k, v in candidates_ij_feat.items():
                    candidates_ij_feat[k] = torch.from_numpy(v[None, :])

                # Run 2nd-stage
                scores_ij, descending_i = self.model.predict(
                    lang_feature, len_query, candidates_ij_feat)

                scores.append(scores_ij)
                moments_in_video_i += 1
                proposal_indices.append(j)
            if moments_in_video_i > 0:
                video_indices.append(video_ind_i.repeat(moments_in_video_i))
            num_proposals += moments_in_video_i
            if num_proposals >= self.topk:
                break

        scores = torch.cat(scores)
        scores, ind = scores.sort(descending=descending_i)
        video_indices = torch.cat(video_indices)
        retrieved_proposals = self.stage1.proposals[proposal_indices, :]
        return video_indices[ind], retrieved_proposals[ind, :]

    def _setup(self):
        self.stage1.indexing()


def preprocess_description(dataset, description):
    "Return tensor representing description as 1) vectors and 2) length"
    # TODO (release): allow to tokenize description
    # assert isinstance(description, list)
    lang_feature_, len_query_ = dataset._compute_language_feature(description)

    # torchify
    lang_feature = torch.from_numpy(lang_feature_[0])
    lang_feature.unsqueeze_(0)
    len_query = torch.tensor(len_query_)
    return lang_feature, len_query


if __name__ == '__main__':
    # Unit-test
    import os
    from dataset_untrimmed import UntrimmedMCN, UntrimmedSMCN
    from proposals import SlidingWindowMSRSS, DidemoICCV17SS
    from model import MCN, SMCN
    np.random.seed(1701)
    torch.manual_seed(1701)

    charades_data = 'data/processed/charades-sta'
    dataset_setup = dict(
        json_file=f'{charades_data}/test-01.json',
        cues={'rgb': {'file': f'{charades_data}/resnet152_rgb_max_cl-3.h5'}},
        loc=0,
        context=True,
        debug=True,
        eval=True,
        proposals_interface=SlidingWindowMSRSS(
            length=3,
            scales=range(2, 9),
            stride=0.3
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
    models_dict = {'rgb': MCN(**arch_setup)}
    engine = MomentRetrievalFromProposalsTable(dataset, models_dict)
    engine.indexing()
    ind = np.random.randint(0, len(dataset))
    engine.query(dataset.metadata[ind]['language_input'])

    # TODO: check that video indices are correct

    # TODO: unit-test for MomentRetrievalFromClipBasedProposalsTable

    # TODO: unit-test for GreedyMomentRetrievalFromClipBasedProposalsTable

    # Create fakes data proportional to real dataset, it takes a while
    h5_1ststage = 'data/interim/debug/dummy_1st_retrieval.h5'
    with h5py.File(h5_1ststage, 'x') as fid:
        num_queries = len(dataset)
        num_proposals = engine.num_moments
        prop_ind = [np.random.permutation(num_proposals)
                    for i in range(num_queries)]
        prop_ind = np.row_stack(prop_ind)
        video_ind = engine.video_indices[
            prop_ind.reshape(-1)].reshape(-1, num_proposals)
        fid.create_dataset(name='proposals', data=engine.proposals)
        fid.create_dataset(name='vid_indices', data=video_ind)
        fid.create_dataset(name='proposals_ind', data=prop_ind)
    didemo_data = 'data/processed/didemo'
    dataset = UntrimmedSMCN(**dataset_setup)
    model = SMCN(**arch_setup)
    model.eval()
    engine = LoopOverKVideos(dataset, model, h5_1ststage=h5_1ststage,
                             nms_threshold=0.6)
    ind = np.random.randint(0, len(dataset))
    engine.query(dataset.metadata[ind]['language_input'], ind)

    engine = LoopOverKMoments(dataset, model, h5_1ststage=h5_1ststage,
                              nms_threshold=0.6)
    ind = np.random.randint(0, len(dataset))
    engine.query(dataset.metadata[ind]['language_input'], ind)
    os.remove(h5_1ststage)

    # unit-test ClipRetrieval
    dataset = UntrimmedSMCN(**dataset_setup)
    models_dict['rgb'] = SMCN(**arch_setup)
    engine = ClipRetrieval(dataset, models_dict)
    engine.indexing()
    ind = np.random.randint(0, len(dataset))
    engine.query(dataset.metadata[ind]['language_input'])