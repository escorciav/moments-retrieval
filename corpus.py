import hashlib
import itertools
import json

import h5py
import numpy as np
import torch


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

    def query(self, description, return_indices=False):
        "Search moments based on a text description given as list of words"
        torch.set_grad_enabled(False)
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
        if return_indices:
            return self.video_indices[ind], self.proposals[ind, :], ind
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

    def query(self, description, return_indices=False):
        "Search moments based on a text description given as list of words"
        torch.set_grad_enabled(False)
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
        if return_indices:
            return self.video_indices[ind], self.proposals[ind, :], ind
        return self.video_indices[ind], self.proposals[ind, :]


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

    def query(self, description, return_indices=False):
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
    # TODO: check that video indices are correct
    # TODO: unit-test for MomentRetrievalFromClipBasedProposalsTable
    # TODO: unit-test for GreedyMomentRetrievalFromClipBasedProposalsTable
