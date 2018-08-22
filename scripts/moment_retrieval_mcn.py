"""Compute Distance matrix for Corpus Video Moment Retrieval in DiDeMo val

In practice, only works for MCN models trained with pytorch. You could
get inspired for other use cases :)

"""
import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from model import MCN
from didemo import (DidemoMCN, RetrievalMode, VisualRepresentationMCN,
                    LanguageRepresentationMCN, TemporalEndpointFeature)

RGB_FEAT_PATH = 'data/interim/didemo/resnet152/320x240_max.h5'
VAL_LIST_PATH = 'data/raw/val_data_wwa.json'


parser = argparse.ArgumentParser(
    description='Exact Moment Retrieval in Didemo (val*)')
parser.add_argument('--model-pth', type=Path, help='pth-tar file')
parser.add_argument('--no-cuda', action='store_false', dest='cuda',
                    help='disable GPU')
args = parser.parse_args()
args.dataset_prm = dict(loc=True, cues=dict(rgb=dict(file=RGB_FEAT_PATH)))
args.smcn_prm = dict(visual_size=4098, lang_size=300, embedding_size=100,
                     dropout=0.3, max_length=50, visual_hidden=500,
                     lang_hidden=1000)


class DidemoMCNRetrieval(DidemoMCN):
    def __init__(self, json_file, cues=None, loc=True, max_words=50,
                 test=False, mode=RetrievalMode.MOMENT_TO_DESCRIPTION):
        self._setup_list(json_file)
        self._set_metadata_per_video()
        self._load_features(cues)
        self.visual_interface = VisualRepresentationMCN()
        self.lang_interface = LanguageRepresentationMCN(max_words)
        self.tef_interface = None
        if loc:
            self.tef_interface = TemporalEndpointFeature()
        self.eval = False
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == RetrievalMode.MOMENT_TO_DESCRIPTION:
            return self._get_moment(idx)
        elif self.mode == RetrievalMode.DESCRIPTION_TO_MOMENT:
            return self._get_phrase(idx)
        elif self.mode == RetrievalMode.VIDEO_TO_DESCRIPTION:
            return self._get_video(idx)
        else:
            raise

    def _get_moment(self, idx):
        moment_i = self.metadata[idx]
        video_id = moment_i['video']
        num_annotators = len(moment_i['times'])
        # TODO: make it flexible, but deterministic
        annot_i = 0
        time = moment_i['times'][annot_i]
        pos_visual_feature = self._compute_visual_feature(video_id, time)
        return idx, pos_visual_feature

    def _get_phrase(self, idx):
        moment_i = self.metadata[idx]
        query = moment_i['language_input']
        # TODO: pack next two vars into a dict
        sentence_feature = self.lang_interface(query)
        len_query = len(query)
        return idx, sentence_feature, len_query

    def _get_video(self, idx):
        video_id, _ = self.metada_per_video[idx]
        visual_feature = self._compute_visual_feature_eval(video_id)
        return idx, visual_feature

    def __len__(self):
        if self.mode == RetrievalMode.VIDEO_TO_DESCRIPTION:
            return len(self.metada_per_video)
        return super().__len__()


def load_model(filename=None):
    model = MCN(**args.smcn_prm)
    model.eval()
    if args.cuda:
        model.cuda()

    if filename is not None:
        snapshot = torch.load(filename)
        model.load_state_dict(snapshot['state_dict'])
    return model


def torchify_and_collate(data, unsqueeze=False):
    if isinstance(data, dict):
        return {k: torchify_and_collate(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        output = torch.from_numpy(data)
        if unsqueeze:
            output.unsqueeze_(0)
        if args.cuda:
            return output.cuda()
        return output
    elif isinstance(data, int):
        if args.cuda:
            return torch.tensor([data]).cuda()
        return torch.tensor([data])
    else:
        raise


def main():
    torch.set_grad_enabled(False)
    model = load_model(args.model_pth)
    val_dataset = DidemoMCNRetrieval(VAL_LIST_PATH, **args.dataset_prm)
    # Setup prediction matrix
    val_dataset.mode = RetrievalMode.VIDEO_TO_DESCRIPTION
    # TODO (extension): future work once we are set with DiDeMo
    N_s = len(val_dataset.segments)
    N_c = len(val_dataset) * N_s
    val_dataset.mode = RetrievalMode.DESCRIPTION_TO_MOMENT
    M_l = len(val_dataset)
    prediction_matrix = torch.empty(M_l, N_c)

    moments_order, videos_order = [], []
    for moment_i_data in tqdm(val_dataset):
        # get visual representation of a moment
        moment_i_ind = moment_i_data[0]
        sentence_i_rep = torchify_and_collate(moment_i_data[1], True)
        sentence_i_length = torchify_and_collate(moment_i_data[2])

        # Switch mode to iterate over phrases
        val_dataset.mode = RetrievalMode.VIDEO_TO_DESCRIPTION
        for video_j_data in val_dataset:
            # get text representation of sentence
            video_j_ind = video_j_data[0]
            video_j_visual_rep = torchify_and_collate(video_j_data[1])
            # TODO (debug): double check that predict works here
            # 1st check, apparently we are good to go. let's try out!
            score_ij, is_similarity = model.predict(
                sentence_i_rep, sentence_i_length, video_j_visual_rep)
            ind_start, ind_end = video_j_ind * N_s, (video_j_ind + 1) * N_s
            prediction_matrix[moment_i_ind, ind_start:ind_end] = score_ij
            # TODO (critical): block-out segments in videos without visual
            # feature e.g. a video only has 5 chunks, similarity for the 6-th
            # should be 0
            video_id = val_dataset.metada_per_video[video_j_ind][0]
            video_id_int = int(
                hashlib.sha256(video_id.encode('utf-8')).hexdigest(),
                16) % 10**8
            videos_order.append((video_j_ind, video_id_int))

        val_dataset.mode = RetrievalMode.DESCRIPTION_TO_MOMENT
        annotation_id = val_dataset.metadata[moment_i_ind]['annotation_id']
        moments_order.append((moment_i_ind, annotation_id))

    output_file = str(args.model_pth).replace(
        '_checkpoint.pth.tar', '_moment_retrieval.h5')
    with h5py.File(output_file, 'x') as fid:
        fid['prediction_matrix'] = prediction_matrix.numpy()
        fid['similarity'] = is_similarity
        fid['_video_index'] = np.array(videos_order)
        fid['_moments_index'] = np.array(moments_order)


if __name__ == '__main__':
    main()