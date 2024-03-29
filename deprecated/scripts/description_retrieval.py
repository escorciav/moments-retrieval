"Retrieve descriptions over corpus for all moments in the dataset"
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from model import SMCN
from didemo import DidemoSMCNRetrieval, RetrievalMode

RGB_FEAT_PATH = 'data/interim/didemo/resnet152/320x240_max.h5'
VAL_LIST_PATH = 'data/raw/val_data.json'


parser = argparse.ArgumentParser(description='Phrase Retrieval')
parser.add_argument('--model-pth', type=Path, help='pth-tar file')
parser.add_argument('--no-cuda', action='store_false', dest='cuda',
                    help='disable GPU')
args = parser.parse_args()
args.dataset_prm = dict(context=False, loc=False,
                        cues=dict(rgb=dict(file=RGB_FEAT_PATH)))
# TODO add stuff to change SMCN_PRM
args.smcn_prm = dict(visual_size=2048, lang_size=300, embedding_size=100,
                     dropout=0.3, max_length=50, visual_hidden=500,
                     lang_hidden=1000)


def load_model(filename=None):
    model = SMCN(**args.smcn_prm)
    model.eval()
    if args.cuda:
        model.cuda()

    if filename is not None:
        snapshot = torch.load(filename)
        model.load_state_dict(snapshot['state_dict'])
    return model


def torchify_and_collate(data):
    if isinstance(data, dict):
        if args.cuda:
            return {k: torch.from_numpy(v).unsqueeze_(0).cuda()
                    for k, v in data.items()}
        return {k: torch.from_numpy(v).unsqueeze_(0)
                for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        if args.cuda:
            return torch.from_numpy(data).unsqueeze_(0).cuda()
        return torch.from_numpy(data).unsqueeze_(0)
    elif isinstance(data, int):
        if args.cuda:
            return torch.tensor([data]).cuda()
        return torch.tensor([data])
    else:
        raise


def main():
    torch.set_grad_enabled(False)
    model = load_model(args.model_pth)
    val_dataset = DidemoSMCNRetrieval(VAL_LIST_PATH, **args.dataset_prm)
    # Ensure mode to iterate over moments
    val_dataset.mode = RetrievalMode.MOMENT_TO_DESCRIPTION
    N = len(val_dataset)
    prediction_matrix = torch.empty(N, N)
    for moment_i_data in tqdm(val_dataset):
        # get visual representation of a moment
        moment_i_ind = moment_i_data[0]
        moment_i_visual_rep = torchify_and_collate(moment_i_data[1])

        # Switch mode to iterate over descriptions
        val_dataset.mode = RetrievalMode.DESCRIPTION_TO_MOMENT
        for moment_j_data in val_dataset:
            # get text representation of sentence
            moment_j_ind = moment_j_data[0]
            sentence_j_rep = torchify_and_collate(moment_j_data[1])
            sentence_j_length = torchify_and_collate(moment_j_data[2])
            score_j, is_similarity = model.predict(
                sentence_j_rep, sentence_j_length, moment_i_visual_rep)
            prediction_matrix[moment_i_ind, moment_j_ind] = score_j

        val_dataset.mode = RetrievalMode.MOMENT_TO_DESCRIPTION

    output_file = str(args.model_pth).replace(
        '_checkpoint.pth.tar', '_description_retrieval.h5')
    with h5py.File(output_file, 'x') as fid:
        fid['prediction_matrix'] = prediction_matrix.numpy()
        fid['similarity'] = is_similarity


if __name__ == '__main__':
    main()