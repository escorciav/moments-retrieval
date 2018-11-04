"""Single video moment retrieval evaluation example

Compute metrics reported by Anne et. al n ICCV-2017 in DiDeMo dataset. It's
only useful for DiDeMo dataset because the metrics cannot be extended to the
other annotations of existing public benchmarks (yet).
The program overwrites the snapshot JSON-file with the test&val_iou/r@1/r@5.
It will backup the original JSON-file as .json.BAK.

Raises
    ValueError if a `.BAK` file already exists for the snapshot.

TODO
    - KIS
    - Currently we are handling four different setups, namely two Dataset
      abstractions and two evaluation functions. Once we figure it out why the
      results do not match on real data, we will apply KIS.
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import dataset_untrimmed
import didemo
import model
from evaluation import didemo_evaluation, video_evaluation
from proposals import DidemoICCV17SS
from utils import setup_logging, collate_data_eval

METRICS = ['iou', 'r@1', 'r@5']
parser = argparse.ArgumentParser(
    description='Single video moment retrieval evaluation (DiDeMo)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data
parser.add_argument('--val-list', type=Path, required=True,
                    help='JSON-file with corpus instances')
parser.add_argument('--test-list', type=Path, required=True,
                    help='JSON-file with corpus instances')
parser.add_argument('--h5-path', type=Path, required=True,
                    help='HDF5-file with features')
# Architecture
parser.add_argument('--snapshot', type=Path, required=True,
                    help='pht.tar file with model parameters')
parser.add_argument('--snapshot-args', type=Path, required=True,
                    help='JSON-file file with model parameters')
# Extras
parser.add_argument('--arch', choices=model.MOMENT_RETRIEVAL_MODELS,
                    default='MCN',
                    help='model architecture, only for old JSON files')
parser.add_argument('--dataset', choices=model.MOMENT_RETRIEVAL_MODELS,
                    default='UntrimmedMCN',
                    help='model architecture, only for old JSON files')
parser.add_argument('--didemo-module', action='store_true',
                    help='Use this for models trained with didemo module')
parser.add_argument('--python-eval', action='store_true',
                    help='Use original evaluation functions')
# logging
parser.add_argument('--disable-tqdm', action='store_true')
# Debug
parser.add_argument('--debug', action='store_true',
                help=('yield incorrect results! to verify we are gluing '
                      'things (dataset, model, eval) correctly'))
args = parser.parse_args()


def main(args):
    TOPK = torch.tensor((1.0, 5.0))
    args.logfile = Path('')
    setup_logging(args)
    logging.info('Single video retrieval evaluation (DiDeMo)')
    logging.info(args)
    logging.info('Parsing JSON file with hyper-parameters')
    with open(args.snapshot_args, 'r') as fid:
        model_hp = json.load(fid)
        if model_hp.get('arch') is None:
            logging.warning(f'Old JSON-file. Using arch: {args.arch}')
            model_hp['arch'] = args.arch
        args.arch = model_hp['arch']

    eval_fn = didemo_evaluation
    if args.python_eval:
        TOPK = TOPK.tolist()
        eval_fn = wrapper_video_evaluation

    dataset_per_subsets = setup_dataset(args, model_hp)
    dataset = dataset_per_subsets[0][1]

    logging.info('Setting up model')
    if model_hp.get('lang_hidden') is None:
        # Fallback option when we weren't recording arch hyper-parameters
        # TODO(tier-2;cleanup;release): move this to model or utils
        logging.warning('Inferring model hyper-parameters from snapshot')
        weights = load_model_weights(args.snapshot)
        model_hp['embedding_size'] = weights['lang_encoder.bias'].shape[0]
        model_hp['visual_hidden'] = weights['visual_encoder.0.bias'].shape[0]
        # Assumed that we were using LSTM
        model_hp['lang_hidden'] = weights[
            'sentence_encoder.weight_ih_l0'].shape[0] // 4
        num_layers = sum([1 for layer_name in weights.keys()
                          if layer_name.startswith('visual_encoder')]) // 2
        model_hp['visual_layers'] = num_layers - 1
    arch_setup = dict(
        visual_size=dataset.visual_size['rgb'],
        lang_size=dataset.language_size,
        max_length=dataset.max_words,
        embedding_size=model_hp['embedding_size'],
        visual_hidden=model_hp['visual_hidden'],
        lang_hidden=model_hp['lang_hidden'],
        visual_layers=model_hp['visual_layers'],
    )
    net = model.__dict__[args.arch](**arch_setup)
    net.load_state_dict(load_model_weights(args.snapshot))
    net.eval()
    torch.set_grad_enabled(False)

    logging.info('Launch evaluation...')
    ind = 2 if args.debug else 0
    collate_fn = collate_data_eval
    for subset, dataset in dataset_per_subsets:
        ind_end = -2
        if args.didemo_module:
            ind, ind_end = 2, None
            collate_fn = dataset.collate_test_data
        results_per_instance = []
        for i, moment_data in tqdm(enumerate(dataset),
                                   disable=args.disable_tqdm):
            moment_data = collate_fn([moment_data])

            # Model evaluation
            results, descending = net.predict(*moment_data[ind:ind_end])
            _, idx = results.sort(descending=descending)

            # Get segments
            if args.didemo_module:
                idx_h = idx.to('cpu')
                sorted_segments = [dataset.segments[i] for i in idx_h]
                gt_segment = dataset.metadata[i]['times']
                if not args.python_eval:
                    sorted_segments = torch.tensor(sorted_segments).float()
                    gt_segment = torch.tensor(gt_segment).float()
                    make_continuous(sorted_segments)
                    make_continuous(gt_segment)
            else:
                gt_segment, segments = moment_data[-2:]
                sorted_segments = segments[idx, :]
                if args.python_eval:
                    make_indices(gt_segment)
                    make_indices(sorted_segments)
                    gt_segment = gt_segment.tolist()
                    sorted_segments = sorted_segments.tolist()

            # Evaluation
            results_per_instance.append(
                eval_fn(gt_segment, sorted_segments, TOPK))

        logging.info('Summarizing results')
        if args.python_eval:
            results = np.row_stack(results_per_instance).mean(axis=0)
        else:
            results = torch.stack(results_per_instance).mean(dim=0)
        for i, metric in enumerate(results):
            model_hp[f'{subset}_{METRICS[i]}'] = metric.item()

    backup = args.snapshot_args.with_name(args.snapshot_args.name + '.BAK')
    if not backup.exists():
        args.snapshot_args.replace(backup)
    else:
        raise ValueError('Wont override backup, please take action')
    with open(args.snapshot_args, 'w') as fid:
        json.dump(model_hp, fid)


def load_model_weights(filename):
    "Device agnostic loading"
    data = torch.load(filename, map_location=lambda storage, loc: storage)
    return data['state_dict']


def make_continuous(x):
    "In-place mapping of discrete time units tensor, [N x 2], to seconds"
    x *= 5
    x[:, 1] += 5


def make_indices(x):
    "In-place mapping of seconds tensor, [N x 2], to time units"
    x[:, 1] -= 5
    x /= 5


def setup_dataset(args, model_hp):
    if args.didemo_module:
        dataset_module = didemo
        args.dataset = f'Didemo{args.arch}'
        dataset_setup = dict(
            json_file=None,
            cues={model_hp['feat']: {'file': args.h5_path}},
            loc=model_hp['loc'],
            test=True
        )
        if args.arch == 'SMCN':
            dataset_setup['context'] = model_hp['context'],
    else:
        dataset_module = dataset_untrimmed
        args.dataset = f'Untrimmed{args.arch}'
        dataset_setup = dict(
            json_file=None,
            cues={model_hp['feat']: {'file': args.h5_path}},
            loc=model_hp['loc'],
            context=model_hp['context'],
            debug=args.debug,
            eval=True,
            proposals_interface=DidemoICCV17SS()
        )

    logging.info('Loading datasets')
    datasets = []
    for subset, filename in [('test', args.test_list),
                             ('val', args.val_list)]:
        if not filename.exists():
            raise ValueError
        dataset_setup['json_file'] = filename
        datasets.append(
            (subset, dataset_module.__dict__[args.dataset](**dataset_setup))
        )
    return datasets


def wrapper_video_evaluation(*args):
    "Wrapper to return numpy array and ease evaluation"
    return np.array(video_evaluation(*args), dtype=np.float32)


if __name__ == '__main__':
    main(args)