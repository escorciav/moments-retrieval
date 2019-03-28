"""Single video moment retrieval evaluation example

Compute metrics reported by Hendricks et. al ICCV-2017 in DiDeMo dataset. It's
only useful for DiDeMo dataset because the metrics cannot be extended to the
other annotations of existing public benchmarks (yet).
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import dataset_untrimmed
import model
from evaluation import didemo_evaluation
from proposals import DidemoICCV17SS
from utils import setup_logging, load_args_from_snapshot, collate_data_eval

METRICS = ['miou', 'rank@1', 'rank@5']
TOPK = [1, 5]
parser = argparse.ArgumentParser(
    description='Single video moment retrieval evaluation (DiDeMo)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data
parser.add_argument('--val-list', type=Path, required=True,
                    help='JSON-file with corpus instances')
parser.add_argument('--test-list', type=Path, required=True,
                    help='JSON-file with corpus instances')
parser.add_argument('--h5-path', type=Path, default=Path(''),
                    help='HDF5-file with features')
# Architecture
parser.add_argument('--snapshot', type=Path, required=True,
                    help='JSON-file file with model parameters')
# logging
parser.add_argument('--dump', action='store_true',
                    help='Save log in text file and json')
parser.add_argument('--logfile', type=Path, default=Path(''),
                    help='Logging file')
parser.add_argument('--n-display', type=float, default=0.2,
                    help='logging rate during epoch')
parser.add_argument('--disable-tqdm', action='store_true',
                    help='Disable progress-bar')
parser.add_argument('--enable-tb', action='store_true',
                    help='Log to tensorboard. Nothing logged by this program')
# Debug
parser.add_argument('--debug', action='store_true',
                help=('yield incorrect results! to verify we are gluing '
                      'things (dataset, model, eval) correctly'))
args = parser.parse_args()


def main(args):
    if args.dump:
        args.disable_tqdm = True
        if len(args.logfile.name) == 0:
            args.logfile = args.snapshot.with_name(
                args.snapshot.stem + '_didemo-eval')
        if args.logfile.exists():
            raise ValueError(
                f'{args.logfile} already exists. Please provide a logfile or'
                'backup existing results.')
    setup_logging(args)

    if load_args_from_snapshot(args):
        if len(args.snapshot.name) > 0:
            # Override snapshot config with user argument
            args = parser.parse_args(namespace=args)
            logging.info(f'Loaded snapshot config: {args.snapshot}')
    else:
        logging.error('Unable to load {args.snapshot}, procedding with args.')

    # Setup metrics
    args.topk = torch.tensor(TOPK, dtype=torch.float)

    logging.info('Single video retrieval evaluation (DiDeMo)')
    logging.info(args)

    logging.info('Setting up datasets')
    dataset = f'Untrimmed{args.arch}'
    no_visual = True
    cues = {args.feat: None}
    if args.h5_path:
        cues = {args.feat: {'file': args.h5_path}}
        no_visual = False
    dataset_per_subsets = [('val', args.val_list), ('test', args.test_list)]
    for i, (subset, filename) in enumerate(dataset_per_subsets):
        dataset_setup = dict(
            json_file=filename,
            cues=cues,
            loc=args.loc,
            context=args.context,
            debug=args.debug,
            eval=True,
            proposals_interface=DidemoICCV17SS(),
            no_visual=no_visual
        )
        if no_visual:
            dataset_setup['clip_length'] = args.clip_length
        dataset_i = dataset_untrimmed.__dict__[dataset](**dataset_setup)
        dataset_per_subsets[i] = (subset, dataset_i)

    logging.info('Setting up model')
    arch_setup = dict(
        visual_size=dataset_i.visual_size[args.feat],
        lang_size=dataset_i.language_size,
        max_length=dataset_i.max_words,
        embedding_size=args.embedding_size,
        visual_hidden=args.visual_hidden,
        lang_hidden=args.lang_hidden,
        visual_layers=args.visual_layers,
    )
    net = model.__dict__[args.arch](**arch_setup)
    snapshot = torch.load(args.snapshot.with_suffix('.pth.tar'),
                          map_location=lambda storage, loc: storage)
    net.load_state_dict(snapshot['state_dict'])
    net.eval()
    torch.set_grad_enabled(False)

    logging.info('Launch evaluation...')
    ind = 2 if args.debug else 0
    for subset, dataset in dataset_per_subsets:
        results_per_instance = []
        logging.info(f'Partition: {subset}')
        for i, moment_data in tqdm(enumerate(dataset),
                                   disable=args.disable_tqdm):
            # torchify
            lang_feat = torch.from_numpy(moment_data[ind])
            len_query = torch.tensor(moment_data[ind + 1])
            moment_feat = {key: torch.from_numpy(value)
                           for key, value in moment_data[ind + 2].items()}
            gt_segment = torch.from_numpy(moment_data[-2])
            segments = torch.from_numpy(moment_data[-1])

            # Model evaluation
            results, descending = net.predict(
                lang_feat, len_query, moment_feat)
            _, idx = results.sort(descending=descending)
            sorted_segments = segments[idx, :]

            result_i = didemo_evaluation(
                gt_segment, sorted_segments, args.topk)
            results_per_instance.append(result_i)

            if args.disable_tqdm and (i + 1) % args.n_display == 0:
                logging.info(f'Processed queries [{i}/{len(dataset)}]')

        logging.info('Summarizing results')
        results = torch.stack(results_per_instance).mean(dim=0)
        for i, metric in enumerate(results):
            key = f'{subset}_{METRICS[i]}'
            setattr(args, key, metric.item())
            logging.info(f'{subset} {METRICS[i]}: {getattr(args, key):.4f}')

    # Dumping arguments
    if args.dump:
        filename = args.snapshot.with_name(
            args.snapshot.stem + '_eval-didemo.json')

        args.topk = args.topk.tolist()
        args_dict = vars(args)
        for key, value in args_dict.items():
            if isinstance(value, Path):
                args_dict[key] = str(value)

        logging.info(f'Dumping resuts into {filename}')
        with open(filename, 'x') as fid:
            json.dump(args_dict, fid, indent=1, sort_keys=True)


if __name__ == '__main__':
    main(args)