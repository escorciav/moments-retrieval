import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import corpus
import dataset_untrimmed
import model
import proposals
from evaluation import CorpusVideoMomentRetrievalEval
from utils import setup_logging\

# TODO(tier-2;clean): remove this hard-coded approach
# we not only use the same arch, but also the same hyper-prm
UNIQUE_VARS = {key: [] for key in
               ['arch', 'loc', 'context', 'proposal_interface']}

parser = argparse.ArgumentParser(
    description='Corpus Retrieval Evaluation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data
parser.add_argument('--test-list', type=Path, required=True,
                    help='JSON-file with corpus instances')
parser.add_argument('--h5-path', type=Path, nargs='+',
                    help='HDF5-file with features')
parser.add_argument('--tags', nargs='+',
                    help='Tag for h5-file features')
# Architecture
parser.add_argument('--snapshot', type=Path, required=True, nargs='+',
                    help='JSON files of model')
parser.add_argument('--snapshot-tags', nargs='+',
                    help='Pair model to a given h5-path')
# Evaluation parameters
parser.add_argument('--topk', nargs='+', type=int,
                    default=[1, 10, 100, 1000, 10000],
                    help='top-k values to compute')
# Extra
parser.add_argument('--greedy', type=int, default=0,
                    help='Top-k seed clips for greedy search over clips')
# Dump results and logs
parser.add_argument('--dump', action='store_true',
                    help='Save log in text file and json')
parser.add_argument('--logfile', type=Path, default='',
                    help='Logging file')
parser.add_argument('--n-display', type=float, default=0.2,
                    help='logging rate during epoch')
parser.add_argument('--disable-tqdm', action='store_true',
                    help='Disable progress-bar')
# Debug
parser.add_argument('--debug', action='store_true',
                    help=('yield incorrect results! to verify things are'
                            'glued correctly (dataset, model, eval)'))
args = parser.parse_args()


def main(args):
    "Put all the pieces together"
    if args.dump:
        args.disable_tqdm = True
        if len(args.logfile.name) == 0:
            args.logfile = args.snapshot_args.with_suffix('').with_name(
                args.snapshot_args.stem + '_corpus-eval')
        if args.logfile.exists():
            raise ValueError(
                f'{args.logfile} already exists. Please provide a logfile or'
                'backup existing results.')
    setup_logging(args)

    logging.info('Corpus Retrieval Evaluation for *MCN')
    load_hyperparameters(args)

    engine_prm = {}
    if args.arch == 'MCN':
        args.dataset = 'UntrimmedMCN'
        args.engine = 'MomentRetrievalFromProposalsTable'
    elif args.arch == 'SMCN':
        args.dataset = 'UntrimmedSMCN'
        args.engine = 'MomentRetrievalFromClipBasedProposalsTable'
        if args.greedy > 0:
            args.engine = 'GreedyMomentRetrievalFromClipBasedProposalsTable'
            engine_prm['topk'] = args.greedy
    else:
        ValueError('Unknown/unsupported architecture')
    if args.greedy > 0 and args.arch != 'SMCN':
        logging.warning('Ignore greedy search. Unsupported model')

    logging.info('Loading dataset')
    dataset_novisual = True
    dataset_cues = {feat: None for feat in args.tags}
    if len(args.h5_path) > 0:
        for i, key in enumerate(args.tags):
            dataset_cues[key] = {'file': args.h5_path[i]}
        dataset_novisual = False
    proposals_interface = proposals.__dict__[args.proposal_interface]
    proposals_prm = dict(
        length=args.min_length, stride=args.stride,
        num_scales=args.num_scales, unique=True)
    dataset_setup = dict(
        json_file=args.test_list, cues=dataset_cues, loc=args.loc,
        context=args.context, debug=args.debug, eval=True,
        no_visual=dataset_novisual,
        proposals_interface=proposals_interface(**proposals_prm)
    )
    dataset = dataset_untrimmed.__dict__[args.dataset](**dataset_setup)
    if args.arch == 'SMCN':
        logging.info('Set padding on UntrimmedSMCN dataset')
        dataset.set_padding(False)

    logging.info('Setting up models')
    models_dict = {}
    for i, key in enumerate(args.snapshot_tags):
        arch_setup = dict(
            visual_size=dataset.visual_size[key],
            lang_size=dataset.language_size,
            max_length=dataset.max_words,
            embedding_size=args.embedding_size,
            visual_hidden=args.visual_hidden,
            lang_hidden=args.lang_hidden,
            visual_layers=args.visual_layers,
        )
        models_dict[key] = model.__dict__[args.arch](**arch_setup)
        filename = args.snapshot[i].with_suffix('.pth.tar')
        snapshot_ = torch.load(
            filename, map_location=lambda storage, loc: storage)
        models_dict[key].load_state_dict(snapshot_['state_dict'])
        models_dict[key].eval()

    logging.info('Creating database alas indexing corpus')
    engine = corpus.__dict__[args.engine](dataset, models_dict, **engine_prm)
    engine.indexing()

    logging.info('Launch evaluation...')
    # log-scale up to the end of the database
    if len(args.topk) == 1 and args.topk[0] == 0:
        exp = int(np.floor(np.log10(engine.num_moments)))
        args.topk = [10**i for i in range(0, exp + 1)]
        args.topk.append(engine.num_moments)
    num_instances_retrieved = []
    judge = CorpusVideoMomentRetrievalEval(topk=args.topk)
    args.n_display = max(int(args.n_display * len(dataset.metadata)), 1)
    for it, query_metadata in tqdm(enumerate(dataset.metadata),
                                   disable=args.disable_tqdm):
        vid_indices, segments = engine.query(query_metadata['language_input'])
        judge.add_single_predicted_moment_info(
            query_metadata, vid_indices, segments, max_rank=engine.num_moments)
        num_instances_retrieved.append(len(vid_indices))
        if args.disable_tqdm and (it + 1) % args.n_display == 0:
            logging.info(f'Processed queries [{it}/{len(dataset.metadata)}]')

    logging.info('Summarizing results')
    moments_scanned_median = np.median(num_instances_retrieved)
    logging.info(f'Number of queries: {len(judge.map_query)}')
    logging.info(f'Number of proposals: {engine.num_moments}')
    logging.info('Median numbers of moments scanned: '
                    f'{int(moments_scanned_median):d}')
    result = judge.evaluate()
    _ = [logging.info(f'{k}: {v}') for k, v in result.items()]
    if args.dump:
        filename = args.logfile.with_suffix('.json')
        logging.info(f'Dumping results into: {filename}')
        with open(filename, 'x') as fid:
            for key, value in result.items():
                result[key] = float(value)
            result['median_moments_scanned'] = moments_scanned_median
            result['snapshot'] = str(args.snapshot)
            result['snapshot_args'] = str(args.snapshot_args)
            result['corpus'] = str(args.test_list)
            result['greedy'] = args.greedy
            result['date'] = datetime.now().isoformat()
            json.dump(result, fid, indent=1)


def load_hyperparameters(args):
    "Update args with model hyperparameters"
    if len(args.tags) == 0:
        single_model_and_cues = len(args.snapshot) == len(args.h5_path)
        assert single_model_and_cues
        # take tag of first model
        logging.info('Set JSON files with hyper-parameters')
        with open(args.snapshot[0], 'r') as fid:
            hyper_prm = json.load(fid)
            args.tags = [hyper_prm['feat']]
            args.snapshot_tags = [args.tags[0]]

    logging.info('Parsing JSON files with hyper-parameters')
    args.tags = dict.fromkeys(args.tags)
    assert len(args.h5_path) == len(args.tags)
    for i, filename in enumerate(args.snapshot):
        with open(filename, 'r') as fid:
            hyper_prm = json.load(fid)
            assert args.snapshot_tags[i] in args.tags
            for key, value in hyper_prm.items():
                if not hasattr(args, key):
                    setattr(args, key, value)
                if key in UNIQUE_VARS:
                    UNIQUE_VARS[key].append(value)

    for value in UNIQUE_VARS.values():
        assert len(np.unique(value)) == 1


if __name__ == '__main__':
    main(args)