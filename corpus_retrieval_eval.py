import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import corpus
import dataset_untrimmed
import model
import proposals
from evaluation import CorpusVideoMomentRetrievalEval, CorpusConceptVideoMomentRetrievalEval
from utils import setup_logging, get_git_revision_hash

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
parser.add_argument('--batch-size-chamfer',  type=int, default=10000,
                    help='Batch size to compute chamfer distance')
parser.add_argument('--gpu-id', type=int, default=-1, help='GPU device')
# Evaluation parameters
parser.add_argument('--topk', nargs='+', type=int,
                    default=[1, 10, 100, 1000, 10000],
                    help='top-k values to compute')
parser.add_argument('--concepts', action='store_true',
                    help='Enable evaluation of concepts, must provide the right input data.')     
parser.add_argument('--merge-rankings', action='store_true',
                    help='Enable the merging of the top k1-k2 ranked moments (standard ranking and reranking)')  
parser.add_argument('--k1', type = int,
                    help='Number of elements to take from original ranking list.')        
parser.add_argument('--k2', type = int,
                    help='Number of elements to take from re-ranking list.')          
parser.add_argument('--reordering', action='store_true',
                    help='Reorder merge')                      
# Extra
parser.add_argument('--greedy', type=int, default=0,
                    help='Top-k seed clips for greedy search over clips')
parser.add_argument('--concepts-oracle', type=int, choices=[0,1,2],
                    help='Index to select which concepts to compute:\n0-Nouns and Verbs \n1-Nouns \n2-Verbs')  
parser.add_argument('--oracle-map', type=Path, help='JSON that map concepts to videos')  
parser.add_argument('--obj-detection-path', type=Path, 
                    help='JSON with information of detected objects in clips, output of Detectron 2.')   
parser.add_argument('--use-concepts-and-obj-predictions', action='store_true',
                    help='Enable usage of both obj info and concept info for reranking.')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--new', action='store_true',
                    help='Enables working with multistream models.')
# Dump results and logs
parser.add_argument('--dump', action='store_true',
                    help='Save log in text file and json')
parser.add_argument('--logfile', type=Path, default='',
                    help='Logging file')
parser.add_argument('--n-display', type=float, default=0.2,
                    help='logging rate during epoch')
parser.add_argument('--disable-tqdm', action='store_true',
                    help='Disable progress-bar')
parser.add_argument('--dump-per-instance-results', action='store_true',
                    help='HDF5 with results')
parser.add_argument('--reduced-dump', action='store_true',
                    help='Only dump video indices per query')
parser.add_argument('--enable-tb', action='store_true',
                    help='Log to tensorboard. Nothing logged by this program')
# Debug
parser.add_argument('--debug', action='store_true',
                    help=('yield incorrect results! to verify things are'
                          'glued correctly (dataset, model, eval)'))
args = parser.parse_args()

def main(args):
    "Put all the pieces together"
    if args.dump_per_instance_results:
        args.dump = True
    if args.dump:
        args.disable_tqdm = True
        if len(args.logfile.name) == 0:
            basename_fusion = [str(i.with_suffix('').with_name(i.stem))
                               for i in args.snapshot]
            args.logfile = Path('-'.join(basename_fusion) + '_corpus-eval')
        if args.logfile.exists():
            raise ValueError(
                f'{args.logfile} already exists. Please provide a logfile or'
                'backup existing results.')
    setup_logging(args)

    logging.info('Corpus Retrieval Evaluation for CAL/MCN')
    logging.info(f'Git revision hash: {get_git_revision_hash()}')
    load_hyperparameters(args)
    logging.info(args)

    engine_prm = {}
    eval_flag = True
    if args.arch == 'MCN':
        if args.concepts: 
            args.dataset = 'UntrimmedCoceptsMCN'
        else:
            args.dataset = 'UntrimmedMCN'
        args.engine = 'MomentRetrievalFromProposalsTable'
    elif args.arch == 'SMCN':
        if args.concepts: 
            args.dataset = 'UntrimmedCoceptsSMCN'
        else:
            args.dataset = 'UntrimmedSMCN'
        args.engine = 'MomentRetrievalFromClipBasedProposalsTable'
        if args.new:
             args.engine = 'MomentRetrievalFromClipBasedProposalsTableNew'
        if args.greedy > 0:
            args.engine = 'GreedyMomentRetrievalFromClipBasedProposalsTable'
            engine_prm['topk'] = args.greedy
    elif args.arch == 'old_SMCN':
        args.dataset = 'UntrimmedSMCN_OLD'
        args.engine = 'MomentRetrievalFromClipBasedProposalsTable'

    elif args.arch == 'CALChamfer':
        eval_flag = False   # needed to get right data in indexing. We want the padded data
        args.dataset = 'UntrimmedCALChamfer'
        args.engine = 'MomentRetrievalFromClipBasedProposalsTableNew'
    elif args.arch == 'EarlyFusion':
        eval_flag = False   # needed to get right data in indexing. We want the padded data
        args.dataset = 'UntrimmedCALChamfer'
        args.engine = 'MomentRetrievalFromClipBasedProposalsTableEarlyFusion'
    else:
        ValueError('Unknown/unsupported architecture')
    if args.greedy > 0 and args.arch != 'SMCN':
        logging.warning('Ignore greedy search. Unsupported model')

    logging.info('Loading dataset')
    dataset_novisual = True
    dataset_cues = {feat: None for feat in args.tags}
    if args.h5_path:
        for i, key in enumerate(args.tags):
            dataset_cues[key] = {'file': args.h5_path[i]}
        dataset_novisual = False
        clip_length = None
    else:
        clip_length = args.clip_length
    proposals_interface = proposals.__dict__[args.proposal_interface](
        args.min_length, args.scales, args.stride)
    dataset_setup = dict(
        json_file=args.test_list, cues=dataset_cues, loc=args.loc,
        context=args.context, debug=args.debug, eval=eval_flag,
        no_visual=dataset_novisual,
        proposals_interface=proposals_interface,
        clip_length=clip_length,
        oracle=args.concepts_oracle,
        oracle_map=args.oracle_map,
        obj_detection_path=args.obj_detection_path
    )
    dataset = dataset_untrimmed.__dict__[args.dataset](**dataset_setup)
    if args.arch == 'SMCN':
        logging.info('Set padding on UntrimmedSMCN dataset')
        dataset.set_padding(False)
    elif args.arch == 'CALChamfer':
        max_clips = dataset.get_max_clips() 
        dataset.set_padding_size(max_clips)

    logging.info('Setting up models')
    
    args.device = device_name = 'cpu'
    if args.gpu_id >= 0 and torch.cuda.is_available() and args.batch_size_chamfer > 0:
        args.device = torch.device(f'cuda:{args.gpu_id}')
        device_name = torch.cuda.get_device_name(args.gpu_id)
    else:
        args.batch_size_chamfer = 0

    models_dict = {}
    if len(args.snapshot)== 1 and len(args.snapshot_tags) > 1:
        #multistream network - early fusiong
        arch_setup = dict(
            visual_size={feat:dataset.visual_size[feat] for feat in args.snapshot_tags},
            lang_size=dataset.language_size,
            max_length=dataset.max_words,
            embedding_size=args.embedding_size,
            visual_hidden=args.visual_hidden,
            lang_hidden=args.lang_hidden,
            visual_layers=args.visual_layers,
            alpha=args.alpha
        )
        key = '-'.join(args.tags)
        models_dict[key] = model.__dict__[args.arch](**arch_setup)
        filename = args.snapshot[0].with_suffix('.pth.tar')
        snapshot_ = torch.load(
            filename, map_location=lambda storage, loc: storage)
        models_dict[key].load_state_dict(snapshot_['state_dict'])
        models_dict[key].to(args.device)
        models_dict[key].eval()
    else:
        #single streams networks - late fusion
        for i, key in enumerate(args.snapshot_tags):
            arch_setup = dict(
                visual_size={key:dataset.visual_size[key]},
                lang_size=dataset.language_size,
                max_length=dataset.max_words,
                embedding_size=args.embedding_size,
                visual_hidden=args.visual_hidden,
                lang_hidden=args.lang_hidden,
                visual_layers=args.visual_layers,
                alpha=args.alpha
            )
            models_dict[key] = model.__dict__[args.arch](**arch_setup)
            filename = args.snapshot[i].with_suffix('.pth.tar')
            snapshot_ = torch.load(
                filename, map_location=lambda storage, loc: storage)
            models_dict[key].load_state_dict(snapshot_['state_dict'])
            models_dict[key].eval()

    logging.info('Creating database alas indexing corpus')
    engine_prm['alpha']  = args.alpha
    engine_prm['device'] = args.device
    engine = corpus.__dict__[args.engine](dataset, models_dict, **engine_prm)
    engine.indexing()

    logging.info('Launch evaluation...')
    # log-scale up to the end of the database
    if len(args.topk) == 1 and args.topk[0] == 0:
        exp = int(np.floor(np.log10(engine.num_moments)))
        args.topk = [10**i for i in range(0, exp + 1)]
        args.topk.append(engine.num_moments)
    num_instances_retrieved = []
    if args.concepts: 
        judge = CorpusConceptVideoMomentRetrievalEval(topk=args.topk)
    else:
        judge = CorpusVideoMomentRetrievalEval(topk=args.topk)
    args.n_display = max(int(args.n_display * len(dataset.metadata)), 1)
    for it, query_metadata in tqdm(enumerate(dataset.metadata),
                                   disable=args.disable_tqdm):
        result_per_query = engine.query(
            description = query_metadata['annotation_id'],
            return_indices=args.dump_per_instance_results,
            batch_size=args.batch_size_chamfer)
        if args.dump_per_instance_results:
            vid_indices, segments, proposals_ind, scores = result_per_query
        else:
            vid_indices, segments, scores = result_per_query
        rerank_vid_indices, rerank_segments, rerank_scores = None,None,None
        if type(args.concepts_oracle) == int and not args.obj_detection_path:
            rerank_vid_indices, rerank_segments = judge.oracle_concept_reranking(query_metadata, 
                            vid_indices, segments, dataset.metadata_per_video, dataset.reverse_map)
        elif type(args.concepts_oracle) == int and args.obj_detection_path:
            clip_length = dataset.clip_length
            if clip_length == 2.5:
                clip_length = 5
            if clip_length == 1.5:
                clip_length = 3
            clip_length = int(clip_length)
            
            rerank_vid_indices, rerank_segments, rerank_scores = judge.oracle_object_reranking(
                            query_metadata, vid_indices, segments, scores, 
                            dataset.metadata_per_video, 
                            dataset.map_concepts_to_obj_class, clip_length,
                            args.use_concepts_and_obj_predictions)
        
        if args.merge_rankings:
            vid_indices, segments = judge.merge_rankings(vid_indices, segments,scores,
                                    rerank_vid_indices, rerank_segments, rerank_scores, 
                                    args.k1, args.k2, args.reordering)
        elif type(args.concepts_oracle) == int:
            vid_indices, segments = rerank_vid_indices, rerank_segments
        
        judge.add_single_predicted_moment_info(
            query_metadata, vid_indices, segments, max_rank=engine.num_moments)
        num_instances_retrieved.append(len(vid_indices))
        if args.disable_tqdm and (it + 1) % args.n_display == 0:
            logging.info(f'Processed queries [{it}/{len(dataset.metadata)}]')

        if args.dump_per_instance_results:
            # TODO: wrap-up this inside a class. We could even dump in a
            # non-blocking thread using a Queue
            if it == 0:
                filename = args.logfile.with_suffix('.h5')
                fid = h5py.File(filename, 'w')
                if args.reduced_dump:
                    fid_vi = fid.create_dataset(
                        name='vid_indices',
                        chunks=True,
                        shape=(len(dataset), dataset.num_videos),
                        dtype='int64')
                else:
                    fid.create_dataset(
                        name='proposals', data=engine.proposals, chunks=True)
                    fid_vi = fid.create_dataset(
                        name='vid_indices',
                        chunks=True,
                        shape=(len(dataset),) + vid_indices.shape,
                        dtype='int64')
                    fid_pi = fid.create_dataset(
                        name='proposals_ind',
                        chunks=True,
                        shape=(len(dataset),) + proposals_ind.shape,
                        dtype='int64')

            if args.reduced_dump:
                fid_vi[it, ...] = pd.unique(vid_indices.numpy())
            else:
                fid_vi[it, ...] = vid_indices
                fid_pi[it, ...] = proposals_ind

    if args.dump_per_instance_results:
        fid.close()

    logging.info('Summarizing results')
    if len(judge.number_of_reranked_clips_per_query) > 0:
        avg = np.mean(np.asarray(judge.number_of_reranked_clips_per_query))
        logging.info(f'Average number of reranked moments per query: {avg}')
    num_instances_retrieved = np.array(num_instances_retrieved)
    logging.info(f'Number of queries: {len(judge.map_query)}')
    logging.info(f'Number of proposals: {engine.num_moments}')
    retrieved_proposals_median = int(np.median(num_instances_retrieved))
    retrieved_proposals_min = int(num_instances_retrieved.min())
    if (num_instances_retrieved != engine.num_moments).any():
        logging.info('Triggered approximate search')
        logging.info('Median numbers of retrieved proposals: '
                     f'{retrieved_proposals_median:d}')
        logging.info('Min numbers of retrieved proposals: '
                     f'{retrieved_proposals_min:d}')
    result = judge.evaluate()
    _ = [logging.info(f'{k}: {v}') for k, v in result.items()]
    if args.dump:
        filename = args.logfile.with_suffix('.json')
        logging.info(f'Dumping results into: {filename}')
        with open(filename, 'x') as fid:
            for key, value in result.items():
                result[key] = float(value)
            result['snapshot'] = [str(i) for i in args.snapshot]
            result['corpus'] = str(args.test_list)
            result['topk'] = args.topk
            result['iou_threshold'] = judge.iou_thresholds
            result['greedy'] = args.greedy
            result['median_proposals_retrieved'] = retrieved_proposals_median
            result['min_proposals_retrieved'] = retrieved_proposals_min
            result['date'] = datetime.now().isoformat()
            result['git_hash'] = get_git_revision_hash()
            json.dump(result, fid, indent=1)


def load_hyperparameters(args):
    "Update args with model hyperparameters"
    if args.tags is None:
        # Parse single model
        assert len(args.snapshot) == 1

        logging.info('Parsing single JSON file with hyper-parameters')
        with open(args.snapshot[0], 'r') as fid:
            if args.h5_path:
                assert len(args.h5_path) == 1
            hyper_prm = json.load(fid)
            args.tags = {hyper_prm['feat']: None}
            args.snapshot_tags = [hyper_prm['feat']]
            for key, value in hyper_prm.items():
                if not hasattr(args, key):
                    setattr(args, key, value)
            return

    logging.info('Parsing multiple JSON files with hyper-parameters')
    args.tags = dict.fromkeys(args.tags)
    if not args.new:
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