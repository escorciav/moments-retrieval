"""MCN grid-search evaluation
"""
import itertools
import os
import sys
from datetime import datetime

import h5py
from evaluation import RetrievalEvaluation

exp_id = int(sys.argv[1])
loss_set = ['', '_intra', '_inter']
cue_set = ['rgb', 'flow']
subset_set = ['val', 'test']
nms_set = [0.25, 0.5, 0.75, 1.0]
k = [1, 5, 10, 100, 1000, 5000, 10000]
output_dir = 'data/interim/mcn_retrieval_results'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

def format_loss(x):
    if '_' in x:
        return x.replace('_', '')
    elif len(x) == 0:
        return 'intra+inter'
    return x

print(f'Experiment id: {exp_id}')
print(f'{datetime.now().isoformat()} Searhing config')
parameter_space = [loss_set, cue_set, subset_set, nms_set]
for i, prm in enumerate(itertools.product(*parameter_space)):
    if i != exp_id:
        continue
    loss, cue, subset, nms_threshold = prm
    corpus_file = f'data/interim/mcn{loss}/corpus_{subset}_{cue}.hdf5'
    queries_file = f'data/interim/mcn{loss}/queries_{subset}_{cue}.hdf5'
    annotations_file = f'data/raw/{subset}_data_wwa.json'
    break

print(f'{datetime.now().isoformat()} Reading corpus and annotations')
judge = RetrievalEvaluation(
    corpus_file, annotations_file, k, nms_threshold=nms_threshold)

print(f'{datetime.now().isoformat()} Evaluation start')
with h5py.File(queries_file, 'r') as fid:
    import time
    start = time.time()
    for sample_key, h5ds in fid.items():
        query_id = int(sample_key)
        query_vector = h5ds[:]
        judge.eval_single_vector(query_vector, query_id)
print(f'{datetime.now().isoformat()} Evaluation finishes')

print(f'{datetime.now().isoformat()} Aggregating results')
recall_k, mean_rank = judge.eval()
loss = format_loss(loss)
file_output = f'{output_dir}/{exp_id}.csv'
with open(file_output, 'w') as f:
    f.write('loss,cue,subset,nms_threshold,mRank,' +
            ','.join([f'R{i}' for i in judge.k]) + '\n')
    f.write(f'{loss},{cue},{subset},{nms_threshold},'
            f'{mean_rank},' +
            ','.join([f'{i:.4f}' for i in recall_k]))
print(f'mRank={mean_rank:.2f}')
print(f'{datetime.now().isoformat()} Finished successfully')