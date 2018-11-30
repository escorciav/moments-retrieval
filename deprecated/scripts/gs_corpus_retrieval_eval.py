"""[S]MCN grid-search evaluation corpus video moment retrieval

The grid-search to search for a sweet spot over nms-threshold and topk-moments
per video. Note that the hyper-parameter space is determined by the constants
at the begining of the program.

Input:
    HDF5 with distance matrix over corpus

"""
import itertools
import os
from datetime import datetime

import fire
import pandas as pd

from evaluation import CorpusVideoMomentRetrievalEvalFromMatrix

NMS_SET = [0.5, 0.75, 1.0]
TOPK_SET = [1, 5, 10, 21]
RECALL_VALUES = (1, 5, 10, 100, 1000, 2000, 10000)
GT_ANNOTATIONS = 'data/raw/{}_data_wwa.json'.format('val')


def evaluation(filename, gt_file=GT_ANNOTATIONS, output_file=None,
               nms_threshold=1.0, topk_per_video=21, at_k=RECALL_VALUES):
    print(f'{datetime.now().isoformat()} Loading distance matrix')
    judge = CorpusVideoMomentRetrievalEvalFromMatrix(
        gt_file, filename, at_k, 0.5,
        nms_threshold=nms_threshold, topk=topk_per_video)

    print(f'{datetime.now().isoformat()} Evaluation start')
    recall, mrank = judge.eval()

    if output_file is None:
        return
    print(f'{datetime.now().isoformat()} Dumping results')
    df = pd.DataFrame([nms_threshold, topk_per_video] + recall + [mrank]).T
    df.columns = (['nms', 'topk'] + [f'R@{i}' for i in RECALL_VALUES] +
                  ['mean-rank'])
    df.to_csv(output_file, index=None)


def main(exp_id, filename):
    print(f'Experiment id: {exp_id}')
    print(f'{datetime.now().isoformat()} Searhing config')
    parameter_space = [NMS_SET, TOPK_SET]
    for i, prm in enumerate(itertools.product(*parameter_space)):
        if i != exp_id:
            continue
        nms_threshold, topk_per_video = prm
        break

    ext = os.path.splitext(filename)[1]
    output_file = filename.replace(f'_moment_retrieval{ext}',
                                   f'_r-at-k_{exp_id}.csv')
    evaluation(filename=filename, output_file=output_file,
               nms_threshold=nms_threshold, topk_per_video=topk_per_video)

    print(f'{datetime.now().isoformat()} Finished successfully')

if __name__ == '__main__':
    fire.Fire()