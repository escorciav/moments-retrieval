from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import os
import random
from pathlib import Path

import flask
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
TOPK = 20
SUBSET = ''
CLIP_SIZE = 5
DATABASE_FILE = ('../data/interim/mcn_retrieval_results/'
                 'rest_val_intra+inter_rgb+flow.json')


with open(DATABASE_FILE, 'r') as f:
    DB = json.load(f)
    QUERY2ID = {v['query']: i for i, v in DB['results'].items()}


def make_gif_name(video_name, segment):
    "Make GIF name out of video-name and segment (interval in seconds)"
    video_name = Path(video_name).stem
    gif_file = os.path.join(
        f'gif{SUBSET}', f'{video_name}_{segment[0]}-{segment[1]}.gif')
    return flask.url_for('static', filename=gif_file)


def rest_gt(query_id):
    results = DB['results'][query_id]
    video_index = results['groundtruth_video_index']
    segment_indices = results['groundtruth_segment_indices']
    segments = [segment_d2c(DB['segments'][segment_index])
                for segment_index in segment_indices]
    random.shuffle(segments)
    video_name = DB['videos'][video_index]
    segment = segments[0]
    output = dict(video=video_name, segments=segments,
                  path=make_gif_name(video_name, segment))
    return output


def rest_topk_search(query_id):
    results = DB['results'][query_id]
    output = {'query': results['query']}
    output['search'] = []
    for i in range(TOPK):
        video_index = results['topk_video_indices'][i]
        segment_index = results['topk_segment_indices'][i]
        tp_fp = results['tp_fp_labels'][i]
        video_name = DB['videos'][video_index]
        segment = segment_d2c(DB['segments'][segment_index])
        top_i = {'video': video_name,
                 'segment': segment,
                 'score': results['topk_scores'][i],
                 'path': make_gif_name(video_name, segment),
                 'true_positive': tp_fp}
        output['search'].append(top_i)
    return output


def segment_d2c(x):
    "Transform segment extend from discrete indices to time"
    t_start = x[0] * CLIP_SIZE
    t_end = (x[1] + 1) * CLIP_SIZE
    return t_start, t_end


def cols_for_table(query_id):
    results = DB['results'][query_id]
    return (
        results['rank'],
        results['video_rank'],
        results['unique_videos_at_k'],
        )


def get_table_by_rows():
    rows = [(i,) + cols_for_table(v) for i, v in QUERY2ID.items()]
    return rows


@app.route('/')
def index():
    return render_template('moment_explorer.html',
                           get_table_by_rows=get_table_by_rows)


@app.route('/process', methods=['POST'])
def process():
    query_id = request.form['query']
    query_id = QUERY2ID[request.form['query']]
    if query_id in DB['results']:
        results = rest_topk_search(query_id)
        results = {**results, **rest_gt(query_id)}
    else:
        results = dict(error='Query was not found in the evaluation corpus',
                       search=[{} for i in range(TOPK)])
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)