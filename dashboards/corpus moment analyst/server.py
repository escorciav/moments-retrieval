"Dashboard to explore charades results"
import json
import os
import numpy as np
import flask
from flask import Flask, render_template, request, jsonify, url_for, redirect
from shutil import copyfile
from pathlib import Path

APP = Flask(__name__)

#Global variables instantiation:
VIDEOS_PATH, VIDEOS_PATH2, METADATA = None,None,None
extension_mapping, extensios, durations = {},{},{}
list_of_dataset = ["charades", "didemo", "activitynet"]

def initialize_data():
    global VIDEOS_PATH, VIDEOS_PATH2, METADATA
    global extension_mapping, extensios, durations

    dataset = list_of_dataset[args.dataset_index]
    print("Dataset under inspection: {}".format(dataset))

    DATA_PATH = Path('./')
    if dataset == "charades":
        VIDEOS_PATH       = Path('DATA/charades')
        JSON_FILE         = DATA_PATH / 'test.json'         # charades and didemo
        DATASET_METADATA  = "./static/METADATA/charades/smcn/1_dataset_metadata.json"
        DURATION_METADATA = "./static/duration_metadata/charades/test.json"


    if dataset == "didemo":
        VIDEOS_PATH       = Path('DATA/didemo/videos_reencoded')
        VIDEOS_PATH2      = Path('DATA/didemo/flower_reencoded')
        JSON_FILE         = DATA_PATH / 'test.json'         # charades and didemo
        DATASET_METADATA  = "./static/METADATA/didemo/smcn/1_dataset_metadata.json"
        DURATION_METADATA = "./static/duration_metadata/didemo/test.json"
        # DATASET_METADATA  = "./static/METADATA/didemo/resnet_flow.json"
        # DURATION_METADATA = "./static/METADATA/didemo/resnet_flow.json"

        with open("./static/duration_metadata/didemo/extension_mapping_didemo_test.json", 'r') as fid:
            extension_mapping = json.load(fid)


    if dataset == "activitynet":
        VIDEOS_PATH       = Path('DATA/activitynet')
        JSON_FILE         = DATA_PATH / 'val.json'        # activitynet
        DATASET_METADATA  = "./static/METADATA/activitynet/smcn/1_dataset_metadata.json"
        DURATION_METADATA = "./static/duration_metadata/activitynet/val.json"

        for l in os.listdir("./static/DATA/activitynet"):
            spl = l.split(".")
            extensios[spl[0]]=spl[1]


    with open(DATASET_METADATA, 'r') as fid:
        METADATA = json.load(fid)

    if args.dataset_index == 1:
        METADATA = METADATA['results']

    with open(DURATION_METADATA, 'r') as fid:
        DURATION = json.load(fid)

    for elem in METADATA:
        durations[elem["video"]] = DURATION["videos"][elem["video"]]["duration"]


def get_table_by_rows_index():
    rows = [(i,) +cols_for_table_index(i) for i,elem in enumerate(METADATA)]
    return rows

def cols_for_table_index(query_id):
    results = METADATA[query_id]
    return (
        results['video'],
        results['description'],
        results['rank_IoU=0.5'],
        results['rank_IoU=0.7'],
        )

def get_table_by_rows(moment_id):
    data = METADATA[moment_id]
    keys = list(durations.keys())
    # NMS
    vid_indices, nms_idx = np.unique(np.asarray(data['vid_indices']), return_index=True)
    vid_indices = [vid_indices[i] for i in np.argsort(nms_idx)][:args.number_videos]
    nms_idx = np.sort(nms_idx)[:args.number_videos]
    #Row collection
    rows = [(i,) + (keys[v],) + cols_for_table_moment(data,nms_idx[i]) +
            (url_for('static', filename=video_file(keys[v])),)
            for i,v in enumerate(vid_indices)]
    return rows

def cols_for_table_moment(data, i):
    return (
        data['segments'][i][0],
        data['segments'][i][1]
        )

def video_file(video_id):
    "return filename associated with a given video-id"
    if VIDEOS_PATH == Path('DATA/charades'):
        extension = ".mp4"

    if VIDEOS_PATH == Path('DATA/didemo/videos_reencoded'):
        extension = extension_mapping[video_id]["extension"]
        if video_id.split(".")[-1] == "":
            return (VIDEOS_PATH2 / video_id[:-1]).with_suffix(extension)

    if VIDEOS_PATH == Path('DATA/activitynet'):
        extension = "."+extensios[video_id]

    return (VIDEOS_PATH / video_id).with_suffix(extension)

@APP.route('/')
def index():
    "index page"
    return render_template('index_page.html',
                            get_table_by_rows=get_table_by_rows_index,
                            dataset=list_of_dataset[args.dataset_index])

@APP.route('/results/<int:moment_id>')
def moment_analyst(moment_id):
    if 0 <= moment_id < len(METADATA):
        result = METADATA[moment_id]
        # GT DATA
        video_id = video_file(result['video'])
        description = result['description']

        #PREDICTED DATA - HERE WE APPLY NMS IN THE FOLLOWING 3 LINES
        # the trick with various sorting is neede because np.unique orders the output.
        vid_indices, nms_idx = np.unique(np.asarray(result['vid_indices']), return_index=True)
        vid_indices = [vid_indices[i] for i in np.argsort(nms_idx)][:args.number_videos]
        nms_idx = np.sort(nms_idx)[:args.number_videos]

        # use index retrieved from nms procedure to select correct temporal segments
        topk_segments = np.asarray([result['segments'][i] for i in nms_idx])
        keys = list(durations.keys())
        # retrieve correct video names
        video_names = [keys[v] for v in vid_indices]

        return render_template(
            # 'single_moment_player.html',
            'corpus_moment_player.html',
            video_file=url_for('static', filename=video_id),
                                        description=description,
                                        num_predictions=len(topk_segments),
                                        get_table_by_rows=get_table_by_rows,
                                        durations=durations,
                                        video_names=video_names,
                                        moment_id=moment_id)
    else:
        return f'Moment not found: {moment_id}'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                        description='Corpus Retrieval Evaluation Visualization Tool',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset-index', type=int, required=True,
                        help='Integer value of indexing for the dataset selection \n'+
                              '0 = charades-sta \n1 = DiDeMo \n2 = activitynet-captions')

    parser.add_argument('--number-videos', type=int, required=False, default=10,
                        help='Number of videos to visualize on corpus inspection page.')

    args = parser.parse_args()

    print("Supplied dataset index: {}".format(args.dataset_index))
    print("Number of videos to visualize: {}".format(args.number_videos))
    initialize_data()

    if args.dataset_index == 0:
        port = 60000
    elif args.dataset_index == 1:
        port = 60001
    elif args.dataset_index == 2:
        port = 60002

    APP.run(debug=True,host="0.0.0.0", port=port)
