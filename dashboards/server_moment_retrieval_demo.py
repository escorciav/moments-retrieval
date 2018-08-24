import os
from pathlib import Path

import flask
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from flask import Flask, render_template, request, jsonify

from corpus import Corpus
from didemo import sentences_to_words
from glove import RecurrentEmbedding

app = Flask(__name__)
TOPK = 20
CLIP_SIZE = 5
SUBSET = 'val'

MODEL = 'mcn'
ALPHA = 0.5
RGB_CORPUS_H5 = f'../data/interim/{MODEL}/corpus_{SUBSET}_rgb.hdf5'
FLOW_CORPUS_H5 = f'../data/interim/{MODEL}/corpus_{SUBSET}_flow.hdf5'
RGB_WEIGHTS = '../data/interim/mcn/rgb-weights.hdf5'
FLOW_WEIGHTS = '../data/interim/mcn/flow-weights.hdf5'
MAX_WORDS = 50
GLOVE_FILE = '../data/raw/glove.6B.300d.txt'
GLOVE_VOCAB = '../data/raw/vocab_glove_complete.txt'
RGB_CORPUS = Corpus(RGB_CORPUS_H5)
FLOW_CORPUS = Corpus(FLOW_CORPUS_H5, videos=RGB_CORPUS.videos)
# We don't care about gradients during inference
torch.set_grad_enabled(False)


class MCNRetrievalFromCaffe(nn.Module):
    "Use MCN trained on Caffe weights"

    def __init__(self, lang_size=300, embedding_size=100, lang_hidden=1000,
                 max_length=50):
        super(MCNRetrievalFromCaffe, self).__init__()
        self.embedding_size = embedding_size
        self.max_length = max_length

        self.sentence_encoder = nn.LSTM(
            lang_size, lang_hidden, batch_first=True)
        self.lang_encoder = nn.Linear(lang_hidden, embedding_size)

    def forward(self, padded_query, query_length):
        # Keep the same signature but does not use neg inputs
        B = len(padded_query)

        packed_query = pack_padded_sequence(
            padded_query, query_length, batch_first=True)
        packed_output, _ = self.sentence_encoder(packed_query)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=self.max_length)
        last_output = output[range(B), query_length - 1, :]
        l_embedding = self.lang_encoder(last_output)
        return l_embedding

    def load_caffe_weights(self, filename):
        with h5py.File(filename) as f:
            ported_weights = {}
            mapping = {}
            for k, v in f.items():
                # print(k, v.shape)
                # ignore visual-part 'cause we have it in a "database"
#                 if k == 'InnerProduct1_0':
#                     mapping['img_encoder.0.weight'] = k
#                 elif k == 'InnerProduct1_1':
#                     mapping['img_encoder.0.bias'] = k
#                 elif k == 'InnerProduct2_0':
#                     mapping['img_encoder.2.weight'] = k
#                 elif k == 'InnerProduct2_1':
#                     mapping['img_encoder.2.bias'] = k
                if k == 'LSTM1_0':
                    dim = v.shape[0] // 4
                    v = np.concatenate([v[:2*dim, ...],
                                        v[3*dim:4*dim, ...],
                                        v[2*dim:3*dim, ...]], axis=0)
                    mapping['sentence_encoder.weight_ih_l0'] = k
                elif k == 'LSTM1_2':
                    dim = v.shape[0] // 4
                    v = np.concatenate([v[:2*dim, ...],
                                        v[3*dim:4*dim, ...],
                                        v[2*dim:3*dim, ...]], axis=0)
                    mapping['sentence_encoder.weight_hh_l0'] = k
                elif k == 'LSTM1_1':
                    dim = v.shape[0] // 4
                    v = np.concatenate([v[:2*dim, ...],
                                        v[3*dim:4*dim, ...],
                                        v[2*dim:3*dim, ...]], axis=0)
                    # sentence_encoder.bias_hh_l0
                    # sentence_encoder.bias_ih_l0
                    mapping['sentence_encoder.bias_hh_l0'] = k
                elif k == 'embedding_text_0':
                    mapping['lang_encoder.weight'] = k
                elif k == 'embedding_text_1':
                    mapping['lang_encoder.bias'] = k
                ported_weights[k] = v[:]

            # Set parameters
            for name, parameter in self.named_parameters():
                # print(name, parameter.shape)
                if name == 'sentence_encoder.bias_ih_l0':
                    parameter.data.zero_()
                    continue
                parameter.data = torch.from_numpy(
                    ported_weights[mapping[name]])


class LanguageRepresentationMCN(object):
    "Get representation of sentence"

    def __init__(self, max_words, glove_file, vocab_file):
        self.max_words = max_words
        self.rec_embedding = RecurrentEmbedding(
            glove_file, vocab_file=vocab_file)
        self.dim = self.rec_embedding.embedding.glove_dim

    def __call__(self, query_str):
        "Return padded sentence feature"
        assert isinstance(query_str, str)
        query = sentences_to_words([query_str])
        feature = np.zeros((self.max_words, self.dim), dtype=np.float32)
        len_query = min(len(query), self.max_words)
        for i, word in enumerate(query[:len_query]):
            if word in self.rec_embedding.vocab_dict:
                feature[i, :] = self.rec_embedding.vocab_dict[word]
        return feature, len_query

LANG_PREPROCESSOR = LanguageRepresentationMCN(
    MAX_WORDS, GLOVE_FILE, GLOVE_VOCAB)
RGB_MODEL = MCNRetrievalFromCaffe()
RGB_MODEL.load_caffe_weights(RGB_WEIGHTS)
FLOW_MODEL = MCNRetrievalFromCaffe()
FLOW_MODEL.load_caffe_weights(FLOW_WEIGHTS)
RGB_MODEL.eval()
FLOW_MODEL.eval()

def mcn_search(query):
    padded_query, len_query = LANG_PREPROCESSOR(query)
    padded_query = torch.from_numpy(padded_query).unsqueeze(0)
    len_query = torch.tensor([len_query])
    embedded_query_rgb = RGB_MODEL(padded_query, len_query)
    embedded_query_flow = FLOW_MODEL(padded_query, len_query)
    rgb_distance = RGB_CORPUS.search(
        embedded_query_rgb.squeeze_().detach().numpy())
    flow_distance = FLOW_CORPUS.search(
        embedded_query_flow.squeeze_().detach().numpy())
    distance = ALPHA * rgb_distance + (1 - ALPHA) * flow_distance
    # Manual indexing
    distance_sorted_indices = np.argsort(distance)
    distance_sorted = distance[distance_sorted_indices]
    results = RGB_CORPUS.ind_to_repo(distance_sorted_indices)
    # TODO: add NMS
    video_indices, segment_indices = results[:2]
    topk_video_indices = video_indices[:TOPK]
    topk_segment_indices = segment_indices[:TOPK]
    topk_score = distance_sorted[:TOPK]
    results = rest_topk_search(topk_video_indices, topk_segment_indices,
                               topk_score)
    return results


def make_gif_name(video_name, segment):
    "Make GIF name out of video-name and segment (interval in seconds)"
    video_name = Path(video_name).stem
    gif_file = os.path.join(
        'gif', f'{video_name}_{segment[0]}-{segment[1]}.gif')
    return flask.url_for('static', filename=gif_file)


def rest_topk_search(topk_video_indices, topk_segment_indices, topk_scores):
    output = dict(search=[])
    for i in range(TOPK):
        video_index = topk_video_indices[i]
        segment_index = topk_segment_indices[i]
        video_name = str(RGB_CORPUS.videos[video_index])
        segment = segment_d2c(RGB_CORPUS.segments[segment_index])
        top_i = {'video': video_name,
                 'segment': segment,
                 'score': f'{topk_scores[i]:.5f}',
                 'path': make_gif_name(video_name, segment)
                }
        output['search'].append(top_i)
    return output


def segment_d2c(x):
    "Transform segment extend from discrete indices to time"
    t_start = int(x[0] * CLIP_SIZE)
    t_end = int((x[1] + 1) * CLIP_SIZE)
    return t_start, t_end


@app.route('/')
def moment_retrieval_page():
    return render_template('moment_retrieval_demo.html')


@app.route('/search', methods=['POST'])
def process():
    query = request.form['query']
    try:
        results = mcn_search(query)
        # TODO: add time and other statistics
        results['time'] = None
    except:
        results = dict(error='Query was not found in the evaluation corpus',
                       search=[{} for i in range(TOPK)])
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)