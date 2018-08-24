import json

import flask
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

with open('../data/processed/test/smcn_12_5_sr_rest.json') as fid:
    DB = json.load(fid)
    SENTENCE2ID = {v['description']: k for k, v in DB.items()}


def make_table_with_ranks_descriptions():
    rows = [(item['description'], item['rank'], item['rank_diff'],
             len(item['noun_subset']) > 0)
            for _, item in DB.items()]
    return rows


@app.route('/')
def index():
    return render_template(
        'sentence_retrieval_results.html',
        get_table_by_rows=make_table_with_ranks_descriptions)


@app.route('/get_sentence_results', methods=['POST'])
def get_sentence_results():
    query_id = SENTENCE2ID.get(request.form['query'])
    if query_id is not None:
        results = DB[query_id]
    else:
        results = dict(error='Query was not found in the evaluation corpus',
                       search=[{} for i in range(TOPK)])
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)