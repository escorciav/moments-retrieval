import json
from collections import Counter
import spacy
import random
import numpy as np
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
import datetime 
import utils
import os
import warnings
import pandas as pd 
import argparse
import tqdm

def count_noun_didemo(dataset, concept_type):
    '''
    DEPRECATED
    '''
    warnings.warn('DEPRECATED FUNCTION')
    nlp = spacy.load('en_core_web_sm')
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        num_descriptions = 0
        didemo_nouns = Counter()
        filename = f'./data/raw/{dataset}/{subset}_data.json'
        with open(filename) as f:
            data = json.load(f)
        for d in data:
            num_descriptions += 1
            d_i = d['description']
            doc_i = nlp(d_i)
            doc_i_nouns = Counter()
            for token in doc_i:
                if token.pos_ in concept_type:  
                    doc_i_nouns.update({token.lemma_: 1})
            didemo_nouns.update(doc_i_nouns)
        print('Number of descriptions', num_descriptions)
        print(f'Number of {" and ".join(concept_type)}', len(didemo_nouns))

        # Comment the following lines if you are not interested in
        # dumping CSV with counts of NOUNs
        filename_d = f'./data/processed/{dataset}/concepts/{"_".join(concept_type)}_{subset}_count.csv'
        with open(filename_d, 'w') as fid:
            fid.write('tag,count\n')
            for i in didemo_nouns.most_common():
                fid.write(f'{i[0]},{i[1]}\n')


def map_noun_to_videos(dataset, concept_type):
    '''
    DEPRECATED
    '''
    warnings.warn('DEPRECATED FUNCTION')
    # Count will be dumped here
    filename_d = f'data/processed/{dataset}/{"_".join(concept_type)}_to_videos.json'
    # Make sure you downloaded DiDeMo data and place it in
    # data/raw/{}_data.json

    nlp = spacy.load('en_core_web_sm')
    subsets = ['train', 'val', 'test']
    num_descriptions = 0
    didemo_nouns = Counter()
    videos = {}
    time = {}
    for subset in subsets:
        filename = f'./data/raw/didemo/{subset}_data.json'
        with open(filename) as f:
            data = json.load(f)
        for d in data:
            num_descriptions += 1
            d_i = d['description']
            doc_i = nlp(d_i)
            doc_i_nouns = Counter()
            for token in doc_i:
                if token.pos_ in concept_type:        
                    doc_i_nouns.update({token.lemma_: 1})
                    random.shuffle(d['times'])
                    time_i = d['times'][0]
                    time_i[0] *= 5
                    time_i[1] *= 5
                    time_i[1] += 5
                    if token.lemma_ in videos:
                        videos[token.lemma_].append(d['video'])
                        time[token.lemma_].append(time_i)
                    else:
                        videos[token.lemma_] = [d['video']]
                        time[token.lemma_] = [time_i]
            didemo_nouns.update(doc_i_nouns)
            
    with open(filename_d, 'x') as fid:
        json.dump({'nouns': didemo_nouns, 'videos': videos, 'time': time}, fid)


def histogram(dataset, concept_type):
    ''' 
    DEPRECATED
    '''
    warnings.warn('DEPRECATED FUNCTION')
    filename = f'./data/processed/{dataset}/{"_".join(concept_type)}_count.csv'
    f = open(filename, "r").readlines()
    data = np.asarray([f[i+1].strip('\n').split(',') for i in range(len(f)-1)])

    i, num_bins = 0, 0
    while int(data[i,1]) > int(data[0,1])/20.0:
        num_bins +=1
        i += 1

    print(num_bins)
    x = data[:num_bins,0]
    y = np.asarray(data[:num_bins,1], dtype=np.int32)
    plt.bar(x, y)
    plt.xticks(rotation='vertical')
    plt.savefig(f'./data/images/{concept_type}_frequency_{dataset}.pdf')


def create_json(dataset, concept_type):
    if dataset == 'didemo':
        subsets = ['train-01.json', 'val-01.json', 'test-01.json']
    elif dataset == 'charades-sta':
        subsets = ['train-01.json', 'val-02_01.json', 'test-01.json']
    elif dataset == 'activitynet-captions':
        subsets = ['train.json', 'val.json']


    for subset in subsets:
        print(f'\nLoading original data... ({subset})')
        filename = f'./data/processed/{dataset}/{subset}'
        original_data = json.load(open(filename, 'r'))
        
        print('Parsing original data...')
        nlp = spacy.load('en_core_web_sm')
        data = {'date': str(datetime.datetime.now()),
                'videos': original_data['videos'],
                'moments': [], # just an initialization, it is filled in the next section
                'git_hash': utils.get_git_revision_hash(),
                'responsible': 'Mattia'
        }
        print('Creating concept moments...')
        unique_concepts = {}
        list_of_concepts = []
        concept_id = 0
        for moment in tqdm.tqdm(original_data['moments']):
            for token in nlp(moment['description']):
                if token.pos_ in concept_type: 
                    concept = token.lemma_  #text
                    if concept not in list_of_concepts:
                        list_of_concepts.append(concept)
                        # data['moments'].append(create_new_concept_single(token,concept,moment,concept_id))       #treat each element as different and unique even if not
                        unique_concepts[concept] = create_new_concept(token,concept,moment,concept_id, dataset)
                        concept_id += 1
                    else:
                        ID = concept
                        unique_concepts[ID] = update_existing_concept(unique_concepts[ID], moment)
                        assert unique_concepts[ID]['description'] == ID

        data['moments'] = [unique_concepts[key] for key in list_of_concepts]
        assert len(data['moments'])==len(list(unique_concepts.keys()))

        print('Number of unique concept/moments: {}'.format(len(data['moments'])))

        filename_d = f'./data/processed/{dataset}/concepts/{"_".join(concept_type)}_{subset}'
        print(f'Dumping new json...({filename_d})')
        with open(filename_d, 'w') as fid:
            json.dump(data, fid)

        ordered_count = _ordered_count(data)
        subset = subset.split('-')[0]
        filename_d = f'./data/processed/{dataset}/concepts/{"_".join(concept_type)}_{subset}_count.csv'
        with open(filename_d, 'w') as fid:
            fid.write('tag,count\n')
            for i in ordered_count.most_common():
                fid.write(f'{i[0]},{i[1]}\n')


def create_new_concept(token,concept, moment, concept_id, dataset):
    new_concept = { 'description': concept,
                    'video' : [moment['video']],
                    'times' : [moment['times']],
                    'time'  : [moment['time']],
                    'annotation_id' : concept_id,
                    'counter': 1,
                    'concept_type': token.pos_
    }
    if dataset == 'didemo':
        new_concept['annotation_id_original'] = [moment['annotation_id_original']]

    return new_concept


def create_new_concept_single(token, concept, moment, concept_id, dataset):
    new_concept = { 'description': concept,
                    'video' : moment['video'],
                    'times' : moment['times'],
                    'time'  : moment['time'],
                    'annotation_id' : concept_id,
                    'counter': 1,
                    'concept_type': token.pos_
    }
    if dataset == 'didemo':
        new_concept['annotation_id_original'] = [moment['annotation_id_original']]

    return new_concept


def update_existing_concept(concept, moment):
    concept['video'].append(moment['video'])
    concept['times'].append(moment['times'])
    concept['time'].append(moment['time'])
    concept['counter'] += 1
    if dataset == 'didemo':
        concept['annotation_id_original'].append(moment['annotation_id_original'])

    return concept


def create_directory(dataset):
    path = f'./data/processed/{dataset}/concepts/'
    if not os.path.exists(path):
        os.makedirs(path)


def merge_outpud_didemo(dataset):
    directory = f'./data/processed/{dataset}/concepts/'
    # files = os.listdir(directory)
    # csv_files = [f for f in files if 'csv' in f]
    # 
    csv_files = ['NOUN_VERB_train_count.csv', 'NOUN_VERB_val_count.csv', 'NOUN_VERB_test_count.csv']
    didemo_nouns = {}
    df = pd.DataFrame() 
    for file_name in csv_files:
        data = pd.read_csv(f"{directory}{file_name}") 
        print('a')


def filter_top_50(dataset, concept_type):
    # subsets = ['train', 'val', 'test']
    if dataset == 'didemo':
        subsets = ['train-01.json', 'val-01.json', 'test-01.json']
    elif dataset == 'charades-sta':
        subsets = ['train-01.json', 'val-02_01.json', 'test-01.json']
    
    f = open(f'./data/processed/{dataset}/concepts/{"_".join(concept_type)}_top_50.txt', "r")
    top_50 = f.readlines()
    top_50 = [i.strip('\n') for i in top_50]

    for subset in subsets:
        print(f'\nLoading original data... ({"_".join(concept_type)}_{subset})')
        filename = f'./data/processed/{dataset}/concepts/{"_".join(concept_type)}_{subset}'
        data = json.load(open(filename, 'r'))
        top_data = {'date': str(datetime.datetime.now()),
                    'videos': data['videos'],
                    'moments': [], # just an initialization, it is filled in the next section
                    'git_hash': utils.get_git_revision_hash(),
                    'responsible': 'Mattia'
        }
        for elem in data['moments']:
            if elem['description'] in top_50:
                top_data['moments'].append(elem)

        assert(len(top_data["moments"])==50)
        subset = subset.split('.').split('_')[0]
        filename_d = f'./data/processed/{dataset}/concepts/{"_".join(concept_type)}_{subset}_top_50.json'
        print(f'Dumping new json...({filename_d})')
        with open(filename_d, 'w') as fid:
            json.dump(top_data, fid)        


def _ordered_count(data):
    count = Counter()
    for elem in data['moments']:
        count.update({elem['description']: len(elem['video'])})
    return count
    

def create_map(dataset, concept_type):
    # filename = f"{'_'.join(concept_type)}_top_50" 
    # if concept_type[0] == 'ALL':
    filename = 'list_of_concepts'
    f = open(f'./data/processed/{dataset}/concepts/{filename}.txt', "r")
    concepts = f.readlines()
    concepts = {c.strip('\n'):[] for c in concepts}
    if filename == 'list_of_concepts':
        filename = 'all_words'
    filename_d = f'./data/processed/{dataset}/concepts/concepts_map_{filename}.json'
    print(f'Dumping new json...({filename_d})')
    with open(filename_d, 'w') as fid:
        json.dump(concepts, fid)        
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset',
                    choices= ['didemo', 'charades-sta', 'activitynet-captions'],
                    help='Name of dataset to evaluate.')
    parser.add_argument('--concept-type', type=int, choices= [0,1,2,3],
                    help='Enable or disable reservation.')

    args = parser.parse_args()
    concept_type_list = [['NOUN', 'VERB'],['NOUN'], ['VERB'],['ALL']]
    concept_type = concept_type_list[args.concept_type]
    dataset = args.dataset               


    ########################### FUNCTIONS CALL 
    # create_directory(dataset=dataset)
    # count_noun_didemo(dataset=dataset, concept_type=concept_type)
    # histogram(dataset=dataset, concept_type=concept_type)
    # map_noun_to_videos(dataset=dataset, concept_type=concept_type)
    # create_json(dataset=dataset, concept_type=concept_type)
    # merge_outpud_didemo(dataset)
    #filter_top_50(dataset=dataset, concept_type=concept_type)
    create_map(dataset=dataset, concept_type=concept_type)
