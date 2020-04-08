import json
import spacy
import tqdm
import math
import numpy as np
import argparse


def get_moment_indices(segment, clip_length, len_predictions):
    start_idx = int(segment[0]/clip_length)         # get initial index of window
    end_idx   = min(int(math.ceil(segment[1]/clip_length)), len_predictions)# get final index of window
    if start_idx == end_idx: 
        end_idx += 1
    return start_idx, end_idx
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--th-list', required=True, nargs='+',
                    help='list of thresholds to apply.')
    args = parser.parse_args()
    th_list = args.th_list
    datasets = ['didemo'] #['didemo','charades-sta']
    obj_dataset = 'visual_genome'  #['coco', 'visual_genome']
    clip_length = 5.0
    cnt=0
    for dataset in datasets:
        print(f'Dataset: {dataset}')
        splits = ['val-01'] # ['test-01', 'val-01','train-01']
        if dataset == 'charades-sta':
            splits = ['test-01', 'val-02_01','train-01']
        for split in splits:
            print(f'Split: {split}')
            for th in th_list:
                th = float(th)
                #ANNOTATIONS
                annotations = json.load(open(f'./processed/{dataset}/{split}.json','r'))
                video_indice_to_id = list(annotations['videos'].keys()) #get positional idx of videos indices
                video_id_to_indice = {k:i for i,k in enumerate(video_indice_to_id)}

                #OBJ DETECTOR
                filename = None
                if obj_dataset == 'coco':
                    filename = f'didemo_obj_detection_th_{th}.json'
                else:
                    filename = f'didemo_obj_detection_obj_th_{th}.json'
                if dataset == 'charades-sta':
                    clip_length = 3.0
                    if obj_dataset == 'coco':
                        filename = f'charades_sta_obj_detection_th_{th}.json'
                    else:
                        filename = f'charades_sta_obj_detection_obj_th_{th}.json'
                obj_predictions = json.load(open(f'./processed/{dataset}/obj_detection/{obj_dataset}/{filename}','r'))
                #FILTERING OF OBJ PREDICTIONS GIVEN THE SPLIT:
                obj_predictions = {k:obj_predictions[k] for k in video_indice_to_id}

                # OBJ CLASSES
                word_map = json.load(open(f'./raw/language/{obj_dataset}/concepts_map_to_{obj_dataset}_classes.json','r'))
                mapping_list = list(word_map.keys())
                for mapping_word in tqdm.tqdm(mapping_list):
                    map_keys = [mapping_word]
                    classes_to_push = list(set([word_map[k] for k in map_keys]))
                    nlp = spacy.load('en_core_web_sm')

                    #CORPUS DATA
                    filename = dataset.replace('-','_')
                    split_name = split.split('-')[0]
                    corpus_data = json.load(open(f'./{filename}_{split_name}_corpus_information.json','r'))
                    proposals   = corpus_data['proposals']
                    video_indices = corpus_data['video_indices']

                    # CREATING INFORMATION PER PROPOSALS GIVEN THE OBJ DETECTOR
                    # print('Indexing object detection results...')
                    moments_obj_predictions = [None]*len(proposals)
                    for i, segment in enumerate(proposals):
                        video_indice    = video_indices[i]
                        video_name      = video_indice_to_id[video_indice]
                        video_obj_prediction = obj_predictions[video_name]
                        len_predictions = int(list(video_obj_prediction.keys())[-1])
                        start_idx, end_idx = get_moment_indices(segment, clip_length, len_predictions)
                        # determine the predicted classes in the moment
                        detected_obj_class_in_moment = []
                        for ii in range(start_idx,end_idx):
                            detected_obj_class_in_moment.extend(video_obj_prediction[str(ii)])
                        moments_obj_predictions[i] = list(set(detected_obj_class_in_moment))    #remove duplicates and store

                    # CREATING INFORMATION PER PROPOSALS GIVEN THE LANGUAGE ANNOTATIONS
                    # print('Indexing concepts...')
                    starting_index_of_video_in_proposals = {'0':0}
                    current = 0
                    for i, idx in enumerate(video_indices):
                        if idx != current:
                            starting_index_of_video_in_proposals[str(idx)] = i
                            current = idx
                    starting_index_of_video_in_proposals[str(len(video_indice_to_id))] = len(proposals)

                    moments_concept_classes = [[]]*len(proposals)
                    for moment in annotations['moments']:
                        tokens          = nlp(moment['description'])                            # tokens of the descriptions
                        concepts        = [t.lemma_ for t in tokens if t.pos_ in ['NOUN']]      # nouns in the description
                        concept_classes = [word_map[t.lemma_] for t in tokens if t.lemma_ in map_keys]  # coco classes id of the nouns 
                        video_indice    = video_id_to_indice[moment['video']]                   # get the numerical id of the video
                        segments        = moment['times']                                       # extract annotation segments

                        # get unique segments out of the moment annotation (speed up process)
                        unique_segments = []
                        for segment in segments:
                            if segment not in unique_segments:
                                unique_segments.append(segment)

                        # get proposals for specific video:
                        start_idx = starting_index_of_video_in_proposals[str(video_indice)]     # given the video id get the starting point in the proposals
                        end_idx   = starting_index_of_video_in_proposals[str(video_indice + 1)] # get the end point
                        proposals_for_video = proposals[start_idx:end_idx]                      # get the total proposals for the specific video

                        # get offset of current proposals to the beginning of the video and 
                        # use it to append the concept information in the right place
                        for segment in unique_segments:                                         # for each annotated segment
                            segment_indice = None
                            if dataset == 'didemo':
                                segment_indice = proposals_for_video.index(segment)                 # get the offset index of the segment in the proposals list with respect to the beginning of the video proposals 
                            if dataset == 'charades-sta':
                                # we need to compute the iou with all the proposals and select the maximum
                                intersection = [min(segment[1], p[1])-max(segment[0], p[0]) for p in proposals_for_video]
                                intersection = [0 if i < 0 else i for i in intersection]

                                union        = [max(segment[1], p[1])-min(segment[0], p[0]) for p in proposals_for_video]
                                iou = list(np.asarray(intersection)/np.asarray(union))
                                segment_indice = iou.index(max(iou))
                            moments_concept_classes[start_idx + segment_indice] = concept_classes    # use the two indices to append the concept classes for the specific segment position

                    #INITIALIZATION
                    true_positive, gt_hit, prediction_hit = 0,0,0
                    precision, recall = 0,0

                    # true positive  = in the description there is a person related word and in the obj detection as well
                    # False positive = in the description there is no reference to person but the prediction for person fires
                    # False negative = the description says there should be a person but the obj predictor do not fire
                    
                    # PERFORMANCE COMPUTATION
                    for i in range(len(proposals)):
                        concep_hit = any(item in classes_to_push for item in moments_concept_classes[i])
                        obj_hit    = any(item in classes_to_push for item in moments_obj_predictions[i])
                        double_hit = any(item in moments_concept_classes[i] for item in moments_obj_predictions[i])

                        if double_hit:
                            true_positive += 1
                        if concep_hit:
                            gt_hit += 1
                        if obj_hit:
                            prediction_hit += 1
                    
                    if prediction_hit > 0:
                        precision = true_positive / prediction_hit
                    if gt_hit > 0:
                        recall = true_positive / gt_hit

                    dump_dict = {'dataset':dataset, 'split':split, 'obj_detector': obj_dataset, 'threshold':th,
                                'concept': [map_keys[0],classes_to_push[0]],'precision': precision, 'recall': recall}
                    
                    with open(f'./interim/analisys_obj_detector/{cnt}_{th}.json','w') as file:
                        json.dump(dump_dict,file)
                    cnt+=1

                    # print(f'Threshold {th}')
                    # print('Precision: {:.4f}'.format(precision))
                    # print('Recall: {:.4f}'.format(recall))