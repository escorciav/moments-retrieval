import json
import os
import sys
import numpy as np
import time
import math


def read_VG_data():
    classes_VG = []
    classes_file = '../data/raw/language/visual_genome/objects_vocab.txt'
    with open(classes_file, 'r') as f:
        for object in f.readlines():
            classes_VG.append(object.split(',')[0].lower().strip())
    return classes_VG


def polish_VG_data(classes_VG, lang_interface):
    classes_VG_revisited = []
    map_glove_word = []
    for c in classes_VG:
        v = lang_interface(c)
        if np.abs(v).sum() != 0.0:
            map_glove_word.append(v)
            classes_VG_revisited.append(c)   
    return classes_VG_revisited, map_glove_word


def load_from_annotations():
    filename = '../data/processed/didemo/test-01.json'
    data     = json.load(open(f'{filename}','r'))
    moments  = data['moments']
    videos   = list(set([m['video'] for m in moments]))
    return moments, videos


def read_dumped_metadata():
    path = '../data/interim/matching_evaluation/dump_done/eval_items/'
    files = sorted(os.listdir(path))[1:]
    dumped_data_input = {}
    for f in files:
        d   = json.load(open(f'{path}{f}','r'))
        idx = list(d.keys())[0]
        # Convert back to numpy
        d[idx]['times']     = np.asarray(d[idx]['times'])
        d[idx]['proposals'] = np.asarray(d[idx]['proposals'])
        d[idx]['feat']      = {k:np.asarray(v) for k,v in d[idx]['feat'].items()}
        # Store in variable for later use - aggregate all information
        dumped_data_input[int(idx)] = d[idx]
    return dumped_data_input

def read_obj_detections():
    obj_file = '../data/processed/didemo/obj_detection/visual_genome/didemo_obj_detection_perc_50_with_scores.json'
    return json.load(open(obj_file,'r'))


def read_dumped_distances():
    path  = '../data/interim/matching_evaluation/dump_done/chamfer_distance/'
    files = sorted(os.listdir(path))
    idx   = 0
    dumped_data_chamfer = {}
    for i in range(0,len(files),2):
        d_rgb = np.load(f'{path}{i:04d}.npz')
        d_obj = np.load(f'{path}{i+1:04d}.npz')
        dumped_data_chamfer[idx]= {'rgb': d_rgb['arr_0'], 
                                   'obj': d_obj['arr_0']}    
        idx += 1
    return dumped_data_chamfer


def read_mapping_between_clips_and_obj():
    input_file = '../data/processed/didemo/obj_classes_per_clip.json'
    mapping_obj_with_boxes = json.load(open(input_file,'r'))

    keys_ = list(mapping_obj_with_boxes.keys())
    mapping_obj = {}
    for k in keys_:
        clip_keys_ = list(mapping_obj_with_boxes[k].keys())
        clip_dict_ = {}
        for ck in clip_keys_:
            clip_list_ = mapping_obj_with_boxes[k][ck]
            reduced_list = [elem[0] for elem in clip_list_]
            clip_dict_[ck] = reduced_list
        mapping_obj[k] = clip_dict_
    
    return mapping_obj



def _get_frames_indices(video_id, clip_size):
    FPS  = 5
    path = '../data/interim/matching_evaluation/didemo_frames/{}/'.format(video_id)
    frame_keys  = sorted(os.listdir(path))            # get list of frames for video     
    num_frames  = len(frame_keys)                     # compute number of frames
    num_windows = math.ceil(num_frames/clip_size)     # compute number of clips
    idx   = [i for i in range(num_frames)]            # ancillary indexes variable
    start = math.ceil(FPS * clip_size/2)-1            # stard idx, middle of clip
    step  = math.ceil(FPS * clip_size)                # step, clips size in frames
    selected_idx    = idx[start::step]                # index of interesting frames
#     selected_idx = sorted([1] + idx[3::int(clip_size*2)] + idx[6::int(clip_size*2)])   # Select best indexes 
    selected_frames = [frame_keys[i] for i in selected_idx]   # distill the wanted frames
    selected_frames = ['{}{}'.format(path,f) for f in selected_frames]
    return selected_frames


def _get_frames_per_proposal(frames, p, clip_size):
    start = int(p[0]/clip_size)
    end   = int(p[1]/clip_size)
    indexes = list(np.arange(start,end))
    return [frames[i] for i in indexes]


def _inverse_map_box(boxes, height, width):
    new_boxes = []
    for box in boxes:
        c1,c2,w,h = box
        new_boxes.append([(c1-w)*width,(c2-h)*height,(c1+w)*width,(c2+h)*height ])
    return new_boxes


def extract_obj_names_and_positions_per_clip(data_obj, prop, clip_size, classes_VG, merge):
    # Determine relevant 
    start = int(prop[0]/clip_size)
    end   = int(prop[1]/clip_size)
    indexes = list(np.arange(start,end))
    positions, names, scores = {},{},{}
    
    for i in indexes:
        clip_data    = data_obj[f'{i}']
        scores[i]    = [d[1] for d in clip_data]
        positions[i] = [d[2] for d in clip_data]
        names[i]     = [classes_VG[d[0]-1] for d in clip_data]
        
    if merge:
        # merge clips of 2.5 seconds to get data in 5s        
        cnt=0
        new_names = {i:[] for i in range(len(indexes[::2]))}
        new_pos   = {i:[] for i in range(len(indexes[::2]))}
        for i in indexes[::2]:
            names_ = list(set(names[i]+names[i+1]))
            name_dict = {n:{'scor':[],'pos':[]} for n in names_}
            for n in names_:
                if n in names[i]:
                    idx = names[i].index(n)
                    name_dict[n]['scor'].append(scores[i][idx])
                    name_dict[n]['pos'].append(positions[i][idx])
                if n in names[i+1]:
                    idx = names[i+1].index(n)
                    name_dict[n]['scor'].append(scores[i+1][idx])
                    name_dict[n]['pos'].append(positions[i+1][idx])
            #Reduce copies:
            for n in names_:
                new_names[cnt].append(n)
                if len(name_dict[n]['scor']) > 1:
                    inx = np.argmax(np.asarray(name_dict[n]['scor']))
                    new_pos[cnt].append(name_dict[n]['pos'][inx])
                else:
                    new_pos[cnt].append(name_dict[n]['pos'])
            cnt+=1
        return {'positions':new_pos, 'names':new_names}
    return {'positions':positions, 'names':names}


def _plot_proposals(video_id,description, proposals_frames, proposals_objects, proposals_mapping, prop_idx):
    path_dump = '../data/interim/matching_evaluation/images/'
    folder    = '{}{}/'.format(path_dump,video_id)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    names  = proposals_objects['names']
    names_keys = list(names.keys())
    bboxes = proposals_objects['positions']
    bboxes_keys = list(bboxes.keys())

    my_dpi=1
    fig, ax = plt.subplots(nrows=1, ncols=len(proposals_frames))
#     DPI = fig.get_dpi()
#     fig.set_size_inches(len(frames)*320/float(DPI),2*240.0/float(DPI))# figsize=(len(frames)*3,len(frames)/2))
    plt.suptitle(description, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=0.0, right=None, top=None, wspace=0.01, hspace=None)
    
    for i, frame in enumerate(proposals_frames):
        im = cv2.imread(frame)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        height, width, _ = im.shape
        
        obj_names_per_frame = names[names_keys[i]]
        obj_loc_per_frame   = _inverse_map_box(bboxes[bboxes_keys[i]], height, width)
        
        ax[i].imshow(im)
        ax[i].axis('off')
        
        for n,bbox in zip(obj_names_per_frame,obj_loc_per_frame):
            ax[i].add_patch(patches.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='red', linewidth=2, alpha=0.5))
            ax[i].text(bbox[0], bbox[1] - 2,
                            '%s' % (n),
                            bbox=dict(facecolor='blue', alpha=0.25),
                            fontsize=10, color='white')
        
    f_name = frame.split('/')[-1]
    dump_path = '{}{}'.format(folder,prop_idx)
    plt.savefig(dump_path,bbox_inches='tight')
    plt.close()