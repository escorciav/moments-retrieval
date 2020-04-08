import json

datasets = ['didemo','charades-sta']
for dataset in datasets:
    name = dataset.replace('-','_')
    data = json.load(open(f'./processed/{dataset}/obj_detection/{name}_obj_detection_th_0.5.json','r'))

    keys = list(data.keys())

    coco_classes = []
    with open('./raw/ms_coco_classnames_read.txt','r') as f:
        coco_classes = f.readlines()
    coco_classes = [c.strip('\n') for c in coco_classes]

    objects_list = {i:0 for i in range(80)}
    for k in keys:
        v_keys = data[k]
        for kk in v_keys:
            predictions = data[k][kk]
            for obj in predictions:
                objects_list[obj] +=1
    
    with open(f'./{name}_prediction_numbers_per_class.txt','w') as f:
        for k in objects_list.keys():
            f.write(f'{coco_classes[int(k)]},{objects_list[k]}\n')