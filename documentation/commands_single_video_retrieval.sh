                                                 SINGLE VIDEO RETRIEVAL
BEST HYPER-PARAMETERS


#### DIDEMO
dataset_dir=data/processed/didemo

data_split=" --train-list $dataset_dir/train-01.json --test-list $dataset_dir/val-01.json --h5-path $dataset_dir/resnet152_rgb_max_cl-2.5.h5 "
logfile=" --dump-results --logfile ./data/interim/didemo/SMCN_TEF_ "
dataset_dependent_parameters=" --momentum 0.9 --negative-sampling-iou 0.99  --nms-threshold 1 --original-setup   --scales 2 3 4 5 6  --min-length 3  --proposals-in-train   --proposal-interface DidemoICCV17SS  "
parameters="--arch SMCN  $data_split $logfile $dataset_dependent_parameters "

parameters="

"


parameters="--arch SMCN --train-list $dataset_dir/train-01.json --test-list $dataset_dir/val-01.json --h5-path $dataset_dir/obj_predictions_perc_50_avg_glove.h5 --momentum 0.9 --negative-sampling-iou 0.99  --nms-threshold 1 --original-setup   --scales 2 3 4 5 6  --min-length 3  --proposals-in-train --nms-threshold 1.0  --proposal-interface DidemoICCV17SS  --loc NONE --no-context " 

python train.py --gpu-id 0 $parameters


#### CHARADES-STA

dataset_dir=data/processed/charades-sta

data_split=" --train-list $dataset_dir/tran-02_01.json  --test-list $dataset_dir/val-02_01.json  --h5-path $dataset_dir/resnet152_rgb_max_cl-3.h5  --feat rgb "

logfile=" --dump-results --logfile ./data/interim/charades-sta/SMCN_TEF_ "

dataset_dependent_parameters=" --momentum 0.95  --negative-sampling-iou 0.35 --nms-threshold 0.6  --original-setup  --scales 2 3 4 5 6 7 8  --min-length 3  --proposals-in-train   --proposal-interface SlidingWindowMSRSS  "

parameters="--arch SMCN  $data_split $logfile $dataset_dependent_parameters "

python train.py --gpu-id 0 $parameters



#############################################################################################################################################################################################################################################################################################################################################################################################

Object detector features 

dataset_dir=data/processed/didemo

parameters="--arch SMCN --train-list $dataset_dir/train-01.json --test-list $dataset_dir/val-01.json --h5-path $dataset_dir/obj_predictions_perc_50_max.h5 --momentum 0.9 --negative-sampling-iou 0.99  --nms-threshold 1 --original-setup   --scales 2 3 4 5 6  --min-length 3  --proposals-in-train   --proposal-interface DidemoICCV17SS  --feat obj --no-context --loc NONE " 

python train.py --gpu-id 0 $parameters

#############################################################################################################################################################################################################################################################################################################################################################################################

Single video convex combination 

dataset_dir=data/processed/didemo

parameters="--arch LateFusion --train-list $dataset_dir/train-01.json --test-list $dataset_dir/val-01.json --h5-path $dataset_dir/resnet152_rgb_max_cl-2.5.h5  $dataset_dir/obj_predictions_perc_50_avg_glove.h5 --feat rgb obj --momentum 0.9 --negative-sampling-iou 0.99  --nms-threshold 1 --original-setup   --scales 2 3 4 5 6  --min-length 3  --proposals-in-train --nms-threshold 1.0  --proposal-interface DidemoICCV17SS --evaluate --snapshot /home/soldanm/Documents/Projects/CVPR2020/moments-retrieval/data/interim/didemo/best_models/smcn_40_new_architecture.json"

python train.py --gpu-id 0 $parameters


#############################################################################################################################################################################################################################################################################################################################################################################################

CALChamfer

dataset_dir=data/processed/didemo
interim=./data/interim/matching_evaluation

parameters="
--arch CALChamfer  
--test-list $dataset_dir/test-01.json 
--h5-path   $dataset_dir/resnet152_rgb_max_cl-2.5.h5 
            $dataset_dir/obj_predictions_perc_50_glove_bb_spatial.h5 
--snapshot  $interim/model/CALChamfer_tef_glove_extra_5.json
--proposal-interface DidemoICCV17SS
--negative-sampling-iou 0.99 
--ground-truth-rate 1.0 
--num-workers-eval 1
--proposals-in-train 
--nms-threshold 1.0 
--scales 2 3 4 5 6 
--original-setup 
--feat rgb obj
--min-length 3 
--momentum 0.9
--stride 3 
--evaluate 
" 

python train.py --gpu-id 0 $parameters