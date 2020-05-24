### DIDEMO

## Retrain
dataset_dir=data/processed/didemo
interim=data/interim/didemo/CALChamfer/retrain/stal_clips
dir_1st_stage=data/interim/didemo/CALChamfer/retrain/smcn_40/4

parameters="--arch CALChamfer  
--train-list $dataset_dir/train-01.json 
--val-list   $dataset_dir/val-01.json 
--test-list  $dataset_dir/test-01.json 
--h5-path    $dataset_dir/resnet152_rgb_max_cl-2.5.h5 $dataset_dir/obj_predictions_perc_50_glove_bb_spatial.h5 
--min-length 3 
--momentum   0.9
--negative-sampling-iou 0.99 
--ground-truth-rate 1.0 
--nms-threshold 1.0 
--original-setup 
--proposals-in-train 
--stride 3 
--scales 2 3 4 5 6 
--feat rgb obj 
--proposal-interface DidemoICCV17SS 
--logfile     $interim/CALChamfer_tef_multistream_4a_retrain_${SLURM_ARRAY_TASK_ID} 
--h5-path-nis $dir_1st_stage/train/4_corpus-eval.h5" 

python train.py --gpu-id 0 $parameters 


## Approx Retrian
dataset_dir=data/processed/didemo
interim=data/interim/didemo/CALChamfer/retrain/stal_clips_approx
dir_1st_stage=data/interim/didemo/CALChamfer/retrain/smcn_40/4

parameters="--arch CALChamfer  
--train-list $dataset_dir/train-01.json 
--val-list   $dataset_dir/val-01.json 
--test-list  $dataset_dir/test-01.json 
--h5-path    $dataset_dir/resnet152_rgb_max_cl-2.5.h5 $dataset_dir/obj_predictions_perc_50_glove_bb_spatial.h5 
--min-length 3 
--momentum   0.9
--negative-sampling-iou 0.99 
--ground-truth-rate 1.0 
--nms-threshold 1.0 
--original-setup 
--proposals-in-train 
--stride 3 
--scales 2 3 4 5 6 
--feat rgb obj 
--proposal-interface DidemoICCV17SS 
--logfile     $interim/CALChamfer_tef_multistream_4a_retrain_approx_${SLURM_ARRAY_TASK_ID} 
--h5-path-nis $dir_1st_stage/approx/4_corpus-eval.h5 " 

python train.py --gpu-id 0 $parameters 


### CHARADES

## Retrain
dataset_dir=data/processed/charades-sta
interim=data/interim/charades-sta/CALChamfer/retrain/stal_clips
dir_1st_stage=data/interim/charades-sta/CALChamfer/retrain/smcn_42/3

parameters="--arch CALChamfer  
--train-list $dataset_dir/train-01.json 
--val-list   $dataset_dir/val-02_01.json 
--test-list  $dataset_dir/test-01.json 
--h5-path    $dataset_dir/resnet152_rgb_max_cl-3.h5 $dataset_dir/obj_predictions_perc_50_glove_bb_spatial.h5 
--ground-truth-rate 1.0
--min-length 3 
--momentum 0.95
--negative-sampling-iou 0.35 
--nms-threshold 0.6
--original-setup 
--proposals-in-train 
--scales 2 3 4 5 6 7 8 
--stride 0.3
--feat rgb obj
--proposal-interface SlidingWindowMSRSS 
--seed 1701 
--logfile $interim/CALChamfer_tef_multistream_5a_retrain_${SLURM_ARRAY_TASK_ID} 
--h5-path-nis $dir_1st_stage/train/3_corpus-eval.h5 " 

python train.py --gpu-id 0 $parameters 


## Approx Retrian
dataset_dir=data/processed/charades-sta
interim=data/interim/charades-sta/CALChamfer/retrain/stal_clips
dir_1st_stage=data/interim/charades-sta/CALChamfer/retrain/smcn_42/3

parameters="--arch CALChamfer  
--train-list $dataset_dir/train-01.json 
--val-list   $dataset_dir/val-02_01.json 
--test-list  $dataset_dir/test-01.json 
--h5-path    $dataset_dir/resnet152_rgb_max_cl-3.h5 $dataset_dir/obj_predictions_perc_50_glove_bb_spatial.h5 
--ground-truth-rate 1.0
--min-length 3 
--momentum 0.95
--negative-sampling-iou 0.35 
--nms-threshold 0.6
--original-setup 
--proposals-in-train 
--scales 2 3 4 5 6 7 8 
--stride 0.3
--feat rgb obj
--proposal-interface SlidingWindowMSRSS 
--seed 1701 
--logfile $interim/CALChamfer_tef_multistream_5a_retrain_approx_${SLURM_ARRAY_TASK_ID} 
--h5-path-nis $dir_1st_stage/approx/3_corpus-eval.h5 " 

python train.py --gpu-id 0 $parameters 