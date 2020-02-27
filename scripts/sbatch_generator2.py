TEMPLATE = """#!/bin/bash
#SBATCH -n 1 --exclude=kw60326,gpu-2040007,gpu-2040006
#SBATCH --job-name={job_name}
#SBATCH --cpus-per-task=9
#SBATCH --mem {ram}GB
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH -o /home/soldanm/projects/CVPR2020/moments-retrieval/log/%A-%a.out
#SBATCH --array=1-2

workdir=/home/soldanm/projects/CVPR2020/moments-retrieval/
cd $workdir

environment=moments-retrieval-devel
conda_root=$HOME/anaconda3

source $conda_root/etc/profile.d/conda.sh
conda activate $environment

dataset_dir="data/processed/{data_dir}"
data_split=" --train-list $dataset_dir/{train_split} --test-list $dataset_dir/{test_split} --h5-path $dataset_dir/{visual_features} "
bert_parameters=" --language-model bert --bert-name bert-{bert_size}-uncased --bert-feat-comb {comb_feat_bert} --bert-load-precomputed-features "
logfile=" --logfile ./data/interim/{data_dir}/language_ablation/logfiles/{model}_{loc_name}_bert_{bert_size}_feat_comb_{comb_feat_bert}_$SLURM_ARRAY_TASK_ID "
dataset_dependent_parameters=" {momentum} {negative_sampling_iou} {nms_threshold} {original_setup} {scales} {min_length} {proposals_in_train} {proposal_interface} "
parameters="--arch {model} {loc} --feat rgb $data_split $bert_parameters $logfile $dataset_dependent_parameters "

python train.py --gpu-id 0 $parameters"""

if __name__ == '__main__':
    counter = 0
    kwargs={}
    model_list     = ['MCN','SMCN']
    loc_list       = ['','--loc NONE']
    bert_size_list = ['base', 'large']
    comb_feat_bert_list = [0,1,2,3]
    data_dir_list  = ['didemo','charades-sta']

    for data_dir in data_dir_list:
        kwargs['data_dir'] = data_dir
        for model in model_list:
            kwargs['model'] = model
            if data_dir == 'didemo':
                kwargs['train_split'] = "train-01.json"
                kwargs['test_split'] = "val-01.json"
                if model == 'MCN':
                    kwargs['visual_features'] = "resnet152_rgb_max_cl-5.h5 "
                else:
                    kwargs['visual_features'] = "resnet152_rgb_max_cl-2.5.h5 "
                kwargs['momentum'] = '--momentum 0.9 '
                kwargs['negative_sampling_iou'] = '--negative-sampling-iou 0.99 '
                kwargs['nms_threshold'] = '--nms-threshold 1 '
                kwargs['original_setup'] = '--original-setup  '
                kwargs['scales'] = '--scales 2 3 4 5 6 '
                kwargs['min_length'] = '--min-length 3 '
                kwargs['proposals_in_train'] = '--proposals-in-train  '
                kwargs['proposal_interface'] = '--proposal-interface DidemoICCV17SS '

            elif data_dir == 'charades-sta':
                kwargs['train_split'] = "train-02_01.json "
                kwargs['test_split']  = "val-02_01.json "
                kwargs['visual_features'] = "resnet152_rgb_max_cl-3.h5 "
                kwargs['momentum'] = '--momentum 0.95 '
                kwargs['negative_sampling_iou'] = '--negative-sampling-iou 0.35 '
                kwargs['nms_threshold'] = '--nms-threshold 0.6 '
                kwargs['original_setup'] = '--original-setup  '
                kwargs['scales'] = '--scales 2 3 4 5 6 7 8  '
                kwargs['min_length'] = '--min-length 3 '
                kwargs['proposals_in_train'] = '--proposals-in-train  '
                kwargs['proposal_interface'] = '--proposal-interface SlidingWindowMSRSS '

            for loc in loc_list:
                kwargs['loc'] = loc
                if loc:
                    kwargs['loc_name'] = 'NOTEF'
                else:
                    kwargs['loc_name'] = 'TEF'
                for bert_size in bert_size_list:
                    kwargs['bert_size'] = bert_size
                    for comb_feat_bert in comb_feat_bert_list:
                        kwargs['ram'] = 60
                        if data_dir=='didemo' and comb_feat_bert==2:
                            kwargs['ram'] = 120
                        kwargs['comb_feat_bert'] = comb_feat_bert
                        counter += 1
                        kwargs['job_name']=counter
                        f = open(f"../sbatches/{counter}.sh", "w")
                        f.write(TEMPLATE.format(**kwargs))
                        f.close()