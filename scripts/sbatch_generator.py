import os
import argparse

TEMPLATE = """#!/bin/sh
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name={job_name}
#SBATCH --chdir={workdir}
##SBATCH --workdir={workdir}
#SBATCH --output=log/%x_%A-%a.out
#SBATCH --array=1-1
##SBATCH --constraint="[tesla_k40m|gtx1080ti|p100|p6000|v100|gtx2080ti]"
##SBATCH --account=ivul
##SBATCH --partition=debug
#SBATCH --qos=ivul
{server}
# Usage: bash [filename-here]
###### Note: experiment parameters ######
data_dir="{data_dir}"
MODEL_ARCH="{arch}"
H5_PATH="{h5_path}"
#CUE_LABEL="{cue}"
PROPOSAL_INTERFACE="{proposal_interface}"
MOMENT_LOC_FEAT="{tef_or_other}"
TRAIN_LIST="{train_list}"
VAL_LIST="{val_list}"
TEST_LIST="{test_list}"
HPS="{hyperparm}"
NMS="{nms_threshold}"
SETUP="{original_setup}"
PROP="{proposals_in_train}"
SAMP="{negative_sampling_iou}"
WORKERS="{num_workers_eval}"
FEAT="{feat}"
STRIDE="{stride}"
MIN_LENGTH="{length}"
SCALES="{scales}"
BI_LSTM="{bi_lstm}"
LR="{lr}"
LR_DECAY="{lr_decay}"
LR_STEP="{lr_step}"
EPOCHS="{epochs}"
LANG_HIDDEN="{lang_hidden}"
UNIT_VECTOR="{unit_vector}"
EV_EP="{eval_on_epoch}"
CT="{consec_train}"
DTV="{disable_tef_val}"
DEBUG="{debug}"
FEAT="{feat}"
N_DISPLAY="{n_display}"
MOMENTUM="{momentum}"
MIN_EPOCHS="{min_epochs}"
CLIP_LEN="{clip_length}"
OPTIMIZER="{optimizer}"
GT_RATE="{gt_rate}"
EXTRAS="$HPS $MODEL_ARCH  $GT_RATE $OPTIMIZER $CT $N_DISPLAY $CLIP_LEN $MOMENTUM $DEBUG $MIN_EPOCHS $FEAT $DTV $EV_EP $DROP_TEF $PROPOSAL_INTERFACE $EVAL $LANG_HIDDEN $UNIT_VECTOR $LDROP $LR $LR_DECAY $LR_STEP $EPOCHS $NMS $SETUP $PROP $SAMP $WORKERS $STRIDE $MIN_LENGTH $SCALES $BI_LSTM $FEAT $TRAIN_LIST $VAL_LIST $TEST_LIST $H5_PATH $MOMENT_LOC_FEAT "
###### Device: use -1 for cpu ######
gpu_device=0
###### Note: comment if wanna print logs and dont dump anything ######
output_dir="{output_dir}"
###### Note: run training sessions sequentially; assising a integer-id for each run ######
repetitions=1
offset=1
###### conda environment variables ######
# Note: comment conda_root if you don't know it. Update it if you install anaconda at an unusual place.
environment=moments-retrieval-devel2
conda_root=$HOME/anaconda3
###### git-friendly ######
# avoid tracking and displaying logs/data in git
dont_track_this="*.log\\n*.json\\n*.csv\\n*.h5\\n*.tar"
# Note: uncomment to run code at commit-hash
# commit_reference=c9e604fab8fdffaa2e920587cab97fbb311179f2

# Forget about the following lines
# Parametrize offset and repetitions with task-id
if [ ! -z ${{SLURM_ARRAY_TASK_ID+x}} ]; then
offset=$SLURM_ARRAY_TASK_ID
fi
let "repetitions=$repetitions + $offset"
# Fackung slurm! could find a nice way to try-catch with slurm+bash
if [ ! -z $conda_root ]; then
source $conda_root/etc/profile.d/conda.sh
fi
# zero-tolerance policy with errors from this point ongoing
set -e
#hostname
[ ! -z $SLURM_ARRAY_TASK_ID ] && echo Job-ID: $SLURM_ARRAY_JOB_ID'_'$SLURM_ARRAY_TASK_ID
#echo number of cores $(python -c "import multiprocessing; print(multiprocessing.cpu_count())")
#nvidia-smi
# go to a given point if history
[ ! -z $commit_reference ] && git checkout $commit_reference
conda activate $environment
# uncoment next line for debugging
# set -x

[ ! -z $output_dir ] && mkdir -p $output_dir
for ((i=$offset; i<$repetitions; i++)); do
ACTUAL_EXTRAS=$EXTRAS
if [ ! -z $output_dir ]; then
    # skip it when experiment-id was consumed before
    [ -f $output_dir/$i".json" ] && continue
    # add gitignore to dont crowd git-repo
    [ ! -f $output_dir/.gitignore ] && printf $dont_track_this >> $output_dir/.gitignore
    # augment EXTRAS with output_dir
    ACTUAL_EXTRAS="$EXTRAS --logfile $output_dir"
fi
# finaly the stuff that we care about
python train.py --gpu-id $gpu_device $ACTUAL_EXTRAS;
done
echo successfully done
"""

def setup_kwargs(**params):
    kwargs = {
    ### Simulation data
    "job_name"              : "{}".format(params["name"]),
    "workdir"               : "/home/soldanm/projects/CVPR2020/moments-retrieval/",
    "output_dir"            : "{}".format(params["output_dir"]),

    ### Model data    
    "data_dir"              : "{} ".format(params["dataset_dir"]),
    "arch"                  : "--arch {} ".format(params["model"]),
    "proposal_interface"    : "--proposal-interface {} ".format(params["interface"]),
    "nms_threshold"         : "--nms-threshold {} ".format(params["nms_threshold"]),        
    "original_setup"        : "--original-setup ",
    "proposals_in_train"    : "--proposals-in-train ", 
    "negative_sampling_iou" : "--negative-sampling-iou {} ".format(params["neg_samp_iou"]),  
    "num_workers_eval"      : "{} ".format(params["num_workers_eval"]),
    "stride"                : "--stride {} ".format(params["stride"]),     
    "length"                : "--min-length {} ".format(params["min_length"]), 
    "scales"                : "--scales {} ".format(params["scales"]), 
    "feat"                  : "{} ".format(params["feat"]),
    "train_list"            : "{} ".format(params["train"]),
    "val_list"              : "{} ".format(params["val"]),               # TO BE FIXES FOR FINAL EXPERIMENTS
    "test_list"             : "{} ".format(params["test"]),
    "h5_path"               : "--h5-path {}/{} ".format(params["dataset_dir"],params["h5_file"]),
    "tef_or_other"          : "{}".format(params["loc"]), 
    "bi_lstm"               : "{} ".format(params["bi_lstm"]),
    "cue"                   : "",
    "hyperparm"             : "{}".format(params["hps"]),
    "lr"                    : "{}".format(params["lr"]),
    "lr_decay"              : "{}".format(params["lr_decay"]),
    "lr_step"               : "{}".format(params["lr_step"]),
    "epochs"                : "{}".format(params["epochs"]),
    "lang_hidden"           : "{}".format(params["lang_hidden"]),
    "unit_vector"           : "{}".format(params["unit_vector"]),
    "eval_on_epoch"         : "{}".format(params["eval_on_epoch"]),
    "server"                : "{}".format(params["server"]),
    "consec_train"          : "{}".format(params["consec_train"]),
    "disable_tef_val"       : "{}".format(params["disable_tef_val"]),    
    "debug"                 : "{}".format(params["debug"]),
    "min_epochs"            : "{}".format(params["min_epochs"]),           
    "momentum"              : "{}".format(params["momentum"]),
    "n_display"             : "{}".format(params["n_display"]),
    "clip_length"           : "{}".format(params["clip_length"]),
    "optimizer"             : "{}".format(params["optimizer"]),
    "gt_rate"               : "{}".format(params["gt_rate"])
    }
    return kwargs


def get_param_per_dataset(dataset,dataset_dir,model,mode,server):
    names, h5_files, locs = None, None, None

    ##### EXPERIMENT DEPENDANT PARAMETERS
    d = {} 
    
    if server == "skynet":
        d["server"]= "##SBATCH --reservation=IVUL"
    elif server == "ibex":
        d["server"]= "#SBATCH --reservation=IVUL"
    else:
        raise Exception("Server not known. Does we have a reservation?")

    # General
    d["model"] = model
    d["dataset_dir"] = dataset_dir

    # Learning
    d["lr"] = "--lr 0.05"                   #"--lr 0.05"                       # pass nothing --> use default, otherwise pass --lr value
    d["lr_decay"] = "--lr-decay 0.1"        #"--lr-decay 0.001"
    d["lr_step"] = "--lr-step 30"           #"--lr-step 50"
    d["epochs"] = "--epochs 108"            #"--epochs 108"
    d["hps"] = ""                           # set to "" to disables, to enable "--hps "
    d["eval_on_epoch"] ="--eval-on-epoch -1"
    d["optimizer"] = ""
    d['gt_rate_list'] = [0.1,0.3,0.5,0.7]        
    d['gt_rate'] = "--ground-truth-rate 1.0"
    
    # Architecture
    d["bi_lstm"] = ""                       # set to "" to disable and " --bi-lstm "
    d["lang_hidden"] = "--lang-hidden 1000"
                                            # d["list_dropout"] = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    d["unit_vector"]  = ""                  #"--unit-vector "

    # Experiment dependent
    d["consec_train"] = ""                  #"--consecutive-training"
    d["disable_tef_val"] = ""               #"--disable-tef-validation"
    d["debug"] = " "
    

    ##### NETWORK DEPENDANT PARAMETERS
    if model == "CALChamfer": 
        locs  = ["","","--loc NONE","--loc NONE"]       #it is only model dependant

        if dataset == "didemo":
            d["interface"] = "DidemoICCV17SS"
            h5_files = ["resnet152_rgb_max_cl-5.h5","resnet152_rgb_max_cl-2.5.h5",\
                            "resnet152_rgb_max_cl-5.h5","resnet152_rgb_max_cl-2.5.h5"]
            names = ["{}_TEF_5s_".format(model),"{}_TEF_2_5s_".format(model),\
                        "{}_NOTEF_5s_".format(model),"{}_NOTEF_2_5s_".format(model)]
        elif dataset == "charades-sta":
            # d["interface"] = "SlidingWindowMSRSS"
            # h5_files = ["resnet152_rgb_max_cl-3.h5","resnet152_rgb_max_cl-1.5.h5",\
            #                 "resnet152_rgb_max_cl-3.h5","resnet152_rgb_max_cl-1.5.h5"]
            # names = ["{}_TEF_3s_".format(model),"{}_TEF_1_5s_".format(model),\
            #             "{}_NOTEF_3s_".format(model),"{}_NOTEF_1_5s_".format(model)]
            h5_files = ["resnet152_rgb_max_cl-3.h5", "resnet152_rgb_max_cl-3.h5"]
            names = ["{}_TEF_3s_".format(model), "{}_NOTEF_3s_".format(model)]
        elif dataset == "activitynet-captions":
            d["interface"] = "SlidingWindowMSRSS"
            h5_files = ["resnet152_rgb_max_cl-5.h5","resnet152_rgb_max_cl-2.5.h5",\
                            "resnet152_rgb_max_cl-5.h5","resnet152_rgb_max_cl-2.5.h5"]
            names = ["{}_TEF_5s_".format(model),"{}_TEF_2_5s_".format(model),\
                        "{}_NOTEF_5s_".format(model),"{}_NOTEF_2_5s_".format(model)]
    
    else:
        if model == "SMCN":
            if dataset == "didemo":
                locs  = ["","--loc NONE"]       # model dependant
                d["interface"] = "DidemoICCV17SS"
                h5_files = ["resnet152_rgb_max_cl-2.5.h5","resnet152_rgb_max_cl-2.5.h5"]
                names = ["{}_TEF_2_5s_".format(model),"{}_NOTEF_2_5s_".format(model)]
            elif dataset == "charades-sta":
                # locs  = ["","","--loc NONE","--loc NONE"]
                # d["interface"] = "SlidingWindowMSRSS"
                # h5_files = ["resnet152_rgb_max_cl-3.h5","resnet152_rgb_max_cl-1.5.h5",\
                #             "resnet152_rgb_max_cl-3.h5","resnet152_rgb_max_cl-1.5.h5"]
                # names = ["{}_TEF_3s_".format(model),"{}_TEF_1_5s_".format(model),\
                #         "{}_NOTEF_3s_".format(model),"{}_NOTEF_1_5s_".format(model)]
                locs  = ["","--loc NONE"]
                d["interface"] = "SlidingWindowMSRSS"
                h5_files = ["resnet152_rgb_max_cl-3.h5", "resnet152_rgb_max_cl-3.h5"]
                names = ["{}_TEF_3s_".format(model), "{}_NOTEF_3s_".format(model)]
            elif dataset == "activitynet-captions":
                locs  = ["","","--loc NONE","--loc NONE"]
                d["interface"] = "SlidingWindowMSRSS"
                h5_files = ["resnet152_rgb_max_cl-5.h5","resnet152_rgb_max_cl-2.5.h5",\
                            "resnet152_rgb_max_cl-5.h5","resnet152_rgb_max_cl-2.5.h5"]
                names = ["{}_TEF_5s_".format(model),"{}_TEF_2_5s_".format(model),\
                        "{}_NOTEF_5s_".format(model),"{}_NOTEF_2_5s_".format(model)]

        elif model == "MCN":
            if dataset == "didemo":
                # locs  = ["","","--loc NONE","--loc NONE"]
                # d["interface"] = "DidemoICCV17SS"
                # h5_files = ["resnet152_rgb_max_cl-5.h5","resnet152_rgb_max_cl-2.5.h5",\
                #             "resnet152_rgb_max_cl-5.h5","resnet152_rgb_max_cl-2.5.h5"]
                # names = ["{}_TEF_5s_".format(model),"{}_TEF_2_5s_".format(model),\
                #         "{}_NOTEF_5s_".format(model),"{}_NOTEF_2_5s_".format(model)]
                locs  = ["","--loc NONE",]
                d["interface"] = "DidemoICCV17SS"
                h5_files = ["resnet152_rgb_max_cl-5.h5","resnet152_rgb_max_cl-5.h5"]
                names = ["{}_TEF_5s_".format(model),"{}_NOTEF_5s_".format(model)]
            elif dataset == "charades-sta":
                locs  = ["","--loc NONE"]
                # d["interface"] = "SlidingWindowMSRSS"
                # h5_files = ["resnet152_rgb_max_cl-3.h5","resnet152_rgb_max_cl-1.5.h5",\
                #             "resnet152_rgb_max_cl-3.h5","resnet152_rgb_max_cl-1.5.h5"]
                # names = ["{}_TEF_3s_".format(model),"{}_TEF_1_5s_".format(model),\
                #         "{}_NOTEF_3s_".format(model),"{}_NOTEF_1_5s_".format(model)]
                d["interface"] = "SlidingWindowMSRSS"
                h5_files = ["resnet152_rgb_max_cl-3.h5", "resnet152_rgb_max_cl-3.h5",]
                names = ["{}_TEF_3s_".format(model), "{}_NOTEF_3s_".format(model),]
            elif dataset == "activitynet-captions":
                locs  = ["","","--loc NONE","--loc NONE"]
                d["interface"] = "SlidingWindowMSRSS"
                h5_files = ["resnet152_rgb_max_cl-5.h5","resnet152_rgb_max_cl-2.5.h5",\
                            "resnet152_rgb_max_cl-5.h5","resnet152_rgb_max_cl-2.5.h5"]
                names = ["{}_TEF_5s_".format(model),"{}_TEF_2_5s_".format(model),\
                        "{}_NOTEF_5s_".format(model),"{}_NOTEF_2_5s_".format(model)]
        else:
            raise Exception("This model does not have predefined parameters.")

    ##### MODE DEPENDANT PARAMETERS
    if mode == "test":
        if dataset == "didemo":
            d["train"] = "--train-list {}/train-01.json".format(dataset_dir)
            d["val"]   = ""#"--val-list {}/val-01json".format(dataset_dir) 
            d["test"]  = "--test-list {}/test-01.json".format(dataset_dir) 
        elif dataset == "charades-sta":
            d["train"] = "--train-list {}/train-01.json".format(dataset_dir)
            d["val"]   = "--val-list {}/val-01.json".format(dataset_dir)       
            d["test"]  = "--test-list {}/test-01.json".format(dataset_dir)             
        elif dataset == "activitynet-captions":
            d["train"] = "--train-list {}/train.json".format(dataset_dir)
            d["val"]   = ""
            d["test"]  = "--test-list {}/val.json".format(dataset_dir)
    elif mode == "val":
        if dataset == "didemo":
            d["train"] = "--train-list {}/train-01.json".format(dataset_dir)
            d["val"]   = "--val-list {}/val-01.json".format(dataset_dir)
            d["test"]  = "--test-list {}/train-01_01.json".format(dataset_dir)
        elif dataset == "charades-sta":
            d["train"] = "--train-list {}/train-02_01.json".format(dataset_dir)
            d["val"]   = ""
            d["test"]  = "--test-list {}/val-02_01.json".format(dataset_dir)
        elif dataset == "activitynet-captions":
            d["train"] = "--train-list {}/train-01.json".format(dataset_dir)
            d["val"]   = "--val-list {}/val-01.json".format(dataset_dir)
            d["test"]  = "--test-list {}/train-01_01.json".format(dataset_dir)
    else:
        raise Exception("Mode not implemented yet!")

    ##### DATASET DEPENDANT
    if dataset == "didemo":
        d["scales"] = str(list(range(2, 17, 2)))[1:-1].replace(",","")  ## DEFAULT PARAMETERS
        d["stride"] = 0.5                    ## DEFAULT PARAMETERS
        d["neg_samp_iou"] = 0.99
        d["nms_threshold"] = 1.0
        d["min_length"] = 1.5                ## DEFAULT PARAMETERS
        d["feat"] = "--feat rgb " 
        d["num_workers_eval"] = "--num-workers-eval 4 " 
        d["min_epochs"] = ""
        d["momentum"] = ""
        d["n_display"] = "--n-display 29.0"
        d["clip_length"] = ""
    elif dataset == "charades-sta":
        d["scales"] = str("2 3 4 5 6 7 8")
        d["stride"] = 0.3
        d["neg_samp_iou"] = 0.35
        d["nms_threshold"] = 0.6
        d["min_length"] = 3   
        d["feat"] = "--feat rgb " 
        d["num_workers_eval"] = "--num-workers-eval 4 "  
        d["min_epochs"] = ""  
        d["momentum"] = ""
        d["n_display"] = "--n-display 29.0"
        d["clip_length"] = ""
    elif dataset == "activitynet-captions":
        d["scales"] = "2 4 6 8 10 12 14 16 18 20 22 24 26"
        d["stride"] = 0.3
        d["neg_samp_iou"] = 0.35
        d["nms_threshold"] = 0.6
        d["min_length"] = 5.0
        d["feat"] = "--feat rgb "
        d["num_workers_eval"] = "--num-workers-eval 6 "
        d["min_epochs"] = "--min-epochs 48.0"
        d["momentum"] = "--momentum 0.95"
        d["n_display"] = "--n-display 29.0"
        d["clip_length"] = ""
        
    return names, h5_files, locs, d


def sbatch_CALChamfer(launch, num_iter,dataset,mode,server):
    #Static parameters
    model = "CALChamfer"
    dataset_dir = "data/processed/{}".format(dataset)
    output_dir = "data/interim/{}/{}/".format(dataset,model)
    detail_exp = ""

    names, h5_files, locs, parameters = get_param_per_dataset(dataset,dataset_dir,model,mode,server)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # we are performing test on val data, check setup_kwargs for changes.        
    for i in range(num_iter):
        for name,loc,h5_file in zip(names,locs,h5_files):
            parameters["loc"] = loc
            parameters["output_dir"] = output_dir+name+detail_exp+str(i)
            parameters["name"] = name+detail_exp+str(i)
            parameters["h5_file"] = h5_file

            kwargs =setup_kwargs(**parameters)
                
            # Write sbatc file
            f = open("./sbatches/{}{}{}.sh".format(name,detail_exp,i), "w")
            f.write(TEMPLATE.format(**kwargs))
            f.close()
            if launch=="True":
                os.system("sbatch ./sbatches/{}{}{}.sh".format(name,detail_exp,i))
            else:
                print("sbatch ./sbatches/{}{}{}.sh".format(name,detail_exp,i))

    # for i in range(num_iter):
    #     for dropout in parameters["list_dropout"]:

    #         # parameters["lang_dropout"] = "--lang-dropout {}".format(dropout)
    #         parameters["dropout_tef"] = "--dropout-tef  {}".format(dropout)
    #         for name,loc,h5_file in zip(names,locs,h5_files):
    #             # descr = "pytorch_drop_"+str(dropout).replace(".","_")
    #             detail_exp2 = detail_exp+"_tef_drop_"+str(dropout).replace(".","_")+"_iter_"
    #             parameters["loc"] = loc
    #             parameters["output_dir"] = output_dir+name+detail_exp2+str(i)
    #             parameters["name"] = name+detail_exp2+str(i)
    #             parameters["h5_file"] = h5_file

    #             kwargs =setup_kwargs(**parameters)
                        
    #             #Write sbatc file
    #             f = open("./sbatches/{}{}{}.sh".format(name, detail_exp2,i), "w")
    #             f.write(TEMPLATE.format(**kwargs))
    #             f.close()
    #             if launch=="True":
    #                 os.system("sbatch ./sbatches/{}{}{}.sh".format(name, detail_exp2,i))
    #             else:
    #                 print("sbatch ./sbatches/{}{}{}.sh".format(name, detail_exp2,i))


def sbatch_SMCN(launch,num_iter,dataset,mode,server):
    #Static parameters
    model = "SMCN"
    dataset_dir = f"data/processed/{dataset}"
    output_dir = f"data/interim/{dataset}/{model}/"
    detail_exp = ""

    names, h5_files, locs, parameters = get_param_per_dataset(dataset,dataset_dir,model,mode,server)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # # we are performing test on val data, check setup_kwargs for changes.
    # for i in range(num_iter):
    #     for name,loc,h5_file in zip(names,locs,h5_files):
    #         parameters["loc"] = loc
    #         parameters["output_dir"] = f"{output_dir}{name}{detail_exp}{i}"
    #         parameters["name"] = f"{name}{detail_exp}{i}"
    #         parameters["h5_file"] = h5_file

    #         kwargs = setup_kwargs(**parameters)
                
    #         # Write sbatc file
    #         f = open(f"./sbatches/{parameters['name']}.sh", "w")
    #         f.write(TEMPLATE.format(**kwargs))
    #         f.close()
    #         cmd = f"sbatch ./sbatches/{parameters['name']}.sh"i) 
    #         if launch=="True":
    #             os.system(cmd)
    #         else:
    #             print(cmd)

    for i in range(num_iter):
        for gt_rate in parameters["gt_rate_list"]:
            parameters["gt_rate"] = "--ground-truth-rate  {}".format(gt_rate)
            for name,loc,h5_file in zip(names,locs,h5_files):
                detail_exp2 = "_gt_rate_"+str(gt_rate).replace(".","_")+f"_iter_{i}"
                parameters["loc"] = loc
                parameters["output_dir"] = output_dir+name+detail_exp2
                parameters["name"] = name+detail_exp2
                parameters["h5_file"] = h5_file

                kwargs =setup_kwargs(**parameters)
                        
                #Write sbatc file
                f = open(f"./sbatches/{parameters['name']}.sh", "w")
                f.write(TEMPLATE.format(**kwargs))
                f.close()
                cmd = f"sbatch ./sbatches/{parameters['name']}.sh"
                if launch=="True":
                    os.system(cmd)
                else:
                    print(cmd)


def sbatch_MCN(launch,num_iter,dataset,mode,server):
    #Static parameters
    model = "MCN"
    dataset_dir = f"data/processed/{dataset}"
    output_dir = f"data/interim/{dataset}/{model}/"
    detail_exp = ""

    names, h5_files, locs, parameters = get_param_per_dataset(dataset,dataset_dir,model,mode,server)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## we are performing test on val data, check setup_kwargs for changes.
    # for i in range(num_iter):
    #     for name,loc,h5_file in zip(names,locs,h5_files):
    #         parameters["loc"] = loc
    #         parameters["output_dir"] = f"{output_dir}{name}{detail_exp}{i}"
    #         parameters["name"] = f"{name}{detail_exp}{i}"
    #         parameters["h5_file"] = h5_file

    #         kwargs = setup_kwargs(**parameters)
                
    #         # Write sbatc file
    #         f = open(f"./sbatches/{parameters['name']}.sh", "w")
    #         f.write(TEMPLATE.format(**kwargs))
    #         f.close()
    #         cmd = f"sbatch ./sbatches/{parameters['name']}.sh"i) 
    #         if launch=="True":
    #             os.system(cmd)
    #         else:
    #             print(cmd)

    for i in range(num_iter):
        for gt_rate in parameters["gt_rate_list"]:
            parameters["gt_rate"] = "--ground-truth-rate  {}".format(gt_rate)
            for name,loc,h5_file in zip(names,locs,h5_files):
                detail_exp2 = "_gt_rate_"+str(gt_rate).replace(".","_")+f"_iter_{i}"
                parameters["loc"] = loc
                parameters["output_dir"] = output_dir+name+detail_exp2
                parameters["name"] = name+detail_exp2
                parameters["h5_file"] = h5_file

                kwargs =setup_kwargs(**parameters)
                        
                #Write sbatc file
                f = open(f"./sbatches/{parameters['name']}.sh", "w")
                f.write(TEMPLATE.format(**kwargs))
                f.close()
                cmd = f"sbatch ./sbatches/{parameters['name']}.sh"
                if launch=="True":
                    os.system(cmd)
                else:
                    print(cmd)


def sbatch_MCN_TEF_only(launch,num_iter,dataset,mode,server):
    #Static parameters
    model = "MCN"
    dataset_dir = "data/processed/{}".format(dataset)
    output_dir = "data/interim/{}/{}/".format(dataset,model)
    detail_exp = "_{}_".format(dataset.replace("-","_"))

    names, h5_files, locs, parameters = get_param_per_dataset(dataset,dataset_dir,model,mode,server)

    cl = 5
    if dataset == "charades-sta":
        cl = 3

    locs        = [""]
    h5_files    = [""]
    names       = ["{}_TEF_only_{}s_".format(model,cl)]
    parameters["clip_length"] = '--clip-length {}'.format(float(cl))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## we are performing test on val data, check setup_kwargs for changes.
    for i in range(num_iter):
        for name,loc,h5_file in zip(names,locs,h5_files):
            parameters["loc"] = loc
            parameters["output_dir"] = output_dir+name+detail_exp+str(i)
            parameters["name"] = name+detail_exp+str(i)
            parameters["h5_file"] = h5_file

            kwargs = setup_kwargs(**parameters)
                
            # Write sbatc file
            f = open("./sbatches/{}{}{}.sh".format(name,detail_exp,i), "w")
            f.write(TEMPLATE.format(**kwargs))
            f.close()
            if launch=="True":
                os.system("sbatch ./sbatches/{}{}{}.sh".format(name,detail_exp,i))
            else:
                print("sbatch ./sbatches/{}{}{}.sh".format(name,detail_exp,i))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Sbatch Moment-retrieval-project',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model',
                    choices= ["MCN","SMCN","CALChamfer","MCN_TEF_only", "ALL"],
                    help='Name of network to test.')
    parser.add_argument('--launch',choices= ["True","False"], default='False',
                    help='Run automatically sbatch.')
    parser.add_argument('--iter', help='Number of iterations.')
    parser.add_argument('--dataset', choices= ["charades-sta", "didemo", "activitynet-captions"],
                    help='Name of dataset.')
    parser.add_argument('--mode', choices= ["test", "val"], 
                    help='What are we doing? Developing or testing?.')
    parser.add_argument('--server', choices= ["skynet", "ibex"],
                    help='Enable or disable reservation.')

    args = parser.parse_args()

    
    if args.model == "CALChamfer" or args.model == "ALL":
        sbatch_CALChamfer(args.launch,int(args.iter),args.dataset,args.mode,args.server)
    if args.model == "SMCN" or args.model == "ALL":
        sbatch_SMCN(args.launch,int(args.iter),args.dataset,args.mode,args.server)
    if args.model == "MCN" or args.model == "ALL":
        sbatch_MCN(args.launch,int(args.iter),args.dataset,args.mode,args.server)
    if args.model == "MCN_TEF_only" or args.model == "ALL":
        sbatch_MCN_TEF_only(args.launch,int(args.iter),args.dataset,args.mode,args.server)
