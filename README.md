# Finding Moments in Video Collections Using Natural Language 
![teaser][teaser]

[teaser]: https://github.com/escorciav/moments-retrieval/blob/fix-readme./data/images/readme.png "teaser image"

## Introduction

TODO: layman explanation of the content in the repo

In case you find this work relevant for your research please cite

```
TODO
```

## Getting started

1. Install all the required dependencies:

    The main requirements of this project are Python==3.6, numpy, matplotlib, jupyter, pytorch and unity-agents. To ease its installation, I recommend the following procedure:

    - [Install miniconda](https://conda.io/docs/user-guide/install/index.html).

      > Feel free to skip this step, if you already have anaconda or miniconda installed in your machine.

    - Creating the environment.

      `conda env create -n moments-retrieval-devel -f environment-devel.yml`

    - Activate the environment

      `conda activate moments-retrieval-devel`
      
2. Download data

    A snapshot of the processed data ready to train new models is available [here](https://drive.google.com/open?id=1hblwPxeI3u9w1VMZH-ZtD6J-Qnl6Q3xt)
  
    - Download it and unzip it. You should see a single directory called `data`.
  
      Let's assume that you place this folder in `[path]./data`.
  
    - Copy it into the data folder of the repo.
  
      ```bash
      cd moments-retrieval
      `cp -r [path]./data .`
      ```
    
      > Please remember to replace `[path]` with the actual folder of the downloaded data in your machine.
  
    TODO: write a bash script to do this.

## Instructions

### Training a new model
In oder to train from scratch a model use the following commands. You are required to specify:
 - Path to json files contraing the ground throuth annotations (see `Getting started` section)
 
 - Path to hdf5 files containing the precomputed
  features (see `Getting started` section)
 - Hyperparameters of the network (we present report the most succesfull) 
 - Logfile name (if required).

#### DiDeMo
```
dataset=./data/processed/didemo
interim=./data/interim/didemo/

parameters="--arch STAL
--train-list $dataset/train-01.json
--val-list   $dataset/val-01.json
--test-list  $dataset/test-01.json
--h5-path    $dataset/resnet152_rgb_max_cl-2.5.h5 $dataset/obj_predictions_perc_50_glove_bb_spatial.h5
--feat rgb obj
--original-setup
--proposals-in-train
--epochs 80
--lr 0.05
--lr-step 50
--lr-decay 0.05
--margin 0.5
--momentum 0.9
--dropout 0.1
--lang-hidden 1000
--visual-layers 1
--lang-dropout 0
--visual-hidden 500
--visual-layers 1
--stride 1
--min-length 4.5
--scales 2 3 4 5 6
--ground-truth-rate 1.0
--nms-threshold     1.0
--negative-sampling-iou 0.35
--proposal-interface DidemoICCV17SS
--logfile $interim/logfile_name"

python train.py --gpu-id 0 $parameters 
```

Alternatively the hyperparameters can be stored to a `hps.yml` file in the folder specified in `interim` and the
 keyword `--hps` can be used. 
 If the `--logfile` keyword is not specified then the code will not dump the snapshot of the trained model nor the
  configuration file.

#### Charades-STA
```
dataset=./data/processed/charades-sta
interim=./data/interim/charades-sta/

parameters="--arch STAL
--train-list $dataset/train-01.json
--val-list   $dataset/val-02_01.json
--test-list  $dataset/test-01.json
--h5-path    $dataset/resnet152_rgb_max_cl-3.h5 $dataset/obj_predictions_perc_50_glove_bb_spatial.h5
--feat rgb obj
--original-setup
--proposals-in-train
--epochs 120
--lr 0.05
--lr-step 50
--lr-decay 0.1
--margin 0.1
--momentum 0.9
--dropout 0.1
--lang-hidden 1000
--visual-layers 1
--lang-dropout 0
--visual-hidden 500
--visual-layers 1
--stride 0.5
--min-length 4.5
--scales 2 4 6 8 10 12
--ground-truth-rate 1.0
--nms-threshold     0.6
--negative-sampling-iou 0.35
--proposal-interface SlidingWindowMSRSS
--logfile $interim/logfile_name"

python train.py --gpu-id 0 $parameters 
```

### Single video retrieval evaluation
To run the single video retrieval evaluation of a pretrained model. An example if provided below for
 DiDeMo:
 ```
dataset=./data/processed/didemo
interim=./data/interim/didemo/

parameters="--arch STAL
--test-list $dataset/test-01.json
--h5-path   $dataset/resnet152_rgb_max_cl-2.5.h5 $dataset/obj_predictions_perc_50_glove_bb_spatial.h5
--snapshot  $interim/model_filename
--logfile   $interim/logfile_name
--evaluate "

python train.py --gpu-id 0 $parameters 
```

### TODO: Corpus video retrieval evaluation (Exhaustive search)
This evaluation requires a pretrained model its snapshot and configuration file (`.json`)

```
dataset=./data/processed/didemo
interim=./data/interim/didemo

parameters="--test-list $dataset/test-01.json
--h5-path  $dataset/resnet152_rgb_max_cl-2.5.h5 $dataset/obj_predictions_perc_50_glove_bb_spatial.h5
--tags     rgb obj
--snapshot $interim/pre-trained_model_name.json
--snapshot-tags rgb obj
--logfile  $interim/logfile_name
--n-display 0.2 --dump  "

python corpus_retrieval_eval.py $parameters
```

### Second stage corpus video retrieval evaluation

- Using a STAL (clips) model as first stage

TODO: More details
```
dataset=./data/processed/didemo
interim=./data/interim/didemo

parameters="--test-list $dataset/test-01.json 
--h5-path     $dataset/resnet152_rgb_max_cl-2.5.h5 $dataset/obj_predictions_perc_50_glove_bb_spatial.h5 
--h5-1ststage $interim/first_stage.h5
--snapshot    $interim/pre-trained_model_name.json
--logfile     $interim/logfile_name
--k-first 200 --nms-threshold 1.0 --n-display 0.2 --dump "

python corpus_retrieval_eval.py $parameters
```


- Using the approximate nearest neighbours search

TODO: More details
```
dataset=./data/processed/didemo
interim=./data/interim/didemo

parameters="--test-list $dataset/test-01.json 
--h5-path           $dataset/resnet152_rgb_max_cl-2.5.h5 $dataset/obj_predictions_perc_50_glove_bb_spatial.h5 
--h5-1ststage       $dataset/resnet152_rgb_max_cl-2.5.h5
--snapshot-1ststage $interim/pre-trained_model_name.json
--snapshot          $interim/pre-trained_model_name.json
--logfile           $interim/logfile_name
--corpus-setup      TwoStageClipPlusGeneric 
--k-first 200 --nms-threshold 1.0 --n-display 0.2 --dump "

python corpus_retrieval_eval.py $parameters
```

### TODO: dashboards

## Do you like the project?

Please give us ⭐️ in the GitHub banner 😉. We are also open for discussions especially accompany with ☕,🍺 or 🍸 (preferably spritz).

## LICENSE

[MIT](https://choosealicense.com/licenses/mit/)

We highly appreciate that you leave attribution notes when you copy portions of our codebase in yours.
