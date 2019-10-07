# Temporal Localization of Moments in Video Collections with Natural Language
![teaser][teaser]

[teaser]: https://github.com/escorciav/moments-retrieval/blob/fix-readme/data/images/readme.png "teaser image"

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

      Let's assume that you place this folder in `[path]/data`.

    - Copy it into the data folder of the repo.

      ```bash
      cd moments-retrieval
      `cp -r [path]/data .`
      ```

      > Please remember to replace `[path]` with the actual folder of the downloaded data in your machine.

    TODO: write a bash script to do this.

## Instructions

### Training a new model

- The following line will launch the training of CAL in Charades-STA dataset.

  ```bash
  dataset_dir=data/processed/charades-sta
  parameters="--arch SMCN --feat rgb --train-list $dataset_dir/train.json --val-list $dataset_dir/val-01.json --test-list $dataset_dir/test.json --h5-path $dataset_dir/rgb_resnet152_max_cs-3.h5"
  python train.py --gpu-id 0 $parameters
  ```

  In case you want to save the logs into a file, add `--logfile [filename]`. For example:

  ```bash
  dataset_dir=data/processed/charades-sta
  parameters="--logfile 1 --arch SMCN --feat rgb --train-list $dataset_dir/train.json --val-list $dataset_dir/val-01.json --test-list $dataset_dir/test.json --h5-path $dataset_dir/rgb_resnet152_max_cs-3.h5"
  python train.py --gpu-id 0 $parameters
  ```

  This time we write all the verbosity into the file `1.log`. Moreover, a model snapshot and the configuration setup are serialized into `1.pth.tar` and `1.json`, respectively.

  > In case you close the terminal, don't forget to activate the environment (`conda activate moments-retrieval`).

### TODO: corpus video retrieval evaluation

### TODO: single video retrieval evaluation

### TODO: dashboards

## Do you like the project?

Please give us ⭐️ in the GitHub banner 😉. We are also open for discussions especially accompany with ☕,🍺 or 🍸 (preferably spritz).

## LICENSE

[MIT](https://choosealicense.com/licenses/mit/)

We highly appreciate that you leave attribution notes when you copy portions of our codebase in yours.

----

## Usage

Below you will find how to use the programs that comes with our codebase.

### Training

Training a model

```bash
data_dir=data/processed/didemo
parameters="--proposal-interface DidemoICCV17SS --arch SMCN --feat rgb --train-list $data_dir/train-03.json --val-list $data_dir/train-03_01.json --test-list $data_dir/val-03.json --h5-path $data_dir/resnet152_rgb_max_cl-2.5.h5 --nms-threshold 1"
python -m ipdb train.py --gpu-id 0 $parameters --debug --epochs 1 --h5-path-nis workers/tyler/data/interim/smcn_40/b/train/1_corpus-eval.h5 --num-workers 0 --snapshot workers/tyler/data/interim/smcn_40/b/1.json
```

### Corpus retrieval

Any corpus retrieval experiment required a pre-trained model.

#### Exhaustive search with single stage

```bash
python corpus_retrieval_eval.py --test-list data/processed/activitynet-captions/val.json --h5-path data/processed/activitynet-captions/resnet152_rgb_max_cl-5.h5 --snapshot workers/skynet-base/data/interim/mcn_43/b/1.json --dump-per-instance-results --dump
```

### Two-stage corpus retrieval

python corpus_retrieval_2nd_eval.py --test-list data/processed/didemo/test-03.json --h5-path data/processed/didemo/resnet152_rgb_max_cl-5.h5 --snapshot workers/ibex-scratch/data/interim/mcn_41/a/1.json --h5-1ststage workers/ibex-scratch/data/interim/mcn_41/b/1_corpus-eval.h5 --k-first 200 --nms-threshold 1.0 --debug

#### Approximated setup 2

Evaluating model, CAL + CAL-TEF:

```bash
python corpus_retrieval_2nd_eval.py --corpus-setup TwoStageClipPlusGeneric --test-list data/processed/didemo/test-03.json --h5-path data/processed/didemo/resnet152_rgb_max_cl-2.5.h5 --snapshot workers/tyler/data/interim/smcn_40/a/1.json --h5-1ststage data/processed/didemo/resnet152_rgb_max_cl-2.5.h5 --snapshot-1ststage workers/tyler/data/interim/smcn_40/b/1.json --k-first 200
```

- Dumping data for re-training

  1. Apply the patch

      `git apply scripts/patches/dump-clip-retrieval-results`

      For ActivityNet, it's better use this or have enough storage

      `git apply scripts/patches/dump-clip-retrieval-results-activitynet`

  2. Launch

      ```bash
      python corpus_retrieval_eval.py --test-list data/processed/didemo/  train-03.json --h5-path data/processed/didemo/resnet152_rgb_max_cl-2.5.h5   --snapshot data/interim/smcn_40/b/4.json --clip-based   --dump-per-instance-results
      ```

      ```bash
      python corpus_retrieval_eval.py --test-list data/processed/charades-sta/  train-01.json --h5-path data/processed/charades-sta/  resnet152_rgb_max_cl-3.h5 --snapshot data/interim/smcn_42/b/3.json   --clip-based --dump-per-instance-results
      ```

      ```bash
      python corpus_retrieval_eval.py --test-list data/processed/activitynet-captions/train.json --h5-path data/processed/activitynet-captions/resnet152_rgb_max_cl-5.h5 --snapshot data/interim/smcn_43/b/3.json --clip-based --dump-per-instance-results
      ```

- Re-training

  1. `git checkout eb7f2d73bd20bb4fbc22a8d2f4dc003807248cef`

  2. Update bash script used for smcn_[47-48].

- Launch re-trained models

```bash
python corpus_retrieval_2nd_eval.py --corpus-setup TwoStageClipPlusGeneric --test-list data/processed/didemo/test-03.json --h5-path data/processed/didemo/resnet152_rgb_max_cl-2.5.h5 --snapshot data/interim/smcn_50/b/1.json --h5-1ststage data/processed/didemo/resnet152_rgb_max_cl-2.5.h5 --snapshot-1ststage data/interim/smcn_40/b/4.json --k-first 200 --dump --output-prefix cr-msm_approx-smcn-40b_nms-1

python corpus_retrieval_2nd_eval.py --corpus-setup TwoStageClipPlusGeneric --test-list data/processed/charades-sta/test-01.json --h5-path data/processed/charades-sta/resnet152_rgb_max_cl-3.h5 --snapshot data/interim/smcn_51/a/1.json --h5-1ststage data/processed/charades-sta/resnet152_rgb_max_cl-3.h5 --snapshot-1ststage data/interim/smcn_42/b/3.json --k-first 200 --dump --output-prefix cr-msm_approx-smcn-42b_nms-1

python corpus_retrieval_2nd_eval.py --corpus-setup TwoStageClipPlusGeneric --test-list data/processed/activitynet-captions/val.json --h5-path data/processed/activitynet-captions/resnet152_rgb_max_cl-5.h5 --snapshot data/interim/smcn_52/a/1.json --h5-1ststage data/processed/activitynet-captions/resnet152_rgb_max_cl-5.h5 --snapshot-1ststage data/interim/smcn_43/a/3.json --k-first 200 --dump --output-prefix cr-msm_approx-smcn-43b_nms-1
```

#### Aproximate setup fast

CAL - MCN

```bash
python corpus_retrieval_2nd_eval.py --corpus-setup TwoStageClipPlusMCNFast --test-list data/processed/activitynet-captions/val.json --h5-path data/processed/activitynet-captions/resnet152_rgb_max_cl-5.h5 --snapshot workers/tyler/data/interim/mcn_43/a/3.json --h5-1ststage data/processed/activitynet-captions/resnet152_rgb_max_cl-5.h5 --snapshot-1ststage workers/tyler/data/interim/smcn_43/b/3.json --k-first 200
```

CAL - CAL

```bash
python -m ipdb corpus_retrieval_2nd_eval.py --corpus-setup TwoStageClipPlusCALFast --test-list data/processed/activitynet-captions/val.json --h5-path data/processed/activitynet-captions/resnet152_rgb_max_cl-5.h5 --snapshot data/interim/smcn_43/a/3.json --h5-1ststage data/processed/activitynet-captions/resnet152_rgb_max_cl-5.h5 --snapshot-1ststage data/interim/smcn_43/b/3.json --k-first 200 --debug
```

Debugging fast-retrieval branch

python -m ipdb corpus_retrieval_2nd_eval.py --corpus-setup TwoStageClipPlusCALFast --test-list data/processed/activitynet-captions/val.json --h5-path data/processed/activitynet-captions/resnet152_rgb_max_cl-5.h5 --snapshot data/interim/smcn_49/a/1.json --h5-1ststage data/processed/activitynet-captions/resnet152_rgb_max_cl-5.h5 --snapshot-1ststage data/interim/smcn_43/b/3.json --k-first 200 --output-prefix replicate-fast

#### Baselines for moment retrieval on a video corpus

Example on Charades-STA

```bash
python mfp_corpus_eval.py \
    --test-list data/processed/charades-sta/test-01.json \
    --snapshot data/processed/charades-sta/mfp/1.json \
    --logfile data/processed/charades-sta/mfp/1_corpus-eval \
    --dump --topk 1 10 100 \
```

Example on ActivityNet-Captions

```bash
python mfp_corpus_eval.py \
    --test-list data/processed/activitynet-captions/val.json \
    --snapshot data/processed/activitynet-captions/mfp/1.json \
    --logfile  data/processed/activitynet-captions/mfp/1_corpus-eval \
    --topk 1 10 100 --dump
```

To compute chance supply the `--chance` to the commands above.

### (optional) video retrieval evaluation

python eval_video_retrieval.py --test-list data/processed/activitynet-captions/val.json --snapshot data/processed/3rdparty/mee_video-retrieval_activitynet-captions_val.h5 --h5-1ststage data/processed/3rdparty/mee_video-retrieval_activitynet-captions_val.h5 --topk 1 10 100 1000 --dump

### Single video moment retrieval

#### DiDeMo evaluation

```bash
python single_video_retrieval_didemo.py --val-list data/processed/didemo/val-03.json --test-list data/processed/didemo/test-03.json --h5-path data/processed/didemo/resnet152_rgb_max_cl-2.5.h5 --snapshot workers/tyler/data/interim/mcn_41/a/1.json --dump
```

#### Moment frequecy prior for moment retrieval on single video

Example Charades-STA:

```bash
python moment_freq_prior.py \
    --train-list data/processed/charades-sta/train-01.json \
    --test-list data/processed/charades-sta/test-01.json \
    --logfile data/processed/charades-sta/mfp/1
    --clip-length 3 --bins 75 \
    --proposal-interface SlidingWindowMSRSS \
    --min-length 3 --scales 2 3 4 5 6 7 8 \
    --stride 0.3 --nms-threshold 0.6 \
```

Example ActivityNet-Captions:

```bash
python moment_freq_prior.py \
    --train-list data/processed/activitynet-captions/train.json \
    --test-list data/processed/activitynet-captions/val.json \
    --logfile data/processed/activitynet-captions/mfp/1
    --clip-length 5 --bins 500 \
    --proposal-interface SlidingWindowMSRSS \
    --min-length 5 --scales 2 4 6 8 10 12 14 16 18 20 22 24 26 \
    --stride 0.3 --nms-threshold 0.6 \
```

Remove the `logfile` if you prefer to to print into shell and copy-paste results.

### Debug CAL-Chamfer

data_dir=data/processed/didemo
parameters="--test-list $dataset_dir/test-03.json --h5-path $dataset_dir/resnet152_rgb_max_cl-5.h5 --corpus-setup LoopOverKMoments --snapshot data/interim/calchamfer_deploy/DIDEMO/whole_trainingset/ModelD_TEF_5s_7.json --k-first 10 --h5-1ststage workers/tyler/data/interim/smcn_40/b/test/1_corpus-eval.h5"
python corpus_retrieval_2nd_eval.py $parameters --k-first 1 --debug