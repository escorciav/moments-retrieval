# Temporal Localization of Moments in Video Collections with Natural Language

TODO: add teaser and overview images

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