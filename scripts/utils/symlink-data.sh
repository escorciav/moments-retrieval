# """
# Link data required from training to repo
#
# Usage: bash symlink-data.sh
#
# """
# Note: edit your list of origin and destintation root folders
origin=$PWD
destintation=$PWD/workers/tyler
# Note: list of data required for training
data_files=(
  data/raw/vocab_glove_complete.txt
  data/raw/glove.6B.300d.txt
  data/processed/charades-sta/resnet152_rgb_max_cl-3.h5
  data/processed/charades-sta/resnet152_rgb_max_cl-1.5.h5
  data/processed/charades-sta/inceptionbn-imagenet-ucf101.1_max_cl-3.h5
  data/processed/charades-sta/test-01.json
  data/processed/charades-sta/train-01.json
  data/processed/charades-sta/val-02_01.json
  data/processed/charades-sta/train-02_01.json
  data/processed/charades-sta/train-02_02.json
  data/processed/activitynet-captions/resnet152_rgb_max_cl-5.h5
  data/processed/activitynet-captions/resnet152_rgb_max_cl-2.5.h5
  data/processed/activitynet-captions/val.json
  data/processed/activitynet-captions/train.json
  data/processed/didemo/resnet152_rgb_max_cl-2.5.h5
  data/processed/didemo/resnet152_rgb_max_cl-5.h5
  # additional HDF5s for DiDeMo
  # data/raw/average_fc7.h5
  # data/raw/average_global_flow.h5
  data/processed/didemo/train-03.json
  data/processed/didemo/train-03_01.json
  data/processed/didemo/val-03.json
  data/processed/didemo/test-03.json
)

if [ ! -d $origin ]; then
  echo Check origin: $origin
  exit 1
fi
if [ ! -d $destination ]; then
  echo Check destination: $destination
  exit 1
fi

for j in ${!data_files[@]}; do
  file_j=${data_files[$j]}
  if [ ! -f $origin/$file_j ]; then
    echo skip, file not found in origin: $file_j
    continue
  fi
  if [ -f $destintation/$file_j ]; then
    echo skip, file found in destination: $file_j
  else
    ln $origin/$file_j $destintation/$file_j
  fi
done