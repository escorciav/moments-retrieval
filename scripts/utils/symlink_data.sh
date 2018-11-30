# """
# Link data required from training to repo
#
# Usage: bash symlink_data.sh
#
# """
# Note: edit your list of origin and destintation root folders
origin=$PWD
destintation=$PWD/workers/tyler
# Note: list of data required for training
data_files=(
  data/raw/vocab_glove_complete.txt
  data/raw/glove.6B.300d.txt
  data/raw/average_fc7.h5
  data/raw/average_global_flow.h5
  data/raw/test_data.json
  data/raw/train_data.json
  data/raw/val_data.json
  data/processed/charades-sta/test.json
  data/processed/charades-sta/train.json
  data/processed/charades-sta/rgb_resnet152_max_cs-3.h5
  data/processed/activitynet-captions/val.json
  data/processed/activitynet-captions/train.json
  data/processed/activitynet-captions/rgb_resnet152_max_cs-5.h5
  data/processed/didemo/rgb_resnet152_max.h5
  data/processed/didemo/train.json
  data/processed/didemo/val.json
  data/processed/didemo/test.json
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
    echo error, file not found in origin: $file_j
    continue
  fi
  if [ -f $destintation/$file_j ]; then
    echo skip, file found in destination: $file_j
  else
    ln $origin/$file_j $destintation/$file_j
  fi
done