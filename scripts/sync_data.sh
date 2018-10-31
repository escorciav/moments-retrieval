# """
# Sync data required from training between master and workers
#
# Requirements: rsync
# Usage: bash sync_data.sh
#
# """
# Note: edit your list of servers mounted in workers/
workers=(ibex-fscratch skynet marla)
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

# Loop over workers
for i in ${!workers[@]}; do
  local_point=workers/${workers[$i]}
  if [ ! -d $local_point ]; then
    echo $(date) missing worker: ${workers[$i]}
    continue
  fi
  for j in ${!data_files[@]}; do
    file_j=${data_files[$j]}
    dirname_j=$(dirname $file_j)
    if [ ! -d $local_point/$dirname_j ]; then
      mkdir -p $local_point/$dirname_j
    fi
    rsync -rlptDL --partial --progress $file_j $local_point/$file_j
  done
done
