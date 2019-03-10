# """
# Sync required data between master and workers
#
# Requirements: rsync
# Usage: bash sync-data.sh
#
# """
# Note: edit your list of servers mounted in workers/
workers=(ibex-fscratch ibex-scratch skynet-root marla-root)
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
