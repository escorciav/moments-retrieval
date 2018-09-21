# Usage: bash [filename-here]
# Note: experiment parameters
EXTRAS="--momentum 0.95 --original-setup --n-display 15 --num-workers 6"
gpu_device=0
# Note: comment to print logs, dont dump anything
output_dir=data/interim/smcn_debug
repetitions=0
offset=0
# environment variables
# Note: comment conda_root line if you don't know it
environment=moments-retrieval-devel
conda_root=$HOME/install/bin/miniconda3
# git variables
dont_track_this=*.log\n*.json\n*.csv\n*.h5\n*.tar
# Note: comment it if u dont care about reproducibility
# commit_reference=c9e604fab8fdffaa2e920587cab97fbb311179f2

# Forget about the following lines
conda --version > /dev/null
if [ $? -ne 0 ]; then
  if [ ! -z $conda_root ]; then
    source $conda_root/etc/profile.d/conda.sh
  else
    exit 127
  fi
fi
# zero-tolerance policy with errors from this point ongoing
set -e
# go to a given point if history
[ ! -z $commit_reference ] && git checkout $commit_reference
conda activate $environment

[ ! -z $output_dir ] && mkdir -p $output_dir
for ((i=$offset; i<=$repetitions; i++)); do
  if [ ! -z $output_dir ]; then
    # skip it when experiment-id was consumed before
    [ -f $output_dir/$i".json" ] && continue
    # add gitignore to dont crowd git-repo
    [ ! -f $output_dir/.gitignore ] && printf $dont_track_this >> $output_dir/.gitignore
    # augment EXTRAS with output_dir
    EXTRAS="$EXTRAS --logfile $output_dir/$i"
  fi
  # finaly the stuff that we care about
  python train_smcn.py --feat rgb \
    --rgb-path data/raw/average_fc7.h5 \
    --gpu-id $gpu_device \
    $EXTRAS;
done