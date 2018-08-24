# Edit the following lines and forget about the rest
# In case, u need a different device. Use -1 for cpu
gpu_device=0
# change it if you dont like the name :)
output_dir=test_output
# change the next line if you use a different name for the conda environment
env_name=pytorch
# Note: edit the following line if you also prefer exotic configs :)
conda_root=$HOME/miniconda3
# For example, the next line shows @escorciav setup for miniconda
# conda_root=$HOME/install/bin/miniconda3

# You should not edit anything from this point onwards
echo $(date): Setup conda
source $conda_root/etc/profile.d/conda.sh
echo $(date): Activating conda environment and setup PYTHONPATH
conda activate $env_name
export PYTHONPATH=$PWD
[ -d $output_dir ] && echo $(date): 'Backup and remove test folder to proceed. See u soon!'  && exit 1
echo $(date): lets do it
[ ! -d $output_dir ] && mkdir -p $output_dir
set -e
set -x

echo $(date): MCN test
source scripts/test_mcn.sh
echo $(date): SMCN test
source scripts/test_smcn.sh
echo $(date): HSMCN test
source scripts/test_hsmcn.sh
echo $(date): Corpus video moment retrieval test
source scripts/test_corpus_eval.sh
echo $(date): YFCC100M curation
python scripts/yfcc100m_subset.py main \
  --wildcard=/mnt/ilcompf2d1/data/yfcc100m/yfcc100m_dataset-0* \
  --prefix=test_output/nouns-under-and-not-leq
echo $(date): dumping distance matrix
source scripts/test_dump_dmatrix.sh