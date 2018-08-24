[ ! -z $conda_env ] && conda_env=pytorch
[ ! -z $conda_root ] && conda_root=$HOME/miniconda3

# You should not edit anything from this point onwards
echo $(date): Setup conda
source $conda_root/etc/profile.d/conda.sh
conda --version &> /dev/null
if [ $? -ne 0 ]; then
  echo Please setup conda properly to proceed, see you soon!
  return
fi
conda activate $conda_env
export PYTHONPATH=$PWD/..

echo $(date): Launching server
export FLASK_APP=server_moment_retrieval_demo.py
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
flask run --host=0.0.0.0 --port 2006