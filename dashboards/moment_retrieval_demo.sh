conda_env=pytorch

if hash conda 2>/dev/null; then
  echo
else
  source $HOME/install/bin/miniconda3/etc/profile.d/conda.sh
fi
conda activate $conda_env
export PYTHONPATH=$PWD/..

export FLASK_APP=server_moment_retrieval_demo.py
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
flask run --host=0.0.0.0 --port 2006