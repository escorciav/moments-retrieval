This document describes (part) of our apporach to get flow features.

# Optical flow images and features

We use the docker image of the [TSN codebase](https://github.com/yjxiong/temporal-segment-networks#testing-provided-models)

- Launch container:

    Preserve user

    `docker run -ti --runtime=nvidia -v /etc/passwd:/etc/passwd -u $(id -u):$(id -g) bitxiong/tsn bash`

    To edit the /app while mounting current directory

    `docker run -ti --runtime=nvidia --v $(pwd):$(pwd) bitxiong/tsn bash`

    Extravaganza

    `docker run -ti --runtime=nvidia -v /etc/passwd:/etc/passwd -u $(id -u):$(id -g) -v $(pwd):$(pwd) -v $(pwd)/projects:$(pwd)/projects -w $(pwd) bitxiong/tsn bash`

- Go to the code:
    `cd /app`

- Make caffe visible:
    `export PYTHONPATH=$PYTHONPATH:/app/lib/caffe-action/python`

- Make opencv visible:
    `export PYTHONPATH=$PYTHONPATH:/app`

Dumping and postprocessing was delegated to @soldelli.

_Notes_

We use the docker-image with tagged as latest. At that time, there were only two. We assumed is the same and corresponds to `cuda9_cudnn7`.