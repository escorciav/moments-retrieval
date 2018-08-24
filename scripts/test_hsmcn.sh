EXTRAS="--epochs 1 --momentum 0.95 --original-setup --n-display 15 --num-workers 0 --no-loc --no-context"
for i in {0..0}; do
  python train_hsmcn.py --feat rgb \
    --rgb-path data/interim/didemo/resnet152/320x240_max.h5 data/interim/yfcc100m/resnet152/320x240_001.h5 \
    --train-list data/interim/didemo_yfcc100m/train_data.json \
    --gpu-id $gpu_device \
    --logfile test_output/hsmcn_$i $EXTRAS;
done