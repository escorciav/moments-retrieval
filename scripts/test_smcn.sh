EXTRAS="--epochs 1 --momentum 0.95 --original-setup --n-display 15 --no-shuffle --num-workers 0"
for i in {0..0}; do
  python train_smcn.py --feat rgb \
    --rgb-path data/interim/didemo/resnet152/320x240_max.h5 \
    --gpu-id $gpu_device \
    --logfile test_output/smcn_$i $EXTRAS;
done