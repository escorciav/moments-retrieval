# test

Files used to ensure packing/shipping went well (our dummy ContinousIntegration XD). This could also help us to track changes in the data a.k.a. towards software 2.0.

# didemo

- a58ff3f9ff67b5c727d602e7d2a2b236  didemo/resnet152_rgb_max_cl-2.5.h5
- 78215344f7cedb2e19d35b45e1375774  didemo/resnet152_rgb_max_cl-5.h5
- 3a0455d9af16b14b2a3b3ee285b62512  didemo/vgg16_rgb_max_cl-2.5.h5
- 929d7747d4f3b00ddf7b979c19eb5961  didemo/vgg16_rgb_max_cl-5.h5
- c8712421a837951d74adc8a5a011cb5a  didemo/test-03.json
- 010d79522f2b496a7d77df877ec465cf  didemo/train-03.json
- 62ee89251542ca8afee120132f47d4c5  didemo/val-03.json
- 740f0eb910621394edd65bef486b5d33  didemo/freq_prior.npz

```
Subset: train
	Num videos: 8511
	Num instances: 33005
	Dumped file: didemo/train.json
Subset: val
	Num videos: 1094
	Num instances: 4180
	Dumped file: didemo/val.json
Subset: test
	Num videos: 1037
	Num instances: 4021
	Dumped file: didemo/test.json
```

# charades-sta

- c1bf2714950bf07203bdcf6d6a74dbea  charades-sta/test.json
- 948b539a6ad9df44b9d43e4f99261068  charades-sta/train.json
- abe8f49c395484353e40122cf04005e3  charades-sta/rgb_resnet152_max_cs-3.h5
- 7394efa0a2e94e9bdc870287766b0d67  charades-sta/freq_prior.npz

# activitynet-captions

- c06ca1a238b78b4e777ac1ba5258f1dc  activitynet-captions/rgb_resnet152_max_cs-5.h5
- 194699da03d1ce870efb135f577c77e8  activitynet-captions/train.json
- 8da5db5ae56f43cecc15939c87ccbb2b  activitynet-captions/val.json
- fb770c4de666485d7535acb865da2561  activitynet-captions/freq_prior.npz

```
Subset: train
    Num videos: 10009
    Num instances: 37421
    Num dumped instances: 37416
    Dumped file: ../data/interim/activitynet/train.json
Subset: val
    Num videos: 4917
    Num instances: 34536
    Num dumped instances: 34526
    Dumped file: ../data/interim/activitynet/val.json
```

# 3rdparty

Data from other methods

- f40c475b4d8e76d726de419445e4ad9e  3rdparty/mee_video-retrieval_activitynet-captions_val.h5
- a53a632d608e57353c97ff543ddef25b  3rdparty/mee_video-retrieval_charades-sta_test.h5
- ad44c0ff4fa1c630ebd0f56f8e8378ff  3rdparty/mee_video-retrieval_didemo_test.h5