# test

Listing relevant details about files used to train/test models or that workeas baselines. Our dummy manual version data continous integration.

The relevant files and sub-directories placed inside these folder are listed as `h2` elements i.e.`##` for those familiar with markdown `##`.
They are *not* sync with git. Instead, you need to download it from Google-Drive using a helper script (TODO).

## annotations.txt

TXT-file listing files with annotations. Used to retrieve data grom Google-Drive.

## features.txt

TXT-file listing visual features. Used to retrieve data grom Google-Drive. In case you are interested on language features, you can find some in Drive for DiDeMo.

## baselines.txt

TXT-file listing files with baselines models and results. Used to retrieve data from Google-Drive.

## didemo

The `[train,val,test]-03` subsets were generated from the original `train/val/test` partitions of DiDeMo. We only augment and did minor formats to the original data such as working in seconds as opposed to clip indixes.

Details of the formatting changes are on the notebook `13-DiDeMo` in the section `1b. JSON files`.

### DiDeMo md5sum

- c8712421a837951d74adc8a5a011cb5a  didemo/test-03.json
- 010d79522f2b496a7d77df877ec465cf  didemo/train-03.json
- 62ee89251542ca8afee120132f47d4c5  didemo/val-03.json
- a58ff3f9ff67b5c727d602e7d2a2b236  didemo/resnet152_rgb_max_cl-2.5.h5
- 78215344f7cedb2e19d35b45e1375774  didemo/resnet152_rgb_max_cl-5.h5

_Note_: The `03` suffix corresponds to a historical versioning of the files. You are free to update them, but we still need to work a bit more on it as there are multiple log&config files referring to them. In other words, we don't want to throw away our reproducibility.

### Baselines

- mfp

These folders contain results and models described in our paper as _Moment Frequency Prior_.

## charades-sta

The `train-01` and `test-01` subset were generated from the original `train/test` partitions of Charades-STA. We sanitize the data:

- clamping moments inside video, according to duration of downloaded videos.

- remove moments with duration <= 0. This step did not affect the `test` set.

Details of the sanitation steps are on the notebook `11-charades-sta` in the section `2.a.1 JSON files`.

### Charades-STA md5sum

- 35051c83f70f2761354b457baa37f4a0  data/processed/charades-sta/test-01.json
- 36e43851e20ee2d59aa45f61784a792b  data/processed/charades-sta/train-01.json
- 3ceac4b937d09a7df2ef64526d7a49b7  data/processed/charades-sta/train-02_01.json
- d36c80c9a41329edcab483a9e2b39388  data/processed/charades-sta/val-02_01.json
- b9a23fbe89980220a380e7aa344e4646  data/processed/charades-sta/resnet152_rgb_max_cl-3.h5

The files [train,val]-02*.json corresponds to a train&validation subsets from `train-01` used to explore hyperparameters.

### Baselines

- mfp
- chance

These folders contain results and models described in our paper as _Moment Frequency Prior_ and _random chance_.

## activitynet-captions

The `train` and `val` subset were generated from the original `train/[val1 + val2]` partitions of ActivityNet-Captions. The sanitation consisted on:

- clamp moments inside video, according to duration of downloaded videos.

- remove moments with duration <= 0

Details of the sanitation steps are on the notebook `12-ActivityNet-captions` in the section `2b.-JSON-files`.

### ActivityNet-Captions md5sum

- f8b6361136e802ec825527d21b325c78  data/processed/activitynet-captions/train.json
- c1fc120bcc691b3fce70c02ec7e3b35d  data/processed/activitynet-captions/val.json
- dfe6b6d35a5d363dd430434d5629194a  data/processed/activitynet-captions/train-01.json
- 79a1686814d14d6e96da5cb20f00cb36  data/processed/activitynet-captions/val-01.json
- 73cadb2b1446b281218409d27ec6c387  data/processed/activitynet-captions/resnet152_rgb_max_cl-5.h5

The files [train,val]-01*.json corresponds to train/val partition used to explore hyperparameters.

## Baselines

- mfp
- chance

These folders contain results and models described in our paper as _Moment Frequency Prior_ and _random chance_.

## 3rdparty

Data from other methods

- f40c475b4d8e76d726de419445e4ad9e  3rdparty/mee_video-retrieval_activitynet-captions_val.h5
- a53a632d608e57353c97ff543ddef25b  3rdparty/mee_video-retrieval_charades-sta_test.h5
- ad44c0ff4fa1c630ebd0f56f8e8378ff  3rdparty/mee_video-retrieval_didemo_test.h5

# License

The data listed and linked from this file regarded as features, models and results are distributed for academic research purposes under the terms of CC BY-NC-4.0.