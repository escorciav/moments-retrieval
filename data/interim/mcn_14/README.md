# VGG-16
___

## Scope

gauge impact of TEF.

- model: MCN on VGG16 RGB features.

- training scheme: only DIDEMO videos.

- features: VGG16 pre-trained on [ILSVCR-2012](http://www.image-net.org/challenges/LSVRC/2012/) for object classification.

  Original features provided by [Lisa Anne](https://github.com/LisaAnne/LocalizingMoments#pre-extracted-features)

- hp: Almost as Lisa with higher momentum. Inherited from [MCN hps experiment](#006.-MCN-pytorch).

## Experiments
 
Local+Global+TEF

### a_replica

_Motivation_: making sure that we are able to obtain the same results, $\pm \epsilon$ tolerance.

_conclusions_:

1. MCN trimmed and/or untrimmed produce similar results in terms of $\mu$ and $\sigma$ for three trials.

1. Our pytorch re-implementation produces the same results as the python implementation.

_details about data_:

- The final JSON files correspond to the use our pytorch re-implementation of DiDeMo metrics.

- "BAK" correspond to original JSON dumped by training script.

- "BAK.1": correspond to the same JSON with results provided by `single_video_retrieval_didemo.py`.

  We used the python functions taken from original DiDeMo repo, and aggregate with numpy functions in float32.
