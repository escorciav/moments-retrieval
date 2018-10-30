# VGG-16
___

## Scope

gauge impact of TEF .

- model: SMCN on VGG16 RGB features.

- training scheme: only DIDEMO videos.

- features: VGG16 pre-trained on [ILSVCR-2012](http://www.image-net.org/challenges/LSVRC/2012/) for object classification.

  Original features provided by [Lisa Anne](https://github.com/LisaAnne/LocalizingMoments#pre-extracted-features)

- hp: Almost as Lisa with higher momentum. Inherited from [MCN hps experiment](#006.-MCN-pytorch).

## Experiments
 
Local+Global