# ResNet-152
___

## Scope

compare effect of TEF with (temporal) mean pooling over clips.

- model: SMCN on ResNet152 RGB features.

- training scheme: only DIDEMO videos.

- features: resnet152 pre-trained on [ILSVCR-2012](http://www.image-net.org/challenges/LSVRC/2012/) for object classification.

- hp: Almost as Lisa with higher momentum. Inherited from [MCN hps experiment](#006.-MCN-pytorch).

## Experiments

- a 
  
  Local+Global+TEF

- b

  Local+Global