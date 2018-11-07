# SMCN-21
___

## Scope

Ablation of SMCN in Charades-STA.

- model: SMCN on ResNet152 RGB features.

- training scheme: only Charades-STA videos.

- features: ResNet-152 pre-trained on [ILSVCR-2012](http://www.image-net.org/challenges/LSVRC/2012/) for object classification.

- hp: Almost as Lisa with higher momentum. Inherited from MCN hps experiment.

## Experiments

### a

Local+Global+TEF

### b

Local+Global

### c

Local

### d

TEF := original temporal endpoints feature


1. Before updating sampling of proposals

### e

TSPF := temporal start endpoint feature

### f

TAF := clip-based TEF

- f1: Hyper-parameter search

  - 1-200:

    `UntrimmedBasedMCNStyle.MAGIC_TIOU == 0.99`

  - 201-350

    `UntrimmedBasedMCNStyle.MAGIC_TIOU == 0.3`

### g

Local+Global+TAF

## Notes

- `.BAK` files were the original JSON files dumped by our program.