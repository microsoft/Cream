# TinyCLIP Training

In this document, we introduce ***auto weight inheritance*** and ***manual weight inheritance method*** to train a TinyCLIP model with the proposed ***cross-modalities distillation***. 

:star: **[Notice]** Please replace the training data loader with the one loading LAION-400M or YFCC-15M.

Reference: [OpenCLIP Data](https://github.com/mlfoundations/open_clip?tab=readme-ov-file#data)

###  Auto weight inheritance training
In this part, we compress OpenCLIP ViT-B/32 to 25% of origin size using three stages, where the model is compressed from 100% to 75%, from 75% to 50% and from 50% to 25% in the three stages, respectively.

One bash script corresponds to one stage training, training for the next stage begins after the completion of the previous stage. We use 4 nodes (8 GPUs per node) to train the model with auto weight inheritance:

```bash
sh script/auto_weight_inherit_100to75.sh # first stage
sh script/auto_weight_inherit_75to50.sh # second stage
sh script/auto_weight_inherit_50to25.sh # third stage
```

###  Manual weight inheritance training
In this part, we compress OpenCLIP ViT-B/32 to 50% of origin size using two stages, where the model is compressed from 100% to 75% and from 75% to 50% in the two stages, respectively.

The training with manual weight inheritance is conducted using four nodes, just as in the case of automatic weight inheritance.

```bash
sh script/manual_weight_inherit_100to75.sh # first stage
sh script/manual_weight_inherit_75to50.sh # second stage
```
