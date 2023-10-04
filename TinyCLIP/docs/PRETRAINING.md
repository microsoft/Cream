# TinyCLIP Training

In this document, we introduce ***auto weight inheritance*** and ***mannual weight inheritance method*** to train a TinyCLIP model with the proposed ***cross-modalities distillation***. 

###  Auto weight inheritance training
In this part, we compress OpenCLIP ViT-B/32 to 25% of origin size using three stages, first stage compresses model from 100% to 75%, second stage compresses model from 75% to 50%, third stage compresses model from 50% to 25%.

One bash script corresponding to one stage training, training for the next stage begins after the completion of the previous stage. We use 4 nodes (8 GPUs per node) to do auto weight inheritance training:

```bash
sh script/auto_weight_inherit_100to75.sh # first stage
sh script/auto_weight_inherit_75to50.sh # second stage
sh script/auto_weight_inherit_50to25.sh # third stage
```

###  Manual weight inheritance training
In this part, we compress OpenCLIP ViT-B/32 to 50% of origin size using two stages,
first stage compresses model from 100% to 75%, second stage compresses model from 75% to 50%.


The training for manual weight inheritance is conducted using four nodes, just as in the case of automatic weight inheritance.

```bash
sh script/manual_weight_inherit_100to75.sh # first stage
sh script/manual_weight_inherit_75to50.sh # second stage
```