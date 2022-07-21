# TinyViT: Fast Pretraining Distillation for Small Vision Transformers

:sunny: Hiring research interns for neural architecture search, tiny transformer design, model compression projects: houwen.peng@microsoft.com.

**This is an official implementation of TinyViT.**

**[ECCV 2022]** - TinyViT: Fast Pretraining Distillation for Small Vision Transformers

TinyViT is a new family of tiny and efficient vision transformers pretrained on large-scale datasets with out proposed fast distillation framework. The central idea is to transfer knowledge from large pretrained models to small ones, while enabling small models to get the dividends of massive pretraining data. More specifically, we apply distillation during pretraining for knowledge transfer. The logits of large teacher models are sparsified and stored in disk in advance to save the memory cost and computation overheads.

<div align="center">
    <img width="80%" alt="TinyViT overview" src=".figure/framework.png"/>
</div>

## Highlights

<div align="center">
    <img width="80%" src=".figure/performance.png"/>
</div>

* 1. A fast pretraining distillation framework to unleash the capacity of small models by fully leveraging the large-scale pretraining data.
* 2. A new family of tiny vision transformer models, striking a good trade-off between computation and accuracy.


## Model Zoo

For evaluation, we provide the links of our models in the following table.


Model                                      | Pretrain | Input | Acc@1 | Acc@5 | #Params | MACs | FPS  | 22k Model | 1k Model
:-----------------------------------------:|:---------|:-----:|:-----:|:-----:|:-------:|:----:|:----:|:---------:|:--------:
TinyViT-5M ![](./.figure/distill.png)       | IN-22k   |224x224| 80.7  | 95.6  | 5.4M    | 1.3G | 3,060|[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22k_distill.pth)/[config](./configs/22k_distill/tiny_vit_5m_22k_distill.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22k_distill.log)|[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.pth)/[config](./configs/22kto1k/tiny_vit_5m_22kto1k.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.log)
TinyViT-11M ![](./.figure/distill.png)      | IN-22k   |224x224| 83.2  | 96.5  | 11M     | 2.0G | 2,468|[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22k_distill.pth)/[config](./configs/22k_distill/tiny_vit_11m_22k_distill.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22k_distill.log)|[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.pth)/[config](./configs/22kto1k/tiny_vit_11m_22kto1k.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.log)
TinyViT-21M ![](./.figure/distill.png)      | IN-22k   |224x224| 84.8  | 97.3  | 21M     | 4.3G | 1,571|[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.pth)/[config](./configs/22k_distill/tiny_vit_21m_22k_distill.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.log)|[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.pth)/[config](./configs/22kto1k/tiny_vit_21m_22kto1k.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.log)
TinyViT-21M-384 ![](./.figure/distill.png)  | IN-22k   |384x384| 86.2  | 97.8  | 21M     | 13.8G| 394  ||[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_384_distill.pth)/[config](./configs/higher_resolution/tiny_vit_21m_224to384.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_384_distill.log)
TinyViT-21M-512 ![](./.figure/distill.png)  | IN-22k   |512x512| 86.5  | 97.9  | 21M     | 27.0G| 167  ||[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_512_distill.pth)/[config](./configs/higher_resolution/tiny_vit_21m_384to512.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_512_distill.log)
TinyViT-5M                                 | IN-1k    |224x224| 79.1  | 94.8  | 5.4M    | 1.3G | 3,060||[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_1k.pth)/[config](./configs/1k/tiny_vit_5m.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_1k.log)
TinyViT-11M                                | IN-1k    |224x224| 81.5  | 95.8  | 11M     | 2.0G | 2,468||[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_1k.pth)/[config](./configs/1k/tiny_vit_11m.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_1k.log)
TinyViT-21M                                | IN-1k    |224x224| 83.1  | 96.5  | 21M     | 4.3G | 1,571||[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_1k.pth)/[config](./configs/1k/tiny_vit_21m.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_1k.log)

The models with ![](./.figure/distill.png) are pretrained on ImageNet-22k with the distillation of CLIP-ViT-L/14-22k, then finetuned on ImageNet-1k.

ImageNet-22k (IN-22k) is the same as ImageNet-21k (IN-21k), where the number of classes is 21,841.


## Getting Started

### Install the requirements and prepare the datasets
- [Preparation](./docs/PREPARATION.md)

### Evaluation
- [Evaluation](./docs/EVALUATION.md)

## Pretrain a TinyViT model on ImageNet
- [Save Teacher Sparse Logits](./docs/SAVE_TEACHER_LOGITS.md)
- [Train TinyViT](./docs/TRAINING.md)

## License
- [License](./LICENSE)

## Acknowledge

Our code is based on [Swin Transformer](https://github.com/microsoft/swin-transformer), [LeViT](https://github.com/facebookresearch/LeViT) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models).
