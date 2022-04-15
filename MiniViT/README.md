# MiniViT: Compressing Vision Transformers with Weight Multiplexing

:sunny: Hiring research interns for neural architecture search, tiny transformer design, model compression projects: houwen.peng@microsoft.com.

**This is an official implementation of [MiniViT](https://arxiv.org/pdf/2204.07154.pdf), including Mini-DeiT and Mini-Swin.**

MiniViT is a new compression framework that achieves parameter reduction in vision transformers while retaining the same performance. The central idea of MiniViT is to multiplex the weights of consecutive transformer blocks. Specifically, we make the weights shared across layers, while imposing a transformation on the weights to increase diversity. Weight distillation over self-attention is also applied to transfer knowledge from large-scale ViT models to weight-multiplexed compact models.

<div align="center">
    <img width="100%" src=".figure/framework.png"/>
</div>

## Highlights

- **Accurate**

MiniViT reduces the size of Swin-B by **48%**, while achieving **1.0%** better Top-1 accuracy on ImageNet.

- **Small**

MiniViT can compress DeiT-B (86M) to **9M** (**9.7x**), without seriously compromising the accuracy.

## Model Zoo

For evaluation, we provide the checkpoints of our models in the following table.

Model | Params. | Input | Top-1 Acc. % | Top-5 Acc. % | Download link
--- |:---:|:---:|:---:|:---:|:---:
Mini-DeiT-Ti | 3M | 224x224 | 73.0 | 91.6 | [model](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini_deit_tiny_patch16_224.pth), [log](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/log_mini_deit_tiny.txt)
Mini-DeiT-S | 11M | 224x224 | 80.9 | 95.6 | [model](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini_deit_small_patch16_224.pth), [log](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/log_mini_deit_small.txt)
Mini-DeiT-B | 44M | 224x224 | 83.2 | 96.5 | [model](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini_deit_base_patch16_224.pth), [log](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/log_mini_deit_base.txt)
Mini-DeiT-B| 44M | 384x384 | 84.9 | 97.2 | [model](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini_deit_base_patch16_384.pth), [log](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/log_mini_deit_base_384.txt)
Mini-Swin-T | 12M | 224x224 | 81.3 | 95.7 | [model](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini-swin-tiny-12m.pth), [log](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/log_mini_swin_tiny.txt)
Mini-Swin-S | 26M | 224x224 | 83.9 | 97.0 | [model](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini-swin-small-26m.pth), [log](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/log_mini_swin_small.txt)
Mini-Swin-B | 46M | 224x224 | 84.5| 97.3 | [model](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini-swin-base-46m.pth), [log](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/log_mini_swin_base.txt)
Mini-Swin-B | 47M | 384x384 | 85.5 | 97.6 | [model](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini-swin-base-224to384.pth), [log](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/log_mini_swin_base_384.txt)


## Getting Started

- For **Mini-DeiT**, please see [Mini-DeiT](./Mini-DeiT) for detailed instructions.
- For **Mini-Swin**, please see [Mini-Swin](./Mini-Swin) for a quick start.

## Bibtex

If this repo is helpful for you, please consider to cite it. Thank you! :)
```bibtex
@article{MiniViT,
  title={MiniViT: Compressing Vision Transformers with Weight Multiplexing},
  author={Zhang, Jinnian and Peng, Houwen and Wu, Kan and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
