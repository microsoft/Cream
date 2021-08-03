Hiring research interns for neural architecture search projects: houwen.peng@microsoft.com

# Rethinking and Improving Relative Position Encoding for Vision Transformer

[[Paper]](https://houwenpeng.com/publications/iRPE.pdf)

Image Classification: DeiT with iRPE

# Model Zoo

We equip DeiT models with contextual product shared-head RPE with 50 buckets, and report their accuracy on ImageNet-1K Validation set.

Resolution: `224 x 224`

Model | RPE-Q | RPE-K | RPE-V | #Params(M) | MACs(M) | Top-1 Acc.(%) | Top-5 Acc.(%) | Link | Log
----- | ----- | ----- | ----- | ---------- | ------- | ------------- | ------------- | ---- | ---
tiny | | ✔ | | 5.76 | 1284 | 73.7 | 92.0 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_tiny_patch16_224_ctx_product_50_shared_k.pth) | [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_deit_tiny_patch16_224_ctx_product_50_shared_k.txt), [detail](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_deit_tiny_patch16_224_ctx_product_50_shared_k.log)
small | | ✔ | | 22.09 | 4659 | 80.9 | 95.4 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_small_patch16_224_ctx_product_50_shared_k.pth) | [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_deit_small_patch16_224_ctx_product_50_shared_k.txt), [detail](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_deit_small_patch16_224_ctx_product_50_shared_k.log)
small | ✔ | ✔ | | 22.13 | 4706 | 81.0 | 95.5 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_small_patch16_224_ctx_product_50_shared_qk.pth) | [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_deit_small_patch16_224_ctx_product_50_shared_qk.txt), [detail](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_deit_small_patch16_224_ctx_product_50_shared_qk.log)
small | ✔ | ✔ | ✔ | 22.17 | 4885 | 81.2 | 95.5 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_small_patch16_224_ctx_product_50_shared_qkv.pth) | [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_deit_small_patch16_224_ctx_product_50_shared_qkv.txt), [detail](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_deit_small_patch16_224_ctx_product_50_shared_qkv.log)
base | | ✔ | | 86.61 | 17684 | 82.3 | 95.9 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_base_patch16_224_ctx_product_50_shared_k.pth) | [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_deit_base_patch16_224_ctx_product_50_shared_k.txt), [detail](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_deit_base_patch16_224_ctx_product_50_shared_k.log)
base | ✔ | ✔ | ✔ | 86.68 | 18137 | 82.8 | 96.1 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_base_patch16_224_ctx_product_50_shared_qkv.pth) | [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_deit_base_patch16_224_ctx_product_50_shared_qkv.txt), [detail](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_deit_base_patch16_224_ctx_product_50_shared_qkv.log)

# Usage

## Setup
1. Install 3rd-party packages from [requirements.txt](./requirements.txt).

Notice that the version of timm should be equal or higher than **0.3.2**, and the version of Pytorch should be equal or higher than **1.7.0**.
```bash
pip install -r ./requirements.txt
```

2. **[Optional, Recommend]** Build iRPE operators implemented by CUDA.

Although iRPE can be implemented by PyTorch native functions, the backward speed of PyTorch index function is very slow. We implement CUDA operators for more efficient training and recommend to build it.
`nvcc` is necessary to build CUDA operators.
```bash
cd rpe_ops/
python setup.py install --user
```

## Data Preparation

You can download the ImageNet-1K dataset from [`http://www.image-net.org/`](http://www.image-net.org/).

The train set and validation set should be saved as the `*.tar` archives:

```
ImageNet/
├── train.tar
└── val.tar
```

Our code also supports storing images as individual files as follow:
```
ImageNet/
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
...
├── val
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
```

## Training
We define the models with iRPE in [`rpe_models.py`](./rpe_models.py).

For example, we train DeiT-S with contextual product relative position encoding on keys with 50 buckets, the model's name is `deit_small_patch16_224_ctx_product_50_shared_k`.

Run the following command:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224_ctx_product_50_shared_k --batch-size 128 --data-path ./ImageNet/ --output_dir ./outputs/ --load-tar
```

You can remove the flag `--load-tar` if storing images as individual files : )

## Evaluation
The step is similar to training. Add `--eval --resume <the checkpoint path>`.
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224_ctx_product_50_shared_k --batch-size 128 --data-path ./ImageNet/ --output_dir ./outputs/ --load-tar --eval --resume deit_small_patch16_224_ctx_product_50_shared_k.pth
```

## Code Structure

Our code is based on [DeiT](https://github.com/facebookresearch/deit) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models). Thank you!

File | Description
-----|------------
[`irpe.py`](./irpe.py) | The implementation of image relative position encoding
[`rpe_models.py`](./rpe_models.py) | The implementation of models with iRPE
[`rpe_vision_transformer.py`](./rpe_vision_transformer.py) | We equip iRPE on `Attention`, `Block`, and `VisionTransformer` modules
[`rpe_ops`](./rpe_ops) | The CUDA implementation of iRPE operators for efficient training

# Citing iRPE
If this project is helpful for you, please cite it. Thank you! : )

```bibtex
@article{iRPE,
  title={Rethinking and Improving Relative Position Encoding for Vision Transformer},
  author={Wu, Kan and Peng, Houwen and Chen, Minghao and Fu, Jianlong and Chao, Hongyang},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```

# License
[Apache License](./LICENSE)
