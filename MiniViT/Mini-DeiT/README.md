# Mini-DeiT

This repo is for MiniViT for DeiTs.

## Model Zoo
Model | Params. | Input | Top-1 Acc. % | Top-5 Acc. % | Download link
--- |:---:|:---:|:---:|:---:|:---:
Mini-DeiT-Ti | 3M | 224x224 | 73.0 | 91.6 | [model](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini_deit_tiny_patch16_224.pth), [log](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/log_mini_deit_tiny.txt)
Mini-DeiT-S | 11M | 224x224 | 80.9 | 95.6 | [model](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini_deit_small_patch16_224.pth), [log](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/log_mini_deit_small.txt)
Mini-DeiT-B | 44M | 224x224 | 83.2 | 96.5 | [model](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini_deit_base_patch16_224.pth), [log](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/log_mini_deit_base.txt)
Mini-DeiT-B| 44M | 384x384 | 84.9 | 97.2 | [model](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini_deit_base_patch16_384.pth), [log](https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/log_mini_deit_base_384.txt)


# Usage

Create the environment:

```bash
pip install -r requirements.txt
```

Compile operations:
```bash
cd rpe_ops
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

Training Mini-DeiT-Ti

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model mini_deit_tiny_patch16_224 --batch-size 128 --data-path ./ImageNet --output_dir ./outputs  --teacher-path <teacher-model-path> --distillation-type soft --distillation-alpha 1.0 --drop-path 0.0
```

<details>
<summary>Training Mini-DeiT-S</summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model mini_deit_small_patch16_224 --batch-size 128 --data-path ./ImageNet --output_dir ./outputs  --teacher-path <teacher-model-path> --distillation-type soft --distillation-alpha 1.0 --drop-path 0.0
</code></pre>
</details>

<details>
<summary>Training Mini-DeiT-B</summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model mini_deit_base_patch16_224 --batch-size 128 --data-path ./ImageNet --output_dir ./outputs  --teacher-path <teacher-model-path> --distillation-type soft --distillation-alpha 1.0
</code></pre>
</details>

<details>
<summary>Finetune Mini-DeiT-B with resolution 384</summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model mini_deit_base_patch16_384 --batch-size 32 --data-path ./ImageNet --output_dir ./outputs --finetune checkpoints/mini_deit_base_patch16_224.pth --input-size 384 --lr 5e-6 --min-lr 5e-6 --weight-decay 1e-8 --epochs 30
</code></pre>
</details>

## Evaluation

Run the following commands for evaluation:

Evaluate Mini-DeiT-Ti
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model mini_deit_tiny_patch16_224 --batch-size 128 --data-path ./ImageNet --output_dir ./outputs  --resume ./checkpoints/mini_deit_tiny_patch16_224.pth --eval
```

<details>
<summary>Evaluate Mini-DeiT-S</summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model mini_deit_small_patch16_224 --batch-size 128 --data-path ./ImageNet --output_dir ./outputs  --resume ./checkpoints/mini_deit_small_patch16_224.pth --eval
</code></pre>
</details>

<details>
<summary>Evaluate Mini-DeiT-B</summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model mini_deit_base_patch16_224 --batch-size 128 --data-path ./ImageNet --output_dir ./outputs  --resume ./checkpoints/mini_deit_base_patch16_224.pth --eval
</code></pre>
</details>

<details>
<summary>Evaluate Mini-DeiT-B-384</summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model mini_deit_base_patch16_384 --batch-size 32 --data-path ./ImageNet --output_dir ./outputs  --resume ./checkpoints/mini_deit_base_patch16_384.pth --input-size 384 --eval
</code></pre>
</details>

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

# License
Our code is based on [DeiT](https://github.com/facebookresearch/deit). Thank you!

[Apache License](./LICENSE)
