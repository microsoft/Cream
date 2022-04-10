# Mini-DeiT

This repo is for MiniViT for DeiTs

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

Training Mini-DeiT-Ti:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model mini_deit_tiny_patch16_224 --batch-size 128 --data-path <data-path> --output_dir ./outputs  --teacher-path <teacher-model-path> --distillation-type soft --distillation-alpha 1.0 --drop-path 0.0
```

Training Mini-DeiT-S:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model mini_deit_small_patch16_224 --batch-size 128 --data-path <data-path> --output_dir ./outputs  --teacher-path <teacher-model-path> --distillation-type soft --distillation-alpha 1.0 --drop-path 0.0
```

Training Mini-DeiT-B:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model mini_deit_base_patch16_224 --batch-size 128 --data-path <data-path> --output_dir ./outputs  --teacher-path <teacher-model-path> --distillation-type soft --distillation-alpha 1.0
```

Finetune Mini-DeiT-B with resolution 384:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model mini_deit_base_patch16_384 --batch-size 128 --data-path <data-path> --output_dir ./outputs --finetune release_checkpoints/mini_deit_base_patch16_224.pth --input-size 384 --lr 5e-6 --min-lr 5e-6 --weight-decay 1e-8 --epochs 30
```

## Evaluation

Run the following commands for evaluation:

```bash
sh eval.sh
```

# License
Our code is based on [DeiT](https://github.com/facebookresearch/deit). Thank you!

[Apache License](./LICENSE)
