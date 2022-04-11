# Mini-Swin

This repo is for MiniViT for swin transformer.

# Usage

Create the environment:

```bash
pip install -r requirements.txt
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


### Mini-Swin-Tiny
```bash
python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/swin_tiny_patch4_window7_224_minivit_sharenum6.yaml --data-path <data-path>  --output <output-folder> --tag mini-swin-tiny --batch-size 128 --is_sep_layernorm --is_transform_heads --is_transform_ffn --do_distill --alpha 0.0 --teacher <teacher-path> --attn_loss --hidden_loss --hidden_relation --student_layer_list 11_9_7_5_3_1 --teacher_layer_list 23_21_15_9_3_1 --hidden_weight 0.1
```

### Mini-Swin-Small
```bash
python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/swin_small_patch4_window7_224_minivit_sharenum2.yaml --data-path <data-path>  --output <output-folder> --tag mini-swin-small --batch-size 128 --is_sep_layernorm --is_transform_heads --is_transform_ffn --do_distill --alpha 0.0 --teacher <teacher-path> --attn_loss --hidden_loss --hidden_relation --student_layer_list 23_21_15_9_3_1 --teacher_layer_list 23_21_15_9_3_1 --hidden_weight 0.1
```

### Mini-Swin-Base
```bash
python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/swin_base_patch4_window7_224_minivit_sharenum2.yaml --data-path <data-path>  --output <output-folder> --tag mini-swin-base --batch-size 128 --is_sep_layernorm --is_transform_heads --is_transform_ffn --do_distill --alpha 0.0 --teacher <teacher-path> --attn_loss --hidden_loss --hidden_relation --student_layer_list 23_21_15_9_3_1 --teacher_layer_list 23_21_15_9_3_1 --hidden_weight 0.1
```

### Finetune Mini-Swin-B with resolution 384:
```bash
python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/swin_base_patch4_window7_224to384_minivit_sharenum2_adamw.yaml --data-path <data-path>  --output <output-folder> --tag mini-swin-base-224to384 --batch-size 16 --accumulation-steps 2 --is_sep_layernorm --is_transform_heads --is_transform_ffn --resume <model-224-ckpt> --resume_weight_only --train_224to384
```

## Evaluation

Run the following commands for evaluation:

```bash
python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/swin_tiny_patch4_window7_224_minivit_sharenum6.yaml --data-path /sdb/imagenet --batch-size 64 --tag inference --is_sep_layernorm --is_transform_ffn --is_transform_heads --resume <checkpoint-path>/mini-swin-tiny-12m.pth
```
To evaluate other MiniViTs, simply replace `--cfg configs/swin_tiny_patch4_window7_224_minivit_sharenum6.yaml` and `--resume <checkpoint-path>/mini-swin-tiny-12m.pth` by corresponding YAML and checkpoint files.

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
Our code is based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer). Thank you!

[MIT License](./LICENSE)
