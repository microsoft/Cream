Hiring research interns for neural architecture search projects: houwen.peng@microsoft.com

# Rethinking and Improving Relative Position Encoding for Vision Transformer

[[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Rethinking_and_Improving_Relative_Position_Encoding_for_Vision_Transformer_ICCV_2021_paper.html)

Object Detection: DETR with iRPE

# Model Zoo

We equip DETR models with contextual product shared-head RPE, and report their mAP on MSCOCO dataset.

- Absolute Position Encoding: Sinusoid

- Relative Position Encoding: iRPE (contextual product shared-head RPE)

enc\_rpe2d              | Backbone  | #Buckets | epoch | AP    | AP\_50 | AP\_75 | AP\_S | AP\_M | AP\_L | Link | Log
----------------------- | --------- | -------- | ----- | ----- | ------ | ------ | ----- | ----- | ----- | ---- | ---
rpe-1.9-product-ctx-1-k | ResNet-50 |  7 x 7   | 150   | 0.409 | 0.614  | 0.429  | 0.195 | 0.443 | 0.605 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/rpe-1.9-product-ctx-1-k.pth)| [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_rpe-1.9-product-ctx-1-k.txt), [detail (188 MB)](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_rpe-1.9-product-ctx-1-k.log)
rpe-2.0-product-ctx-1-k | ResNet-50 |  9 x 9   | 150   | 0.410 | 0.615  | 0.434  | 0.192 | 0.445 | 0.608 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/rpe-2.0-product-ctx-1-k.pth)| [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_rpe-2.0-product-ctx-1-k.txt), [detail (188 MB)](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_rpe-2.0-product-ctx-1-k.log)
rpe-2.0-product-ctx-1-k | ResNet-50 |  9 x 9   | 300   | 0.422 | 0.623  | 0.446  | 0.205 | 0.457 | 0.613 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/rpe-2.0-product-ctx-1-k_300epochs.pth)| [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_rpe-2.0-product-ctx-1-k_300epochs.txt), [detail (375 MB)](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_rpe-2.0-product-ctx-1-k_300epochs.log)

`--enc_rpe2d` is an argument to represent the attributions of relative position encoding.


# Usage

## Setup
1. Install 3rd-party packages from [requirements.txt](./requirements.txt).

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

You can download the MSCOCO dataset from [`https://cocodataset.org/#download`](https://cocodataset.org/#download).

Please downlaod the following files:
- [2017 Train images [118K/18GB]](http://images.cocodataset.org/zips/train2017.zip)
- [2017 Val images [5K/1GB]](http://images.cocodataset.org/zips/val2017.zip)
- [2017 Train/Val annotations [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

After downloading them, move the three archieves into the same directory, then decompress the annotations archive by `unzip ./annotations_trainval2017.zip`. We **DO NOT** compress the images archieves.

The dataset should be saved as follow,
```
coco_data
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017.zip
└── val2017.zip
```

The zipfile `train2017.zip` and `val2017.zip` can also be decompressed.

```
coco_data
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017
│   └── 000000000009.jpg
└── val2017
│   └── 000000000009.jpg
```

## Argument for iRPE
We add an extra argument `--enc_rpe2d rpe-{ratio}-{method}-{mode}-{shared_head}-{rpe_on}` for iRPE. It means that we add relative position encoding on all the encoder layers.

Here is the format of the variables `ratio`, `method`, `mode`, `shared_head` and `rpe_on`.

```python
Parameters
----------
ratio: float
    The ratio to control the number of buckets.
    Example: 1.9, 2.0, 2.5, 3.0
    For the product method,

    ratio | The number of buckets
    ------|-----------------------
    1.9   | 7 x 7
    2.0   | 9 x 9
    2.5   | 11 x 11
    3.0   | 13 x 13

method: str
    The method name of image relative position encoding.
    Example: `euc` or `quant` or `cross` or `product`
    euc: Euclidean method
    quant: Quantization method
    cross: Cross method
    product: Product method
mode: str
    The mode of image relative position encoding.
    Example: `bias` or `ctx`
shared_head: bool
    Whether to share weight among different heads.
    Example: 0 or 1
    0: Do not share encoding weight among different heads.
    1: Share encoding weight among different heads.
rpe_on: str
    Where RPE attaches.
    "q": RPE on queries
    "k": RPE on keys
    "v": RPE on values
    "qk": RPE on queries and keys
    "qkv": RPE on queries, keys and values
```

If we want a image relative position encoding with contextual product shared-head `9 x 9` buckets, the argument is `--enc_rpe2d rpe-2.0-product-ctx-1-k`.

## Training
- Train a DETR-ResNet50 with iRPE (contextual product shared-head `9 x 9` buckets) for **150 epochs**:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --lr_drop 100 --epochs 150 --coco_path ./coco_data --enc_rpe2d rpe-2.0-product-ctx-1-k --output_dir ./output'
```

- Train a DETR-ResNet50 with iRPE (contextual product shared-head `9 x 9` buckets) for **300 epochs**:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --lr_drop 200 --epochs 300 --coco_path ./coco_data --enc_rpe2d rpe-2.0-product-ctx-1-k --output_dir ./output'
```

where `--nproc_per_node 8` means using 8 GPUs to train the model. `/coco_data` is the dataset folder, and `./output` is the model checkpoint folder.

## Evaluation
The step is similar to training. Add the checkpoint path and the flag `--eval --resume <the checkpoint path>`.
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --lr_drop 100 --epochs 150 --coco_path ./coco_data --enc_rpe2d rpe-2.0-product-ctx-1-k --output_dir ./output --eval --resume rpe-2.0-product-ctx-1-k.pth'
```

## Code Structure

Our code is based on [DETR](https://github.com/facebookresearch/detr). The implementation of `MultiheadAttention` is based on PyTorch native operator ([module](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py), [function](https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py)). Thank you!

File | Description
-----|------------
[`models/rpe_attention/irpe.py`](./models/rpe_attention/irpe.py) | The implementation of image relative position encoding
[`models/rpe_attention/multi_head_attention.py`](./models/rpe_attention/multi_head_attention.py) | The nn.Module `MultiheadAttention` with iRPE
[`models/rpe_attention/rpe_attention_function.py`](./models/rpe_attention/rpe_attention_function.py) | The function `rpe_multi_head_attention_forward` with iRPE
[`rpe_ops`](./rpe_ops) | The CUDA implementation of iRPE operators for efficient training

# Citing iRPE
If this project is helpful for you, please cite it. Thank you! : )

```bibtex
@InProceedings{iRPE,
    author    = {Wu, Kan and Peng, Houwen and Chen, Minghao and Fu, Jianlong and Chao, Hongyang},
    title     = {Rethinking and Improving Relative Position Encoding for Vision Transformer},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {10033-10041}
}
```

# License
[Apache License](./LICENSE)
