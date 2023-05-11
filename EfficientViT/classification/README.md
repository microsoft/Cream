# EfficientViT for Image Classification

The codebase implements the image classification with EfficientViT.

## Model Zoo

|Model | Data | Input | Acc@1 | Acc@5 | #FLOPs | #Params | Throughput | Link | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|EfficientViT-M0 | ImageNet-1k   |224x224| 63.2 | 85.2  | 79   | 2.3M     | 27644 | [model](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m0.pth)/[log](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m0_log.txt)/[onnx](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/EfficientViT_M0.onnx) |
|EfficientViT-M1 | ImageNet-1k   |224x224| 68.4 | 88.7 | 167  | 3.0M     | 20093 | [model](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m1.pth)/[log](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m1_log.txt)/[onnx](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/EfficientViT_M1.onnx)|
|EfficientViT-M2 | ImageNet-1k   |224x224| 70.8 | 90.2 | 201  | 4.2M     | 18218 | [model](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m2.pth)/[log](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m2_log.txt)/[onnx](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/EfficientViT_M2.onnx)|
|EfficientViT-M3 | ImageNet-1k   |224x224| 73.4 | 91.4 | 263  | 6.9M     | 16644 | [model](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m3.pth)/[log](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m3_log.txt)/[onnx](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/EfficientViT_M3.onnx)  |
|EfficientViT-M4 | ImageNet-1k   |224x224| 74.3 | 91.8 | 299  | 8.8M     | 15914 | [model](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m4.pth)/[log](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m4_log.txt)/[onnx](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/EfficientViT_M4.onnx)  |
|EfficientViT-M5 | ImageNet-1k   |224x224| 77.1 | 93.4 | 522  | 12.4M    | 10621 | [model](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m5.pth)/[log](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m5_log.txt)/[onnx](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/EfficientViT_M5.onnx) |

## Get Started


### Install requirements

Run the following command to install the dependences:

```bash
pip install -r requirements.txt
```

### Data preparation

We need to prepare ImageNet-1k dataset from [`http://www.image-net.org/`](http://www.image-net.org/).

- ImageNet-1k

ImageNet-1k contains 1.28 M images for training and 50 K images for validation.
The images shall be stored as individual files:

```
ImageNet/
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
...
├── val
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
...
```

Our code also supports storing the train set and validation set as the `*.tar` archives:

```
ImageNet/
├── train.tar
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
...
└── val.tar
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
...
```

## Evaluation

Before evaluation, we need to prepare the pre-trained models from [model-zoo](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo).

Run the following command to evaluate a pre-trained EfficientViT-M4 on ImageNet val with a single GPU:
```bash
python main.py --eval --model EfficientViT_M4 --resume ./efficientvit_m4.pth --data-path $PATH_TO_IMAGENET
```

This should give
```
* Acc@1 74.266 Acc@5 91.788 loss 1.242
```

Here are the command lines for evaluating other pre-trained models:
<details>

<summary>
EfficientViT-M0
</summary>

```bash
python main.py --eval --model EfficientViT_M0 --resume ./efficientvit_m0.pth --data-path $PATH_TO_IMAGENET
```
giving
```
* Acc@1 63.296 Acc@5 85.150 loss 1.741
```

</details>
<details>

<summary>
EfficientViT-M1
</summary>

```bash
python main.py --eval --model EfficientViT_M1 --resume ./efficientvit_m1.pth --data-path $PATH_TO_IMAGENET
```
giving
```
* Acc@1 68.356 Acc@5 88.672 loss 1.513
```

</details>
<details>

<summary>
EfficientViT-M2
</summary>

```bash
python main.py --eval --model EfficientViT_M2 --resume ./efficientvit_m2.pth --data-path $PATH_TO_IMAGENET
```
giving
```
* Acc@1 70.786 Acc@5 90.150 loss 1.442
```

</details>
<details>

<summary>
EfficientViT-M3
</summary>

```bash
python main.py --eval --model EfficientViT_M3 --resume ./efficientvit_m3.pth --data-path $PATH_TO_IMAGENET
```
giving
```
* Acc@1 73.390 Acc@5 91.350 loss 1.285
```

</details>

<details>

<summary>
EfficientViT-M5
</summary>

```bash
python main.py --eval --model EfficientViT_M5 --resume ./efficientvit_m5.pth --data-path $PATH_TO_IMAGENET
```
giving
```
* Acc@1 77.124 Acc@5 93.360 loss 1.127
```

</details>


## Training

To train an EfficientViT-M4 model on a single node with 8 GPUs for 300 epochs and distributed evaluation, run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model EfficientViT_M4 --data-path $PATH_TO_IMAGENET --dist-eval
```

<details>

<summary>
EfficientViT-M0
</summary>

To train an EfficientViT-M0 model on a single node with 8 GPUs for 300 epochs and distributed evaluation, run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model EfficientViT_M0 --data-path $PATH_TO_IMAGENET --dist-eval
```

</details>
<details>

<summary>
EfficientViT-M1
</summary>

To train an EfficientViT-M1 model on a single node with 8 GPUs for 300 epochs and distributed evaluation, run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model EfficientViT_M1 --data-path $PATH_TO_IMAGENET --dist-eval
```

</details>
<details>

<summary>
EfficientViT-M2
</summary>

To train an EfficientViT-M2 model on a single node with 8 GPUs for 300 epochs and distributed evaluation, run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model EfficientViT_M2 --data-path $PATH_TO_IMAGENET --dist-eval
```

</details>
<details>

<summary>
EfficientViT-M3
</summary>

To train an EfficientViT-M3 model on a single node with 8 GPUs for 300 epochs and distributed evaluation, run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model EfficientViT_M3 --data-path $PATH_TO_IMAGENET --dist-eval
```

</details>

<details>

<summary>
EfficientViT-M5
</summary>

To train an EfficientViT-M5 model on a single node with 8 GPUs for 300 epochs and distributed evaluation, run: 

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model EfficientViT_M5 --data-path $PATH_TO_IMAGENET --dist-eval
```

</details>

## Speed test

Run the following command to compare the throughputs on GPU/CPU:

```bash
python speed_test.py
```

which should give 
```
EfficientViT_M0 cuda:0 27643.941865437002 images/s @ batch size 2048
EfficientViT_M1 cuda:0 20093.286204638334 images/s @ batch size 2048
EfficientViT_M2 cuda:0 18218.347390415714 images/s @ batch size 2048
EfficientViT_M3 cuda:0 16643.905520424512 images/s @ batch size 2048
EfficientViT_M4 cuda:0 15914.449955135608 images/s @ batch size 2048
EfficientViT_M5 cuda:0 10620.868156518267 images/s @ batch size 2048
```

## Acknowledge

We sincerely appreciate [Swin Transformer](https://github.com/microsoft/swin-transformer), [LeViT](https://github.com/facebookresearch/LeViT), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), and [PyTorch](https://github.com/pytorch/pytorch) for their awesome codebases.

## License

- [License](./LICENSE)