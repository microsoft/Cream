# EfficientViT for Object Detection and Instance Segmentation

The codebase implements the object detection and instance segmentation framework with [MMDetection](https://github.com/open-mmlab/mmdetection), using EfficientViT as the backbone.

## Model Zoo

### RetinaNet Object Detection
|Model | Pretrain | Lr Schd | Box AP | AP@50 | AP@75 | Config | Link | 
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|EfficientViT-M4 | ImageNet-1k   | 1x | 32.7  | 52.2   | 34.1     | [config](./configs/retinanet_efficientvit_m4_fpn_1x_coco.py) | [model](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/retinanet_efficientvit_m4_fpn_1x_coco.pth)/[log](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/retinanet_efficientvit_m4_fpn_1x_coco_log.json) |


### Mask R-CNN Instance Segmentation
|Model | Pretrain | Lr Schd | Mask AP | AP@50 | AP@75 | Config | Link | 
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|EfficientViT-M4 | ImageNet-1k   |1x| 31.0 | 51.2 | 32.2      | [config](./configs/mask_rcnn_efficientvit_m4_fpn_1x_coco.py) | [model](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/mask_rcnn_efficientvit_m4_fpn_1x_coco.pth)/[log](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/mask_rcnn_efficientvit_m4_fpn_1x_coco_log.json) |

## Get Started

Please follow the following steps to setup EfficientViT for downstream tasks.

### Install requirements

Install [mmcv-full](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection) via [MIM](https://github.com/open-mmlab/mim):
```
pip install -U openmim
mim install mmcv-full
mim install mmdet
```

### Data preparation

Prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).
The dataset should be organized as 
```
downstream
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## Evaluation

Firstly, prepare the MSCOCO pretrained models by downloading from the [model-zoo](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo).

Below are the instructions for evaluating the models on MSCOCO 2017 val set:

<details>

<summary>
Object Detection
</summary>

To evaluate the RetinaNet model with EfficientViT_M4 as backbone, run:

```bash
bash ./dist_test.sh configs/retinanet_efficientvit_m4_fpn_1x_coco.py ./retinanet_efficientvit_m4_fpn_1x_coco.pth 8 --eval bbox
```

where 8 refers to the number of GPUs. For the usage of more arguments, please refer to [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#training-on-multiple-gpus).

</details>

<details>

<summary>
Instance Segmentation
</summary>

To evaluate the Mask R-CNN model with EfficientViT_M4 as backbone, run:

```bash
bash ./dist_test.sh configs/mask_rcnn_efficientvit_m4_fpn_1x_coco.py ./mask_rcnn_efficientvit_m4_fpn_1x_coco.pth 8 --eval bbox segm
```

where 8 refers to the number of GPUs. For the usage of more arguments, please refer to [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#training-on-multiple-gpus).

</details>


## Training

Firstly, prepare the ImageNet-1k pretrained EfficientViT-M4 model by downloading from the [model-zoo](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo).

Below are the instructions for training the models on MSCOCO 2017 train set:

<details>

<summary>
Object Detection
</summary>

To train the RetinaNet model with EfficientViT_M4 as backbone on a single machine using multi-GPUs, run:

```bash
bash ./dist_train.sh configs/retinanet_efficientvit_m4_fpn_1x_coco.py 8 --cfg-options model.backbone.pretrained=$PATH_TO_IMGNET_PRETRAIN_MODEL
```

where 8 refers to the number of GPUs. For the usage of more arguments, please refer to [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#training-on-multiple-gpus).

</details>


<details>

<summary>
Instance Segmentation
</summary>

To train the Mask R-CNN model with EfficientViT_M4 as backbone on a single machine using multi-GPUs, run:

```bash
bash ./dist_train.sh configs/mask_rcnn_efficientvit_m4_fpn_1x_coco.py 8 --cfg-options model.backbone.pretrained=$PATH_TO_IMGNET_PRETRAIN_MODEL
```

where 8 refers to the number of GPUs. For the usage of more arguments, please refer to [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#training-on-multiple-gpus).

</details>


## Acknowledge

The downstream task implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[MMDetection](https://github.com/open-mmlab/mmdetection), [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection), [PoolFormer](https://github.com/sail-sg/poolformer/tree/main/detection).
