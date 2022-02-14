## Prerequisites
- Ubuntu 16.04
- Python 3.7
- CUDA 11.1 (lower versions may work but were not tested)
- NVIDIA GPU (>= 11G graphic memory) + CuDNN v7.3

This repository has been tested on RTX 3090. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

## Installation
* Clone this repo:
```bash
cd CDARTS_segmentation
```
* Install dependencies:
```bash
bash install.sh
```

## Usage
### 0. Prepare the dataset
* Download the [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3) and [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1) from the Cityscapes.
* Prepare the annotations by using the [createTrainIdLabelImgs.py](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py).
* Put the [file of image list](tools/datasets/cityscapes/) into where you save the dataset.


### 1. Train from scratch
* `cd HRTNet/train`
* Set the dataset path via  `ln -s $YOUR_DATA_PATH ../DATASET`
* Set the output path via `mkdir ../OUTPUT`
* Train from scratch
```
export DETECTRON2_DATASETS="$Your_DATA_PATH"
NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --world_size $NGPUS --seed 12367 --config ../configs/cityscapes/cydas.yaml
```

### 2. Evaluation
We provide training models and logs, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1CkFp24bEDq0wUp504BQ68jn5Vs069qox?usp=sharing).

```bash
cd train
```
* Download the pretrained weights of the from [Google Drive](https://drive.google.com/drive/folders/1CkFp24bEDq0wUp504BQ68jn5Vs069qox?usp=sharing).
* Set `config.model_path = $YOUR_MODEL_PATH` in `cydas.yaml`.
* Set `config.json_file = $CDARTS_MODEL` in `cydas.yaml`.
* Start the evaluation process:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py