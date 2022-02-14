# CyDAS Detection Code Base

### Environments
- Python 3.7
- Pytorch>=1.8.2
- Torchvision == 0.9.2

You can directly run the code ```sh env.sh``` and ```sh compile.sh``` to setup the running environment.
We use 8 GPUs (24GB RTX 3090) to train our detector, you can adjust the batch size in configs by yourselves.

### Data Preparatoin

Your directory tree should be look like this:

````bash
$HitDet.pytorch/data
├── coco
│   ├── annotations
│   ├── train2017
│   └── val2017
│
├── VOCdevkit
│   ├── VOC2007
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   └── SegmentationObject
│   └── VOC2012
│       ├── Annotations
│       ├── ImageSets
│       ├── JPEGImages
│       ├── SegmentationClass
│       └── SegmentationObject
````

### Getting Start

Our pretrained backbone params can be found in [GoogleDrive](https://drive.google.com/drive/folders/1CkFp24bEDq0wUp504BQ68jn5Vs069qox)

Installation
* Clone this repo:
```bash
cd CDARTS_detection
```
* Install dependencies:
```bash
bash env.sh
bash compile.sh
```

Train:
```
sh train.sh
```

## Acknowledgement
Our code is based on the open source project [MMDetection](https://github.com/open-mmlab/mmdetection).
