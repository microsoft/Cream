# Preparation

### Install the dependencies
```bash
pip install -r requirements-training.txt
pip install -v -e .
```

### Data Preparation

We need to prepare [ImageNet-1k](http://www.image-net.org/) datasets to do zero-shot classification task.

- ImageNet-1k

ImageNet-1k contains 1.28 M images for training and 50 K images for validation.
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
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
...
├── val
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
```
