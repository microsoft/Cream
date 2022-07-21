# Preparation

### Install the requirements

Run the following command to install the dependences:

```bash
pip install -r requirements.txt
```

### Data Preparation

We need to prepare ImageNet-1k and ImageNet-22k datasets from [`http://www.image-net.org/`](http://www.image-net.org/).

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

- ImageNet-22k

ImageNet-22k contains 14 M images with 21,841 classes, without overlapping part with ImageNet-1k validation set.

The filelist (`in22k_image_names.txt`) can be download at [here](https://github.com/wkcn/TinyViT-model-zoo/releases/download/datasets/in22k_image_names.txt).

Each class is stored as an archive file.
```
ImageNet-22k/
├── in22k_image_names.txt
├── n00004475.zip
├── n00005787.zip
├── n00006024.zip
...
├── n15102455.zip
└── n15102894.zip
```

The config `DATA.FNAME_FORMAT` defines the image filename format in the archive file, default: `{}.jpeg`.

We need IN-1k to evaluate the model, so the folders should be placed like the following (`a soft link is available`):
```
datasets/
├── ImageNet-22k/  # the folder of IN-22k
└── ImageNet/  # the folder of IN-1k
```
