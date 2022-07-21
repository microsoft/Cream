# Image Augmentation for TinyViT

The code is based on [timm.data](https://github.com/rwightman/pytorch-image-models/tree/master/timm/data) of [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) written by [Ross Wightman](https://github.com/rwightman) and the contributors. Thanks a lot!

We adapt it for TinyViT.

Apache License

## Code Structure

File                                         | Description
---------------------------------------------|--------------------------
[`aug_random.py`](./aug_random.py)           | unify all random values of augmentation with a random generator
[`dataset_wrapper.py`](./dataset_wrapper.py) | a dataset wrapper for saving logits
[`manager.py`](./manager.py)                 | The writter and reader for logits files
