# AutoFormer: Searching Transformers for Visual Recognition

**This is an official implementation of AutoFormer.**

AutoFormer is new one-shot architecture search framework dedicated to vision transformer search. It entangles the weights of different vision transformer blocks in the same layers during supernet training. 
Benefiting from the strategy, the trained supernet allows thousands of subnets to be very well-trained. Specifically, the performance of these subnets with weights inherited from the supernet is comparable to those retrained from scratch.

<div align="center">
    <img width="49%" alt="AutoFormer overview" src="https://github.com/microsoft/AutoML/releases/download/static_files/autoformer_overview.gif"/>
    <img width="49%" alt="AutoFormer detail" src="https://github.com/microsoft/AutoML/releases/download/static_files/autoformer_details.gif"/>
</div>


## Highlights
- Once-for-all

AutoFormer is a simple yet effective method to train a once-for-all vision transformer supernet.

- Competive performance

AutoFormers consistently outperform DeiTs.

## Environment Setup

To set up the enviroment you can easily run the following command:
```buildoutcfg
conda create -n Autoformer python=3.6
conda activate Autoformer
pip install -r requirements.txt
```

## Data Preparation 
You need to first download the [ImageNet-2012](http://www.image-net.org/) to the folder `./data/imagenet` and move the validation set to the subfolder `./data/imagenet/val`. To move the validation set, you cloud use the following script: <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh>

The directory structure is the standard layout as following.
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```


## Model Zoo
For evaluation, we provide the checkpoints of our models in [Google Drive](https://drive.google.com/drive/folders/1HqzY3afqQUMI6pJ5_BgR2RquJU_b_3eg?usp=sharing) and [GitHub](https://github.com/silent-chen/AutoFormer-model-zoo).

After downloading the models, you can do the evaluation following the description in *Quick Start - Test*).

Model download links:

Model | Params. | Top-1 Acc. % | Top-5 Acc. % | Download link 
--- |:---:|:---:|:---:|:---:
AutoFormer-T | 5.8M | 75.3 | 92.7 | [Google Drive](https://drive.google.com/file/d/1uRCW3doQHgn2H-LjyalYEZ4CvmnQtr6Q/view?usp=sharing), [GitHub](https://github.com/silent-chen/AutoFormer-model-zoo/releases/download/v1.0/supernet-tiny.pth)
AutoFormer-S | 22.9M | 81.7 | 95.7 | [Google Drive](https://drive.google.com/file/d/1JTBmLR_nW7-ZbTKafWFvSl8J2orJXiNa/view?usp=sharing), [GitHub](https://github.com/silent-chen/AutoFormer-model-zoo/releases/download/v1.0/supernet-small.pth)
AutoFormer-B | 53.7M | 82.4 | 95.7 | [Google Drive](https://drive.google.com/file/d/1KPjUshk0SbqkaTzlirjPHM9pu19N5w0e/view?usp=sharing), [GitHub](https://github.com/silent-chen/AutoFormer-model-zoo/releases/download/v1.0/supernet-base.pth)


## Quick Start
We provide *Supernet Train, Search, Test* code of AutoFormer as follows.

### Supernet Train 

To train the supernet-T/S/B, we provided the corresponding supernet configuration files in `/experiments/supernet/`. For example, to train the supernet-B, you can run the following command. The default output path is `./`, you can specify the path with argument `--output`.

```buildoutcfg
python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /PATH/TO/IMAGENT --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --epochs 500 --warmup-epochs 20 \
--output /OUTPUT_PATH --batch-size 128
```

### Search
We run our evolution search on part of the ImageNet training dataset and use the validation set of ImageNet as the test set for fair comparison. To generate the subImagenet in `/PATH/TO/IMAGENET`, you could simply run:
```buildoutcfg
python ./lib/subImageNet.py --data-path /PATH/TO/IMAGENT
```
 

After obtaining the subImageNet and training of the supernet. We could perform the evolution search using below command. Please remember to config the specific constraint in this evolution search using `--min-param-limits` and `--param-limits`: 
```buildoutcfg
python -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path /PATH/TO/IMAGENT --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --resume /PATH/TO/CHECKPOINT \
--min-param-limits YOUR/CONFIG --param-limits YOUR/CONFIG --data-set EVO_IMNET
```

### Test
To test our trained models, you need to put the downloaded model in `/PATH/TO/CHECKPOINT`. After that you could use the following command to test the model (Please change your config file and model checkpoint according to different models. Here we use the AutoFormer-B as an example).
```buildoutcfg
python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /PATH/TO/IMAGENT --gp \
--change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/subnet/AutoFormer-B.yaml --resume /PATH/TO/CHECKPOINT --eval 
```

## Performance

**Left:** Top-1 accuracy on ImageNet. Our method achieves very competitive performance, being superior to the recent DeiT and ViT. **Right:** 1000 random sampled good architectures in the supernet-S. The supernet trained under our strategy allows subnets to be well optimized.

<div align="half">
    <img src=".figure/performance.png" width="49%"/>
    <img src=".figure/ofa.png" width="49%"/>
</div>

## Bibtex

If this repo is helpful for you, please consider to cite it. Thank you! :)
```bibtex
@InProceedings{AutoFormer,
    title     = {AutoFormer: Searching Transformers for Visual Recognition},
    author    = {Chen, Minghao and Peng, Houwen and Fu, Jianlong and Ling, Haibin},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {12270-12280}
}
```

## Acknowledgements

The codes are inspired by [HAT](https://github.com/mit-han-lab/hardware-aware-transformers), [timm](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit), [SPOS](https://github.com/megvii-model/SinglePathOneShot).
