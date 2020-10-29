Hiring research interns for neural architecture search projects: houwen.peng@microsoft.com

# Cream of the Crop: Distilling Prioritized Paths For One-Shot Neural Architecture Search 
This is an official implementation for our Cream NAS work presented in NeurIPS'20.

**[[Paper]]() [[Models-Google Drive]](https://drive.google.com/drive/folders/1NLGAbBF9bA1IUAxKlk2VjgRXhr6RHvRW?usp=sharing)[[Models-Baidu Disk]](https://pan.baidu.com/s/4hymmwni) [[Results]]() [[Slides]]() [[BibTex]]()**  <br/>

In this work, we present a simple yet effective architecture distillation method. The central idea is that subnetworks can learn collaboratively and teach each other throughout the training process, aiming to boost the convergence of individual models. We introduce the concept of prioritized path, which refers to the architecture candidates exhibiting superior performance during training. Distilling knowledge from the prioritized paths is able to boost the training of subnetworks. Since the prioritized paths are changed on the fly depending on their performance and complexity, the final obtained paths are the cream of the crop.
<div >
    <img src="./demo/intro.jpg" width="800"/>
</div>

## Environment Setup

To set up the enviroment you can easily run the following command:
```buildoutcfg
git clone https://github.com/mapleam/Cream.git
cd Cream
conda create -n Cream python=3.6
conda activate Cream
pip install -r requirements

# (optional) install apex to accelerate the training, a little bit faster than pytorch DistributedDataParallel
cd lib
git clone https://github.com/NVIDIA/apex.git
python ./apex/setup.py install --cpp_ext --cuda_ext
```

## Data Preparation 
You need to first download the [ImageNet-2012](http://www.image-net.org/) to the folder `./data/imagenet` and move the validation set to the subfolder `./data/imagenet/val`. To move the validation set, you cloud use the following script: <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh> *[todo]*

Put the imagenet data in ./data. It should look like:
```buildoutcfg
./data/imagenet/train
./data/imagenet/val
...
```

## Checkpoints For Test
For quick test, we have stored the checkpoints of our models in [Google Drive](https://drive.google.com/drive/folders/1NLGAbBF9bA1IUAxKlk2VjgRXhr6RHvRW?usp=sharing) (or [Baidu Disk](https://pan.baidu.com/s/4hymmwni)).

Just download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1NLGAbBF9bA1IUAxKlk2VjgRXhr6RHvRW?usp=sharing) (or [Baidu Disk](https://pan.baidu.com/s/4hymmwni)) and put the checkpoints in `./experiments/workspace/ckps/`.

Then you can do the test right now(as described in the following *Quick Start, III. Test*).

## Quick Start
We provide *search*, *retrain* and *test* code of our Cream NAS algorithm as follows.

### I. Search
To search for an architecture, you need to configure the parameters `FLOPS_MINIMUM` and `FLOPS_MAXIMUM` to specify the desired model Flops, for example, pick up a value from 0 to 600 MB Flops. You can specify the Flops interval [`FLOPS_MINIMUM`, `FLOPS_MAXIMUM`] in the YAML configure file: `./experiments/configs/train/train.yaml`.
```buildoutcfg
Line 51：FLOPS_MINIMUM: 0 # Minimum Flops of Architecture
Line 52：FLOPS_MAXIMUM: 600 # Maximum Flops of Architecture
```

After you specify the Flops of the architectures, you can search an architecture now by running:
```buildoutcfg
python ./tools/main.py train ./experiments/configs/train/train.yaml
```

### II. Retrain 
We also give the architecture we searched. To train those architectures, you need to configure the parameter `MODEL_SELECTION` to specify the model Flops. To specify which model to train, you should add `MODEL_SELECTION` in `./experiments/configs/retrain/retrain.yaml`. You can select one from [14,42,112,285,470,600], which stands for different Flops(MB).
```buildoutcfg
MODEL_SELECTION: 42 # Retrain 42m model
MODEL_SELECTION: 470 # Retrain 470m model
......
```

After specifying the Flops, you need to choose the config settings in `./experiments/configs/retrain/retrain.yaml`. The config files are in `./expperiments/configs/retrain`
```buildoutcfg
./experiments/configs/retrain/42.yaml
./experiments/configs/retrain/470.yaml
......
```

After adding `MODEL_SELECTION` in `retrain.yaml`, you need to use the following command to train the model.
```buildoutcfg
python ./tools/main.py retrain ./experiments/configs/retrain/retrain.yaml
```

The trained model and log file will be saved in `./experiments/workspace/retrain`. You can configure the `SAVE_PATH` in `./experiments/configs/retrain/retrain.yaml` to specify a path for saving the model and log file.

### III. Test
To test our trained of models, you need to use `MODEL_SELECTION` in `./experiments/configs/test/test.yaml` to specify which model to test.
```buildoutcfg
MODEL_SELECTION: 42 # test 42m model
MODEL_SELECTION: 470 # test 470m model
......
```

After specifying the Flops of the model, you need to write the path to the resume model in `./experiments/configs/test/test.yaml`.
```buildoutcfg
RESUME_PATH: './experiments/workspace/ckps/42.pth.tar'
RESUME_PATH: './experiments/workspace/ckps/470.pth.tar'
......
```

After adding `MODEL_SELECTION` and `RESUME_PATH` in `./experiments/configs/test/test.yaml`, you need to use the following command to test the model.
```buildoutcfg
python ./tools/main.py test ./experiments/configs/test/test.yaml
```

The test result will be saved in `./experiments/workspace/test`. You can configure the `SAVE_PATH` in `./experiments/configs/test/test.yaml` to specify a path for it.

##### Test Rank Correlation 

To perform a correlation analysis, we randomly sample 30 subnetworks from the hypernetwork and calculate the rank correlation between the weight sharing performance and the true performance of training from scratch. Unfortunately, training these subnetworks on ImageNet is very computational expensive, we thus construct a subImageNet dataset, which only consists of 100 classes randomly sampled from ImageNet. Each class has 250 training images and 50 validation images. We can generate the imagelist by running the following script: 

```buildoutcfg
python ./tools/generate_subImageNet.py
```
Thus we get the subImageNet in `./data/subImagnet`. The class list is provided in `./data/subclass_list.txt`. The images list is provided in `./data/subimages_list.txt`


## Performance
The top-1 accuracy on ImageNet. Our method achieves very competitive performance, being superior to the recent MobileNetV3 and EfficientNet-B0/B1.

<div align="half">
    <img src="./demo/results_600.jpg" width="400"/>
    <img src="./demo/results_100.jpg" width="400"/>
</div>

## BibTex

## License
License under an MIT license.

