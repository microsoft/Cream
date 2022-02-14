# Cyclic Differentiable Architecture Search

**This is an official implementation of CDARTS**

In this work, we propose new joint optimization objectives and a novel Cyclic Differentiable ARchiTecture Search framework, dubbed CDARTS. Considering the structure difference, CDARTS builds a cyclic feedback mechanism between the search and evaluation networks with introspective distillation. First, the search network generates an initial architecture for evaluation, and the weights of the evaluation network are optimized. Second, the architecture weights in the search network are further optimized by the label supervision in classification, as well as the regularization from the evaluation network through feature distillation. Repeating the above cycle results in a joint optimization of the search and evaluation networks and thus enables the evolution of the architecture to fit the final evaluation network.

<div align="center">
    <img width="50%" alt="CDARTS overview" src="demo/framework1.png"/>
</div>

## Model Zoo
For evaluation, we provide the checkpoints and configs of our models in [Google Drive](https://drive.google.com/drive/folders/1CkFp24bEDq0wUp504BQ68jn5Vs069qox?usp=sharing).

After downloading the models, you can do the evaluation following the description in *SETUP.md*.

Model download links:

### DARTS Search Space
#### CIFAR10
| Top-1 Acc. %       | 97.60                                                                                              | 97.45                                                                                              | 97.52                                                                                              | 97.53                                                                                              | 97.54                                                                                              | 97.77                                                                                              |
|--------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Cell Download link | [Cell-1](https://drive.google.com/file/d/1mlRQUo2DyiZvwfhVfkcjJRAaaxJLmkjs/view?usp=sharing) | [Cell-2](https://drive.google.com/file/d/1W-2uvAQZVTuWEDHEhvb_FHv_US9Pl6tS/view?usp=sharing) | [Cell-3](https://drive.google.com/file/d/12j6SwGAfE4_eKIBr38PSy9pyxMG_9avB/view?usp=sharing) | [Cell-4](https://drive.google.com/file/d/1muuQLTxFX7oKAd8hjsxwOzRx1hGYsCjP/view?usp=sharing) | [Cell-5](https://drive.google.com/file/d/1eBJjEldqfo3AsfPT5wemI46PdhtgQh5i/view?usp=sharing) | [Cell-6](https://drive.google.com/file/d/1nZ1XNOAb325-UZN-rbs17fzLfjKrhd-S/view?usp=sharing) |

#### ImageNet
| Top-1 Acc. %       | 75.90                                                                                              | 75.93                                                                                              | 76.40                                                                                              | 76.60                                                                                              | 76.44                                                                                              |
|--------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Cell Download link | [Cell-1](https://drive.google.com/file/d/1VY8MyWaDbrWQdi4xyKEcaX88ndH_o9XP/view?usp=sharing) | [Cell-2](https://drive.google.com/file/d/1nokqF1HaPrKbW0vkeN3mB9M5mHRI745U/view?usp=sharing) | [Cell-3](https://drive.google.com/file/d/1Rk8JbHAUUG5pE4t3AU94yfIsPsE5GTJH/view?usp=sharing) | [Cell-4](https://drive.google.com/file/d/1fgQk3o4svX8hoP__MK2qikokrDpn9rx7/view?usp=sharing) | [Cell-5](https://drive.google.com/file/d/12_TG4F0cnHc9lmsRiKK7TiN5CsyncBkx/view?usp=sharing) |

### NATS-Bench
| Model | CIFAR10 Validation | CIFAR10 Test | CIFAR100 Validation | CIFAR100 Test | ImageNet-16-120 Validation | ImageNet-16-120 Test | Download link |
|-------|--------------------|--------------|---------------------|---------------|----------------------------|----------------------|---------------|
| Cell-1 | 91.50%             | 94.37%       | 73.31%              | 73.09%        | 45.59%                     | 46.33%               | [Cell, Log](https://drive.google.com/file/d/13CpMr1V-S0d8C2WbIHwSzApmdBInKn0U/view?usp=sharing)             |
| Cell-2 | 91.37%             | 94.09%       | 72.64%              | 72.57%        | 45.46%                     | 45.63%               | [Cell, Log](https://drive.google.com/file/d/1Gbnm61NbYmEkdW6YCBUWDA_sQGtm83vR/view?usp=sharing)             |
| Cell-3 | 90.51%             | 93.62%       | 70.43               | 70.10%        | 44.23%                     | 44.57%               | [Cell, Log](https://drive.google.com/file/d/1zq2Eg8IZt5MVXFnmKuuCDS5bei-ENlWw/view?usp=sharing)             |

### Chain-structured Search Space

Model | Params. | Flops | Top-1 Acc. % | Download link 
--- |:---:|:---:|:---:|:---:
CDARTS-a | 7.0M | 294M | 77.4 | [Model, Config, Log](https://drive.google.com/drive/folders/146h42gj9yNhmOoJX87hTeHw5PvOIyGPi?usp=sharing)
CDARTS-b  | 6.4M | 394M | 78.2 | [Model, Config, Log](https://drive.google.com/drive/folders/1LUY9dfLIGSQicfoInHaffIepunujyuGe?usp=sharing)

### Object Detection
Backbone            | Input Size  | Params. | Flops | AP    | AP\_50 | AP\_75 | AP\_S | AP\_M | AP\_L | Download link
----------------------- | --------- | -------- | ----- | ----- | ------ | ------ | ----- | ----- | ----- | ---
CDARTS-a | 1280x800 |  6.0G  | 7.0M  | 35.2 | 55.5 | 37.5  | 19.8 | 38.7 | 47.5 | [Model, Config, Log](https://drive.google.com/drive/folders/1xkV_ZJXhHPkDbL1Ogc3AQJvi9HC1Egyn?usp=sharing)
CDARTS-b | 1280x800 |  8.1G  | 6.4M  | 36.2 | 56.7 | 38.3  | 20.9 | 39.8 | 48.5 | [Model, Config, Log](https://drive.google.com/drive/folders/1xkV_ZJXhHPkDbL1Ogc3AQJvi9HC1Egyn?usp=sharing)


### Semantic Segmentation
| Dataset    | Encoder  | Input Size  | Params. | Flops | mIoU % | Download link |
|------------|----------|-------------|---------|-------|--------|---------------|
| Cityscapes | CDARTS-b | 1024x2048 | 5.9M    | 20.7G | 78.1 | [Model, Config, Log](https://drive.google.com/drive/folders/1MO_hgwWcUf1c1OFYO-6Rit6IA-3fMKAO?usp=sharing) |
| ADE20K     | CDARTS-b | 640x640   | 2.7M    | 5.9G  | 40.4 | [Model, Config, Log](https://drive.google.com/drive/folders/1OJyNLkMMK1IIu1F3USrkNsa9BeLlGiUp?usp=sharing) |


## Bibtex

If this repo is helpful for you, please consider to cite it. Thank you! :)
```bibtex
@article{CDARTS,
  title={Cyclic Differentiable Architecture Search},
  author={Yu, Hongyuan and Peng, Houwen and Huang, Yan and Fu, Jianlong and Du, Hao and Wang, Liang and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2022}
}
```