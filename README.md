Hiring research interns for neural architecture search projects: houwen.peng@microsoft.com

# AutoML - Neural Architecture Search

***This is a collection of our AutoML-NAS work***

> [**iRPE**](./iRPE) (```NEW```): **Rethinking and Improving Relative Position Encoding for Vision Transformer**


> [**AutoFormer**](https://github.com/microsoft/AutoML/tree/main/AutoFormer) (```NEW```): **AutoFormer: Searching Transformers for Visual Recognition**


> [**Cream**](https://github.com/microsoft/AutoML/tree/main/Cream) (```@NeurIPS'20```): **Cream of the Crop: Distilling Prioritized Paths For One-Shot Neural Architecture Search**


## News
- :boom: July, 2021: [**iRPE code**](./iRPE) (**with CUDA Acceleration**) is now released. Paper is [here](https://houwenpeng.com/publications/iRPE.pdf).
- :boom: July, 2021: [**iRPE**](https://houwenpeng.com/publications/iRPE.pdf) has been accepted by ICCV'21.
- :boom: July, 2021: [**AutoFormer**](https://arxiv.org/abs/2107.00651) has been accepted by ICCV'21.
- :boom: July, 2021: [**AutoFormer**](https://github.com/microsoft/AutoML/tree/main/AutoFormer) is now available on [arXiv](https://arxiv.org/abs/2107.00651).
- :boom: Oct, 2020: Code for [**Cream**](https://github.com/microsoft/AutoML/tree/main/Cream) is now released.
- :boom: Oct, 2020: [**Cream**](https://github.com/microsoft/AutoML/tree/main/Cream) was accepted to NeurIPS'20

## Works

### [AutoFormer](https://github.com/microsoft/AutoML/tree/main/AutoFormer)

***Coming soon!!!***

AutoFormer is new one-shot architecture search framework dedicated to vision transformer search. It entangles the weights of different vision transformer blocks in the same layers during supernet training. 
Benefiting from the strategy, the trained supernet allows thousands of subnets to be very well-trained. Specifically, the performance of these subnets with weights inherited from the supernet is comparable to those retrained from scratch.
<div align="center">
    <img width="49%" alt="AutoFormer overview" src="https://github.com/microsoft/AutoML/releases/download/static_files/autoformer_overview.gif"/>
    <img width="49%" alt="AutoFormer detail" src="https://github.com/microsoft/AutoML/releases/download/static_files/autoformer_details.gif"/>
</div>

### [iRPE](./iRPE)
**Image RPE (iRPE for short) methods are new relative position encoding methods dedicated to 2D images**, considering directional relative distance modeling as well as the interactions between queries and relative position embeddings in self-attention mechanism. The proposed iRPE methods are simple and lightweight, being easily plugged into transformer blocks. Experiments demonstrate that solely due to the proposed encoding methods, **DeiT and DETR obtain up to 1.5% (top-1 Acc) and 1.3% (mAP) stable improvements** over their original versions on ImageNet and COCO respectively, without tuning any extra hyperparamters such as learning rate and weight decay. Our ablation and analysis also yield interesting findings, some of which run counter to previous understanding.
<div align="center">
    <img width="70%" alt="iRPE overview" src="iRPE/iRPE.png"/>
</div>


### [Cream](https://github.com/microsoft/AutoML/tree/main/Cream)
**[[Paper]](https://papers.nips.cc/paper/2020/file/d072677d210ac4c03ba046120f0802ec-Paper.pdf) [[Models-Google Drive]](https://drive.google.com/drive/folders/1NLGAbBF9bA1IUAxKlk2VjgRXhr6RHvRW?usp=sharing)[[Models-Baidu Disk (password: wqw6)]](https://pan.baidu.com/s/1TqQNm2s14oEdyNPimw3T9g) [[Slides]]() [[BibTex]](https://scholar.googleusercontent.com/scholar.bib?q=info:ICWVXc_SsKAJ:scholar.google.com/&output=citation&scisdr=CgUmooXfEMfTi0cV5aU:AAGBfm0AAAAAX7sQ_aXoamdKRaBI12tAVN8REq1VKNwM&scisig=AAGBfm0AAAAAX7sQ_RdYtp6BSro3zgbXVJU2MCgsG730&scisf=4&ct=citation&cd=-1&hl=ja)**  <br/>

In this work, we present a simple yet effective architecture distillation method. The central idea is that subnetworks can learn collaboratively and teach each other throughout the training process, aiming to boost the convergence of individual models. We introduce the concept of prioritized path, which refers to the architecture candidates exhibiting superior performance during training. Distilling knowledge from the prioritized paths is able to boost the training of subnetworks. Since the prioritized paths are changed on the fly depending on their performance and complexity, the final obtained paths are the cream of the crop.
<div >
    <img src="Cream/demo/intro.jpg" width="90%"/>
</div>


## Bibtex
```bibtex
@article{iRPE,
  title={Rethinking and Improving Relative Position Encoding for Vision Transformer},
  author={Wu, Kan and Peng, Houwen and Chen, Minghao and Fu, Jianlong and Chao, Hongyang},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}

@article{chen2021autoformer,
  title={AutoFormer: Searching Transformers for Visual Recognition},
  author={Chen, Minghao and Peng, Houwen and Fu, Jianlong and Ling, Haibin},
  journal={arXiv preprint arXiv:2107.00651},
  year={2021}
}

@article{Cream,
  title={Cream of the Crop: Distilling Prioritized Paths For One-Shot Neural Architecture Search},
  author={Peng, Houwen and Du, Hao and Yu, Hongyuan and Li, Qi and Liao, Jing and Fu, Jianlong},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## License
License under an MIT license.

