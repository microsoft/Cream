
# Neural Architecture Design and Search [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=A%20new%20collection%20of%20tiny%20and%20efficient%20models%20thru%20architecture%20design%20and%20search,%20SOTA%20performance!!&url=https://github.com/microsoft/Cream&via=houwen_peng&hashtags=NAS,ViT,vision_transformer)

***This is a collection of our NAS and Vision Transformer work***

> [**TinyCLIP**](./TinyCLIP) (```@ICCV'23```): **TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance**

> [**EfficientViT**](./EfficientViT) (```@CVPR'23```): **EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention**

> [**TinyViT**](./TinyViT) (```@ECCV'22```): **TinyViT: Fast Pretraining Distillation for Small Vision Transformers**

> [**MiniViT**](./MiniViT) (```@CVPR'22```): **MiniViT: Compressing Vision Transformers with Weight Multiplexing**

> [**CDARTS**](./CDARTS) (```@TPAMI'22```): **Cyclic Differentiable Architecture Search**


> [**AutoFormerV2**](./AutoFormerV2) (```@NeurIPS'21```): **Searching the Search Space of Vision Transformer**


> [**iRPE**](./iRPE) (```@ICCV'21```): **Rethinking and Improving Relative Position Encoding for Vision Transformer**


> [**AutoFormer**](./AutoFormer) (```@ICCV'21```): **AutoFormer: Searching Transformers for Visual Recognition**


> [**Cream**](./Cream) (```@NeurIPS'20```): **Cream of the Crop: Distilling Prioritized Paths For One-Shot Neural Architecture Search**

We also implemented our NAS algorithms on Microsoft [**NNI**](https://github.com/microsoft/nni) (Neural Network Intelligence).

## News
- :sunny: Hiring research interns for next-generation model design, efficient large model inference: houwen.peng@microsoft.com
- :boom: Sep, 2023: Code for [**TinyCLIP**](./TinyCLIP) is now released.
- :boom: Jul, 2023: [**TinyCLIP**](./TinyCLIP) accepted to ICCV'23
- :boom: May, 2023: Code for [**EfficientViT**](./EfficientViT) is now released.
- :boom: Mar, 2023: [**EfficientViT**](./EfficientViT) accepted to CVPR'23
- :boom: Jul, 2022: Code for [**TinyViT**](./TinyViT) is now released.
- :boom: Apr, 2022: Code for [**MiniViT**](./MiniViT) is now released.
- :boom: Mar, 2022: [**MiniViT**](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_MiniViT_Compressing_Vision_Transformers_With_Weight_Multiplexing_CVPR_2022_paper.html) has been accepted by CVPR'22.
- :boom: Feb, 2022: Code for [**CDARTS**](./CDARTS) is now released.
- :boom: Feb, 2022: [**CDARTS**](./CDARTS) has been accepted by TPAMI'22.
- :boom: Jan, 2022: Code for [**AutoFormerV2**](./AutoFormerV2) is now released.
- :boom: Oct, 2021: [**AutoFormerV2**](./AutoFormerV2) has been accepted by NeurIPS'21, code will be released soon.
- :boom: Aug, 2021: Code for [**AutoFormer**](./AutoFormer) is now released.
- :boom: July, 2021: [**iRPE code**](./iRPE) (**with CUDA Acceleration**) is now released. Paper is [here](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Rethinking_and_Improving_Relative_Position_Encoding_for_Vision_Transformer_ICCV_2021_paper.html).
- :boom: July, 2021: [**iRPE**](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Rethinking_and_Improving_Relative_Position_Encoding_for_Vision_Transformer_ICCV_2021_paper.html) has been accepted by ICCV'21.
- :boom: July, 2021: [**AutoFormer**](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_AutoFormer_Searching_Transformers_for_Visual_Recognition_ICCV_2021_paper.html) has been accepted by ICCV'21.
- :boom: July, 2021: [**AutoFormer**](./AutoFormer) is now available on [arXiv](https://arxiv.org/abs/2107.00651).
- :boom: Oct, 2020: Code for [**Cream**](./Cream) is now released.
- :boom: Oct, 2020: [**Cream**](./Cream) was accepted to NeurIPS'20

## Works

### [TinyCLIP](./TinyCLIP)
**TinyCLIP** is a novel **cross-modal distillation** method for large-scale language-image pre-trained models. The method introduces two core techniques: **affinity mimicking** and **weight inheritance**. This work unleashes the capacity of small CLIP models, fully leveraging large-scale models as well as pre-training data and striking the best trade-off between speed and accuracy.
<div align="center">
    <img width="85%" alt="TinyCLIP overview" src="./TinyCLIP/figure/TinyCLIP.jpg"/>
</div>

### [EfficientViT](./EfficientViT)
**EfficientViT** is a family of high-speed vision transformers. It is built with a new memory efficient building block with a **sandwich layout**, and an efficient **cascaded group attention** operation which mitigates attention computation redundancy. 
<div align="center">
    <img width="69%" alt="EfficientViT overview" src="./EfficientViT/classification/.figures/efficientvit_main_static.png"/>
</div>

### [TinyViT](./TinyViT)
TinyViT is a new family of **tiny and efficient** vision transformers pretrained on **large-scale** datasets with our proposed **fast distillation framework**. The central idea is to **transfer knowledge** from **large pretrained models** to small ones. The logits of large teacher models are sparsified and stored in disk in advance to **save the memory cost and computation overheads**.
<div align="center">
    <img width="80%" alt="TinyViT overview" src="./TinyViT/.figure/framework.png"/>
</div>

### [MiniViT](./MiniViT)
MiniViT is a new compression framework that achieves parameter reduction in vision transformers while retaining the same performance. The central idea of MiniViT is to multiplex the weights of consecutive transformer blocks. Specifically, we make the weights shared across layers, while imposing a transformation on the weights to increase diversity. Weight distillation over self-attention is also applied to transfer knowledge from large-scale ViT models to weight-multiplexed compact models.
<div align="center">
    <img width="70%" alt="MiniViT overview" src="./MiniViT/.figure/framework.png"/>
</div>

### [CDARTS](./CDARTS)
In this work, we propose new joint optimization objectives and a novel Cyclic Differentiable ARchiTecture Search framework, dubbed CDARTS. Considering the structure difference, CDARTS builds a cyclic feedback mechanism between the search and evaluation networks with introspective distillation. 
<div align="center">
    <img width="50%" alt="CDARTS overview" src="CDARTS/demo/framework1.png"/>
</div>


### [AutoFormerV2](./AutoFormerV2)
In this work, instead of searching the architecture in a predefined search space, with the help of AutoFormer, we proposed to search the search space to automatically find a great search space first. 
After that we search the architectures in the searched space. In addition, we provide insightful observations and guidelines for general vision transformer design.
<div align="center">
    <img width="70%" alt="AutoFormerV2 overview" src="AutoFormerV2/.figure/overview.jpg"/>
</div>


### [AutoFormer](./AutoFormer)

AutoFormer is new one-shot architecture search framework dedicated to vision transformer search. It entangles the weights of different vision transformer blocks in the same layers during supernet training. 
Benefiting from the strategy, the trained supernet allows thousands of subnets to be very well-trained. Specifically, the performance of these subnets with weights inherited from the supernet is comparable to those retrained from scratch.
<div align="center">
    <img width="70%" alt="AutoFormer overview" src="AutoFormer/.figure/overview.png"/>
</div>

### [iRPE](./iRPE)
**Image RPE (iRPE for short) methods are new relative position encoding methods dedicated to 2D images**, considering directional relative distance modeling as well as the interactions between queries and relative position embeddings in self-attention mechanism. The proposed iRPE methods are simple and lightweight, being easily plugged into transformer blocks. Experiments demonstrate that solely due to the proposed encoding methods, **DeiT and DETR obtain up to 1.5% (top-1 Acc) and 1.3% (mAP) stable improvements** over their original versions on ImageNet and COCO respectively, without tuning any extra hyperparamters such as learning rate and weight decay. Our ablation and analysis also yield interesting findings, some of which run counter to previous understanding.
<div align="center">
    <img width="70%" alt="iRPE overview" src="iRPE/iRPE.png"/>
</div>


### [Cream](./Cream)
**[[Paper]](https://papers.nips.cc/paper/2020/file/d072677d210ac4c03ba046120f0802ec-Paper.pdf) [[Models-Google Drive]](https://drive.google.com/drive/folders/1NLGAbBF9bA1IUAxKlk2VjgRXhr6RHvRW?usp=sharing)[[Models-Baidu Disk (password: wqw6)]](https://pan.baidu.com/s/1TqQNm2s14oEdyNPimw3T9g) [[Slides]]() [[BibTex]](https://scholar.googleusercontent.com/scholar.bib?q=info:ICWVXc_SsKAJ:scholar.google.com/&output=citation&scisdr=CgUmooXfEMfTi0cV5aU:AAGBfm0AAAAAX7sQ_aXoamdKRaBI12tAVN8REq1VKNwM&scisig=AAGBfm0AAAAAX7sQ_RdYtp6BSro3zgbXVJU2MCgsG730&scisf=4&ct=citation&cd=-1&hl=ja)**  <br/>

In this work, we present a simple yet effective architecture distillation method. The central idea is that subnetworks can learn collaboratively and teach each other throughout the training process, aiming to boost the convergence of individual models. We introduce the concept of prioritized path, which refers to the architecture candidates exhibiting superior performance during training. Distilling knowledge from the prioritized paths is able to boost the training of subnetworks. Since the prioritized paths are changed on the fly depending on their performance and complexity, the final obtained paths are the cream of the crop.
<div >
    <img src="Cream/demo/intro.jpg" width="90%"/>
</div>


## Bibtex
```bibtex
@InProceedings{tinyclip,
    title     = {TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance},
    author    = {Wu, Kan and Peng, Houwen and Zhou, Zhenghong and Xiao, Bin and Liu, Mengchen and Yuan, Lu and Xuan, Hong and Valenzuela, Michael and Chen, Xi (Stephen) and Wang, Xinggang and Chao, Hongyang and Hu, Han},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {21970-21980}
}

@InProceedings{liu2023efficientvit,
    title     = {EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention},
    author    = {Liu, Xinyu and Peng, Houwen and Zheng, Ningxin and Yang, Yuqing and Hu, Han and Yuan, Yixuan},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023},
}

@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={European conference on computer vision (ECCV)},
  year={2022}
}

@InProceedings{MiniViT,
    title     = {MiniViT: Compressing Vision Transformers With Weight Multiplexing},
    author    = {Zhang, Jinnian and Peng, Houwen and Wu, Kan and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {12145-12154}
}

@article{CDARTS,
  title={Cyclic Differentiable Architecture Search},
  author={Yu, Hongyuan and Peng, Houwen and Huang, Yan and Fu, Jianlong and Du, Hao and Wang, Liang and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2022}
}

@article{S3,
  title={Searching the Search Space of Vision Transformer},
  author={Minghao, Chen and Kan, Wu and Bolin, Ni and Houwen, Peng and Bei, Liu and Jianlong, Fu and Hongyang, Chao and Haibin, Ling},
  booktitle={Conference and Workshop on Neural Information Processing Systems (NeurIPS)},
  year={2021}
}

@InProceedings{iRPE,
    title     = {Rethinking and Improving Relative Position Encoding for Vision Transformer},
    author    = {Wu, Kan and Peng, Houwen and Chen, Minghao and Fu, Jianlong and Chao, Hongyang},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {10033-10041}
}

@InProceedings{AutoFormer,
    title     = {AutoFormer: Searching Transformers for Visual Recognition},
    author    = {Chen, Minghao and Peng, Houwen and Fu, Jianlong and Ling, Haibin},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {12270-12280}
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

