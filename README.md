

### [AutoFormer](./AutoFormer)

AutoFormer is new one-shot architecture search framework dedicated to vision transformer search. It entangles the weights of different vision transformer blocks in the same layers during supernet training. 
Benefiting from the strategy, the trained supernet allows thousands of subnets to be very well-trained. Specifically, the performance of these subnets with weights inherited from the supernet is comparable to those retrained from scratch.
<div align="center">
    <img width="70%" alt="AutoFormer overview" src="AutoFormer/.figure/overview.png"/>
</div>



## Bibtex
```bibtex

@article{AutoFormer,
  title={AutoFormer: Searching Transformers for Visual Recognition},
  author={Chen, Minghao and Peng, Houwen and Fu, Jianlong and Ling, Haibin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
