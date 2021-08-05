# AutoFormer: Searching Transformers for Visual Recognition

## code is coming soon

This is an official implementation of AutoFormer.

<div align="center">
    <img width="49%" alt="AutoFormer overview" src="https://github.com/microsoft/AutoML/releases/download/static_files/autoformer_overview.gif"/>
    <img width="49%" alt="AutoFormer detail" src="https://github.com/microsoft/AutoML/releases/download/static_files/autoformer_details.gif"/>
</div>


## Highlights
- Once-for-all

AutoFormer is a simple yet effective method to train a once-for-all vision transformer supernet.

- Competive performance

AutoFormers consistently outperform DeiTs.

## Performance

**Left:** Top-1 accuracy on ImageNet. Our method achieves very competitive performance, being superior to the recent DeiT and ViT. **Right:** 1000 random sampled good architectures in the supernet-S. The supernet trained under our strategy allows subnets to be well optimized.

<div align="half">
    <img src=".figure/performance.png" width="49%"/>
    <img src=".figure/ofa.png" width="49%"/>
</div>

