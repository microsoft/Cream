Hiring research interns for neural architecture search projects: houwen.peng@microsoft.com
# Rethinking and Improving Relative Position Encoding for Vision Transformer

Image RPE (iRPE for short) methods are new relative position encoding methods dedicated to 2D images, considering directional relative distance modeling as well as the interactions between queries and relative position embeddings in self-attention mechanism. The proposed iRPE methods are simple and lightweight, being easily plugged into transformer blocks. Experiments demonstrate that solely due to the proposed encoding methods, DeiT and DETR obtain up to 1.5% (top-1 Acc) and 1.3% (mAP) stable improvements over their original versions on ImageNet and COCO respectively, without tuning any extra hyperparamters such as earning rate and weight decay. Our ablation and analysis also yield interesting findings, some of with run counter to previous understanding.

We provide the implementation of image RPE (iRPE) for image classficiation and object detection.

## Image Classification

[[Code]](./DeiT-with-iRPE)

We equip DeiT models with contextual product shared-head RPE with 50 buckets, and report their accuracy on ImageNet-1K Validation set.

Model | RPE-Q | RPE-K | RPE-V | #Params(M) | MACs(M) | Top-1 Acc.(%) | Top-5 Acc.(%) | Link | Log
----- | ----- | ----- | ----- | ---------- | ------- | ------------- | ------------- | ---- | ---
tiny | | ✔ | | 5.76 | 1284 | 73.7 | 92.0 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_tiny_patch16_224_ctx_product_50_shared_k.pth) | [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_deit_tiny_patch16_224_ctx_product_50_shared_k.txt), [detail](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_deit_tiny_patch16_224_ctx_product_50_shared_k.log)
small | | ✔ | | 22.09 | 4659 | 80.9 | 95.4 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_small_patch16_224_ctx_product_50_shared_k.pth) | [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_deit_small_patch16_224_ctx_product_50_shared_k.txt), [detail](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_deit_small_patch16_224_ctx_product_50_shared_k.log)
small | ✔ | ✔ | | 22.13 | 4706 | 81.0 | 95.5 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_small_patch16_224_ctx_product_50_shared_qk.pth) | [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_deit_small_patch16_224_ctx_product_50_shared_qk.txt), [detail](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_deit_small_patch16_224_ctx_product_50_shared_qk.log)
small | ✔ | ✔ | ✔ | 22.17 | 4885 | 81.2 | 95.5 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_small_patch16_224_ctx_product_50_shared_qkv.pth) | [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_deit_small_patch16_224_ctx_product_50_shared_qkv.txt), [detail](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_deit_small_patch16_224_ctx_product_50_shared_qkv.log)
base | | ✔ | | 86.61 | 17684 | 82.3 | 95.9 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/deit_base_patch16_224_ctx_product_50_shared_k.pth) | [log](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/log_deit_base_patch16_224_ctx_product_50_shared_k.txt), [detail](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/detail_deit_base_patch16_224_ctx_product_50_shared_k.log)

## Object Detection 

[[Code]](./DETR-with-iRPE)

We equip DETR models with contextual product shared-head RPE, and report their mAP on MS COCO Validation set.

Absolute Position Encoding: Sinusoid
Relative Position Encoding: iRPE (contextual product shared-head RPE)

Backbone  | #Buckets | epoch | AP    | AP\_50 | AP\_75 | AP\_S | AP\_M | AP\_L | Link | Log
--------- | -------- | ----- | ----- | ------ | ------ | ----- | ----- | ----- | ---- | ---
ResNet-50 |  7 x 7   | 150   | 0.409 | 0.614  | 0.429  | 0.195 | 0.443 | 0.605 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/rpe-1.9-product-ctx-1-k.pth)     |
ResNet-50 |  9 x 9   | 150   | 0.410 | 0.615  | 0.434  | 0.192 | 0.445 | 0.608 | [link](https://github.com/wkcn/iRPE-model-zoo/releases/download/1.0/rpe-2.0-product-ctx-1-k.pth)     |

We will release the DETR-ResNet-50 with iRPE for 300 epochs and logs as soon as possible.


## How to equip iRPE ?
The detail is shown in [Tutorial](./HOW_TO_EQUIP_iRPE.md).

# Citing iRPE
If this project is helpful for you, please cite it. Thank you! : )

```bibtex
TO BE FILLED
```
