# Evaluation

Before evaluation, we need to prepare [the ImageNet-1k dataset](./PREPARATION.md) and [the checkpoints in model zoo](../README.md).

Run the following command for evaluation:

**Evaluate TinyViT with pretraining distillation**

<details>
<summary>Evaluate TinyViT-5M <img src="../.figure/distill.png"></summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/22kto1k/tiny_vit_5m_22kto1k.yaml --data-path ./ImageNet --batch-size 128 --eval --resume ./checkpoints/tiny_vit_5m_22kto1k_distill.pth
</code></pre>
</details>

<details>
<summary>Evaluate TinyViT-11M <img src="../.figure/distill.png"></summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/22kto1k/tiny_vit_11m_22kto1k.yaml --data-path ./ImageNet --batch-size 128 --eval --resume ./checkpoints/tiny_vit_11m_22kto1k_distill.pth
</code></pre>
</details>

<details>
<summary>Evaluate TinyViT-21M <img src="../.figure/distill.png"></summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/22kto1k/tiny_vit_21m_22kto1k.yaml --data-path ./ImageNet --batch-size 128 --eval --resume ./checkpoints/tiny_vit_21m_22kto1k_distill.pth
</code></pre>
</details>

<details>
<summary>Evaluate TinyViT-21M-384 <img src="../.figure/distill.png"></summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/higher_resolution/tiny_vit_21m_224to384.yaml --data-path ./ImageNet --batch-size 64 --eval --resume ./checkpoints/tiny_vit_21m_22kto1k_384_distill.pth
</code></pre>
</details>

<details>
<summary>Evaluate TinyViT-21M-512 <img src="../.figure/distill.png"></summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/higher_resolution/tiny_vit_21m_384to512.yaml --data-path ./ImageNet --batch-size 32 --eval --resume ./checkpoints/tiny_vit_21m_22kto1k_512_distill.pth
</code></pre>
</details>

**Evaluate TinyViT trained from scratch in IN-1k**

<details>
<summary>Evaluate TinyViT-5M</summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_5m.yaml --data-path ./ImageNet --batch-size 128 --eval --resume ./checkpoints/tiny_vit_5m_1k.pth
</code></pre>
</details>

<details>
<summary>Evaluate TinyViT-11M</summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_11m.yaml --data-path ./ImageNet --batch-size 128 --eval --resume ./checkpoints/tiny_vit_11m_1k.pth
</code></pre>
</details>

<details>
<summary>Evaluate TinyViT-21M</summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path ./ImageNet --batch-size 128 --eval --resume ./checkpoints/tiny_vit_21m_1k.pth
</code></pre>
</details>

**The model pretrained on IN-22k can be evaluated directly on IN-1k**

Since the model pretrained on IN-22k is not finetuned on IN-1k, the accuracy is lower than the model finetuned 22kto1k.

<details>
<summary>Evaluate TinyViT-5M-22k <img src="../.figure/distill.png"></summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/22k_distill/tiny_vit_5m_22k_distill.yaml --data-path ./ImageNet --batch-size 128 --eval --resume ./checkpoints/tiny_vit_5m_22k_distill.pth --opts DATA.DATASET imagenet
</code></pre>
</details>

<details>
<summary>Evaluate TinyViT-11M-22k <img src="../.figure/distill.png"></summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/22k_distill/tiny_vit_11m_22k_distill.yaml --data-path ./ImageNet --batch-size 128 --eval --resume ./checkpoints/tiny_vit_11m_22k_distill.pth --opts DATA.DATASET imagenet
</code></pre>
</details>

<details>
<summary>Evaluate TinyViT-21M-22k <img src="../.figure/distill.png"></summary>
<pre><code>python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/22k_distill/tiny_vit_21m_22k_distill.yaml --data-path ./ImageNet --batch-size 128 --eval --resume ./checkpoints/tiny_vit_21m_22k_distill.pth --opts DATA.DATASET imagenet
</code></pre>
</details>
