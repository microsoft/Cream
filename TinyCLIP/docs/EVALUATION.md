# TinyCLIP-ViT Inference

## Download checkpoints

Download a checkpoint from [Model Zoo](../README.md#model-zoo).

## Zero-shot inference on ImageNet-1k

Please change the paths to `imagenet-val` and `resume`.

### For manual weight inference checkpoint:

<details>
<summary>Evaluate TinyCLIP ResNet-30M (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model RN50_P75 \
--eval \
--resume ./checkpoints/TinyCLIP-ResNet-30M_epoch_6.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ResNet-19M (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model RN50_P50 \
--eval \
--resume ./checkpoints/TinyCLIP-ResNet-19M_epoch_12.pt
</code></pre>
</details>


<details>
<summary>Evaluate TinyCLIP ViT-61M/32 (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model TinyCLIP-61M \
--eval \
--resume ./checkpoints/TinyCLIP-ViT-61M-32_epoch_6.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ViT-40M/32 (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model TinyCLIP-40M \
--eval \
--resume ./checkpoints/TinyCLIP-ViT-40M-32_epoch_16.pt
</code></pre>
</details>

### For auto weight inference checkpoint:

<details>
<summary>Evaluate TinyCLIP ViT-63M/32 (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model ViT-B-32 \
--prune-image \
--prune-text \
--eval \
--resume ./checkpoints/TinyCLIP-ViT-63M-32-LAION.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ViT-45M/32 (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model ViT-B-32 \
--prune-image \
--prune-text \
--eval \
--resume ./checkpoints/TinyCLIP-ViT-45M-32-LAION.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ViT-22M/32 (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model ViT-B-32 \
--prune-image \
--prune-text \
--eval \
--resume ./checkpoints/TinyCLIP-ViT-22M-32-LAION.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ViT-63M/32 (LAION+YFCC-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model ViT-B-32 \
--prune-image \
--prune-text \
--eval \
--resume ./checkpoints/TinyCLIP-ViT-63M-32-LAION-YFCC.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ViT-45M/32 (LAION+YFCC-400M)
</summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model ViT-B-32 \
--prune-image \
--prune-text \
--eval \
--resume ./checkpoints/TinyCLIP-ViT-45M-32-LAION-YFCC.pt
</code></pre>
</details>
