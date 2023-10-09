# TinyCLIP-ViT Inference

## Download checkpoints

Download a checkpoint from [Model Zoo](../README.md#model-zoo).

## Zero-shot inference on ImageNet-1k

Please change the paths to `imagenet-val` and `resume`.

### For manual weight inference checkpoint:

<details>
<summary>Evaluate TinyCLIP ViT-39M/16 + Text-19M (YFCC-15M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model TinyCLIP-ViT-39M-16-Text-19M \
--eval \
--resume ./checkpoints/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ViT-8M/16 + Text-3M (YFCC-15M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model TinyCLIP-ViT-8M-16-Text-3M \
--eval \
--resume ./checkpoints/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ResNet-30M + Text-29M (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model TinyCLIP-ResNet-30M-Text-29M \
--eval \
--resume ./checkpoints/TinyCLIP-ResNet-30M-Text-29M-LAION400M.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ResNet-19M + Text-19M (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model TinyCLIP-ResNet-19M-Text-19M \
--eval \
--resume ./checkpoints/TinyCLIP-ResNet-19M-Text-19M-LAION400M.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ViT-61M/32 + Text-29M (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model TinyCLIP-ViT-61M-32-Text-29M \
--eval \
--resume ./checkpoints/TinyCLIP-ViT-61M-32-Text-29M-LAION400M.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ViT-40M/32 + Text-19M (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model TinyCLIP-ViT-40M-32-Text-19M \
--eval \
--resume ./checkpoints/TinyCLIP-ViT-40M-32-Text-19M-LAION400M.pt
</code></pre>
</details>

### For auto weight inference checkpoint:

<details>
<summary>Evaluate TinyCLIP ViT-63M/32 + Text-31M (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model ViT-B-32 \
--prune-image \
--prune-text \
--eval \
--resume ./checkpoints/TinyCLIP-auto-ViT-63M-32-Text-31M-LAION400M.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ViT-45M/32 + Text-18M (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model ViT-B-32 \
--prune-image \
--prune-text \
--eval \
--resume ./checkpoints/TinyCLIP-auto-ViT-45M-32-Text-18M-LAION400M.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ViT-22M/32 + Text-10M (LAION-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model ViT-B-32 \
--prune-image \
--prune-text \
--eval \
--resume ./checkpoints/TinyCLIP-auto-ViT-22M-32-Text-10M-LAION400M.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ViT-63M/32 + Text-31M (LAION+YFCC-400M) </summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model ViT-B-32 \
--prune-image \
--prune-text \
--eval \
--resume ./checkpoints/TinyCLIP-auto-ViT-63M-32-Text-31M-LAIONYFCC400M.pt
</code></pre>
</details>

<details>
<summary>Evaluate TinyCLIP ViT-45M/32 + Text-18M (LAION+YFCC-400M)
</summary>
<pre><code>python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py \
--imagenet-val ./ImageNet \
--model ViT-B-32 \
--prune-image \
--prune-text \
--eval \
--resume ./checkpoints/TinyCLIP-auto-ViT-45M-32-Text-18M-LAIONYFCC400M.pt
</code></pre>
</details>
