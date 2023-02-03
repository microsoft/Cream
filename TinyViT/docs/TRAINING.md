# Training TinyViT

In this document, we introduce how to pretrain TinyViT with the proposed fast pretraining distillation.

Note: If the GPU memory is not enough to fit the batch size, you can use `Gradient accumulation steps` by adding the argument `--accumulation-steps <acc_steps>`. For example, the accumulated batch size per GPU is 128 (= 32 x 4) when passing the arguments `--batch-size 32 --accumulation-steps 4`.

## Pretrain the model on ImageNet-22k with the proposed fast pretraining distillation.

Before training with the proposed fast pretraining distillation, we need to store the teacher sparse soft labels by [the tutorial](./SAVE_TEACHER_LOGITS.md).

Assume that the teacher sparse soft labels are stored in the path `./teacher_logits/`, and the IN-22k dataset is stored in the folder `./ImageNet-22k`.

We use 4 nodes (8 GPUs per node) to pretrain the model on IN-22k with the distillation of stored soft labels.

```bash
python -m torch.distributed.launch --master_addr=$MASTER_ADDR --nproc_per_node 8 --nnodes=4 --node_rank=$NODE_RANK main.py --cfg configs/22k_distill/tiny_vit_21m_22k_distill.yaml --data-path ./ImageNet-22k --batch-size 128 --output ./output --opts DISTILL.TEACHER_LOGITS_PATH ./teacher_logits/
```

where `$NODE_RANK` and `$MASTER_ADDR` are the rank of a node and the IP address of the master node.

## Finetune on ImageNet-1k

- Finetune the pretrained model from IN-22k to IN-1k

After pretrained on IN-22k, the model can be finetuned on IN-1k by the following command.

```
python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/22kto1k/tiny_vit_21m_22kto1k.yaml --data-path ./ImageNet --batch-size 128 --pretrained ./checkpoints/tiny_vit_21m_22k_distill.pth --output ./output
```

where `tiny_vit_21m_22k.pth` is the checkpoint of pretrained TinyViT-21M on IN-22k dataset.

- Finetune with higher resolution

To obtain better accuracy, we finetune the model to higher resolution progressively (224 -> 384 -> 512).

<details>
<summary>Finetune with higher resolution from 224 to 384</summary>
<pre><code> python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/higher_resolution/tiny_vit_21m_224to384.yaml --data-path ./ImageNet --batch-size 32 --pretrained ./checkpoints/tiny_vit_21m_22kto1k_distill.pth --output ./output  --accumulation-steps 4
</code></pre>
</details>

<details>
<summary>Finetune with higher resolution from 384 to 512</summary>
<pre><code> python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/higher_resolution/tiny_vit_21m_384to512.yaml --data-path ./ImageNet --batch-size 32 --pretrained ./checkpoints/tiny_vit_21m_22kto1k_384_distill.pth --output ./output  --accumulation-steps 4
</code></pre>
</details>

## Train the model from scratch on ImageNet-1k

Here is the command to train TinyViT from scratch on ImageNet-1k.

```bash
python -m torch.distributed.launch --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path ./ImageNet --batch-size 128 --output ./output
```
