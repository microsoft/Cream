# The tutorial of saving teacher sparse logits

This document shows how to save and check teacher sparse soft labels.

We provide an example to store the sparse soft labels of **CLIP-ViT-Large/14-22k** on ImageNet-22k. With the pretrained teacher, **TinyViT-5/11/21M** will achieve the Top-1 accuracy of **80.7/83.2/84.8 %** on ImageNet-1k valiadation set.

## Save teacher sparse logits
Firstly, we prepare the IN-22k dataset ([Data Preparation](./PREPARATION.md)), then download the checkpoint of CLIP-ViT-Large/14-22k in [the link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/pretrained_teacher/clip_vit_large_patch14_22k.pth).

The following command will store the teacher sparse logits.

```bash
python -m torch.distributed.launch --nproc_per_node 8 save_logits.py --cfg configs/teacher/clip_vit_large_patch14_22k.yaml --data-path ./ImageNet-22k --batch-size 128 --eval --resume checkpoints/clip_vit_large_patch14_22k.pth --opts DISTILL.TEACHER_LOGITS_PATH ./teacher_logits/
```

**The accuracy of CLIP-ViT-Large/14-22k (w/o finetune on IN-1k) on IN-1k is Acc@1 85.894 Acc@5 97.566.**

Since IN-22k is too large, we recommend to use few data to debug by adding the argument `DATA.DEBUG True`.

- How to save sparse logits **in parallel** ?

Since the teacher logits per epoch is independent, they can be saved in parallel. Specifically, each machine saves a segment of the whole epochs individually.
We can add the epoch interval into the command, e.g.
```bash
python -m torch.distributed.launch --nproc_per_node 8 save_logits.py --cfg configs/teacher/clip_vit_large_patch14_22k.yaml --data-path ./ImageNet-22k --batch-size 128 --eval --resume checkpoints/clip_vit_large_patch14_22k.pth --opts DISTILL.TEACHER_LOGITS_PATH ./teacher_logits/ TRAIN.START_EPOCH 30 TRAIN.EPOCHS 40
```
The sparse logits between 30 to 40 will be saved.

## Check teacher sparse logits
After saving the logits, we can check them by adding the extra argument `--check-saved-logits`.
```bash
python -m torch.distributed.launch --nproc_per_node 8 save_logits.py --cfg configs/teacher/clip_vit_large_patch14_22k.yaml --data-path ./ImageNet-22k --batch-size 128 --eval --resume checkpoints/clip_vit_large_patch14_22k.pth --check-saved-logits --opts DISTILL.TEACHER_LOGITS_PATH ./teacher_logits
```
