export NNODES=1
export GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES"
torchrun $DISTRIBUTED_ARGS src/training/main.py \
 --save-frequency 1 \
 --report-to wandb \
 --train-data synthetic \
 --dataset-type synthetic \
 --imagenet-val ./ImageNet \
 --warmup 2000 \
 --batch-size 1024 \
 --epochs 6 \
 --workers 8 \
 --model TinyCLIP-ViT-61M-32-Text-29M \
 --name exp_name \
 --seed 0 \
 --local-loss \
 --grad-checkpointing \
 --logs ./outputs/ViT-B-32 \
 --lr 0.0001 \
 --gather-with-grad \
 --pretrained-image-file ViT-B-32@laion2b_e16 \
 --pretrained-text-file ViT-B-32@laion2b_e16 \
 --distillation-teacher ViT-B-32@laion2b_e16 \
 --logit-scale 50 \
 --norm_gradient_clip 5 \
 --train-num-samples 400000000
