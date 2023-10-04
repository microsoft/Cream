export NNODES=1
export GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES"
torchrun $DISTRIBUTED_ARGS src/training/main.py \
 --save-frequency 1 \
 --report-to wandb \
 --train-data synthetic \
 --dataset-type synthetic \
 --imagenet-val ./ImageNet \
 --warmup 3000 \
 --batch-size 1024 \
 --epochs 6 \
 --workers 8 \
 --model ViT-B-32 \
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
 --norm_gradient_clip 5 \
 --train-num-samples 400000000 \
 --prune-step 3000 \
 --prune-image \
 --prune-text \
 --total-loss-flag \
 --target-sparsity 0.25 \
 --start-sparsity 0.0 \
 --sparsity-warmup 1000 \
 --logit-scale 50
