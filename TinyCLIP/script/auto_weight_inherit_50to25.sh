export NNODES=1
export GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES"
torchrun $DISTRIBUTED_ARGS src/training/main.py \
 --save-frequency 1 \
 --report-to wandb \
 --train-data synthetic \
 --dataset-type synthetic \
 --imagenet-val ./ImageNet \
 --warmup 8000 \
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
 --logit-scale 50 \
 --norm_gradient_clip 5 \
 --train-num-samples 400000000 \
 --prune-step 8000 \
 --prune-image \
 --prune-text \
 --total-loss-flag \
 --target-sparsity 0.75 \
 --start-sparsity 0.5 \
 --sparsity-warmup 1000 \
 --resume ./checkpoints/TinyCLIP-auto-ViT-45M-32-Text-18M-LAION400M.pt \
 --load-last-stage
