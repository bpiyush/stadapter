#!/usr/bin/env sh
output_dir=/work/piyush/experiments/stadapter/
n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
model_id=clip4clip_vit_base_patch16_meanpool
batch_size=32

torchrun --nproc_per_node $n_gpus main_wandb.py \
    --model $model_id \
    --save_dir $output_dir/ssv2/$model_id \
    --auto_resume --auto_remove \
    --dataset ssv2 \
    --num_frames 8 \
    --sampling_rate 0 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --batch_size $batch_size \
    --epochs 100 \
    --warmup_epochs 2 \
    --eval_freq 5 \
    --num_workers 4
