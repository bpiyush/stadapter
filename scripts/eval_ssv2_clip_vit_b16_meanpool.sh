output_dir=/work/piyush/experiments/stadapter/
n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
n_gpus=1
model_id=clip4clip_vit_base_patch16_meanpool
batch_size=32

torchrun --nproc_per_node $n_gpus main_wandb.py \
    --model $model_id \
    --save_dir $output_dir/ssv2/$model_id \
    --eval_only \
    --pretrain /work/piyush/experiments/stadapter/ssv2/clip4clip_vit_base_patch16_meanpool/checkpoint-80.pth \
    --dataset ssv2 \
    --num_frames 8 \
    --sampling_rate 0 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
    --batch_size $batch_size \
    --num_workers 4