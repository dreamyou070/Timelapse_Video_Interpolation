#bash train_pseudo.sh 를 nohup 으로 돌려줘
accelerate launch --config_file ../accelerate_config/acc_2_3.yaml \
    --main_process_port 29500 \
    train_svd.py \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
    --pretrain_unet "./checkpoints/framer_512x320/unet" \
    --controlnet_model_name_or_path "./checkpoints/framer_512x320/controlnet" \
    --seed 42 \
    --num_workers 4  \
    --per_gpu_batch_size 2 \
    --output_dir /workspace/data/diffusion/Framer/Result/Test_Experience_20250603_nerf_only_temporal_training \
    --report_to wandb \
    --wandb_project_name 'flood_video_project' \
    --wandb_name 'Test_Experience_firstframe_conditioned_20250604_nerf_only_temporal_training' \
    --checkpointing_steps 50 \
    --validation_steps 1 \

