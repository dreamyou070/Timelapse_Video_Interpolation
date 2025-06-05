#bash train_pseudo.sh 를 nohup 으로 돌려줘
accelerate launch --config_file ../accelerate_config/acc_3.yaml \
    --main_process_port 22332 \
    train_pseudo.py \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt --seed 42 \
    --num_workers 4  \
    --per_gpu_batch_size 2 \
    --output_dir /workspace/data/diffusion/Framer/Result/nerf_data_firstframe_conditioned \
    --report_to wandb \
    --wandb_project_name 'flood_video_project' \
    --wandb_name 'nerf_data_firstframe_conditioned' \
    --checkpointing_steps 50 \
    --validation_steps 1000 \
    --checkpoints_total_limit 10 \
    --sample_n_frames 5 --endframe_conditioned
# if accelerator.mixed_precision == "fp16":