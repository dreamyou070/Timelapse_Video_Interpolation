# 학습 job 이름 설정
name="nerf_data_train_motion_prompt"

# 백그라운드에서 학습 실행
# nohup
accelerate launch --config_file ../accelerate_config/acc_0.yaml \
    --main_process_port 20500 \
    train.py \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
    --seed 42 \
    --num_workers 4 \
    --per_gpu_batch_size 2 \
    --output_dir "/workspace/data/diffusion/Framer/Result/${name}" \
    --report_to wandb \
    --wandb_project_name "flood_video_project" \
    --wandb_name "${name}" \
    --checkpointing_steps 50 \
    --validation_steps 1000 \
    --checkpoints_total_limit 10 \
    --sample_n_frames 5 \
    --max_train_steps 50000 \
    --projector_input_dim 2048

#> "nohup_${name}.out" 2>&1 &