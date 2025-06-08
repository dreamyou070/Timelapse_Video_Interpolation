#bash train_with_newembedding2.sh 를 nohup 으로 돌려줘
name="nerf_data_train_motion_crossattention_decoupled"
accelerate launch --config_file ../accelerate_config/acc_1.yaml \
    --main_process_port 20501 \
    train_decouples.py \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt --seed 42 \
    --num_workers 4  \
    --per_gpu_batch_size 2 \
    --output_dir "/workspace/data/diffusion/Framer/Result/"$name \
    --report_to wandb \
    --wandb_project_name 'flood_video_project' \
    --wandb_name=$name \
    --checkpointing_steps 50 \
    --validation_steps 1000 \
    --checkpoints_total_limit 10 \
    --sample_n_frames 5 --no_point_tracks \
    --max_train_steps 50000 --without_controlnet \
    --projector_input_dim 1024

# if accelerator.mixed_precision == "fp16":