#!/bin/bash

# (1) Conda 또는 환경 설정 (필요 시)
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# (2) 결과 저장 폴더
result_folder="/workspace/data/diffusion/Framer/Result/nerf_data_slerp"

# (3) 반복할 체크포인트 리스트
checkpoints=("checkpoint-550")
device_num=1
test_folders=("013_snow" "014_snow" "015_smog")

# (4) 반복 실행
for ckpt in "${checkpoints[@]}"; do

  echo "================================================================================================"
  echo "Running for $ckpt"
  echo "================================================================================================"

  for test_folder in "${test_folders[@]}"; do
    echo "----------- Testing ${test_folder} -----------"

    CUDA_VISIBLE_DEVICES=$device_num python test.py \
      --model "${result_folder}/${ckpt}" \
      --output_dir "${result_folder}/result/${ckpt}/${test_folder}" \
      --first_frame "./assets/${test_folder}/input_frames/flood_0.png" \
      --last_frame  "./assets/${test_folder}/input_frames/flood_1.png" \
      --track_file  "./assets/${test_folder}/input_frames/track.txt" \
      --num_frames 14 \
      --width 512 \
      --height 320 \
      --controlnet_cond_scale 1.0 \
      --motion_bucket_id 100 \
      --dtype float32 \
      --name ${test_folder}
  done
done
