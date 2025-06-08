#!/bin/bash

name="base"
result_folder="/workspace/data/diffusion/Framer/Result/${name}"
device_num=2
test_folders=("001" "002" "003" "004" "005" "006" "007" "008" "009" "010"
              "011" "012" "013" "014" "015" "016" "017" "018" "019" "020" "021")

echo "================================================================================================"
echo "Running for $ckpt"
echo "================================================================================================"

for test_folder in "${test_folders[@]}"; do
  output_dir="${result_folder}/result/${ckpt}/${test_folder}_without_track"
  echo "----------- Testing ${test_folder} -----------"

  CUDA_VISIBLE_DEVICES=${device_num} python test.py \
    --output_dir "$output_dir" \
    --model "${result_folder}/${ckpt}" \
    --num_frames 14 --width 512 --height 320 \
    --controlnet_cond_scale 1.0 --motion_bucket_id 100 --dtype float32 \
    --with_no_track --without_controlnet --base
done