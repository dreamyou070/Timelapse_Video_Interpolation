import os
import torch
import numpy as np
from PIL import Image
import uuid
import torchvision
from models_diffusers.controlnet_svd import ControlNetSVDModel
from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from pipelines.pipeline_stable_video_diffusion_interp_control import StableVideoDiffusionInterpControlPipeline
import argparse

def main(args) :

    # ⚡️ (1) 경로 설정 및 이미지 로드 ------------------------------------------------
    folders = ['012_Experiment']
    for folder in folders:

        args.folder = folder
        base_image_dir = f'assets/{args.folder}/input_frames'
        input_image_path = os.path.join(base_image_dir, 'flood_0.png')
        input_image_end_path = os.path.join(base_image_dir, 'flood_1.png')
        output_dir = f'assets/{args.folder}/output_videos'
        os.makedirs(output_dir, exist_ok=True)

        num_frames = 14
        if args.use_custom_point :
            for i in range(num_frames):
                heat_map_dir = os.path.join(base_image_dir, f'{i}.png')
                heat_map_pil = Image.open(heat_map_dir).convert('L')


        # ⚡️ (2) 설정
        model_dir = "./checkpoints/framer_512x320"
        # model_lengths = [3,4,5,6,7,8,9,10]
        model_lengths = [49]
        for model_length in model_lengths :
            width, height = 512, 320
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float32

            # ⚡️ (3) 모델 불러오기
            print(f' (3.1) call unet')
            unet = UNetSpatioTemporalConditionModel.from_pretrained(os.path.join(model_dir, "unet"),
                                                                    torch_dtype=dtype,
                                                                    low_cpu_mem_usage=True,
                                                                    #low_cpu_mem_usage=False,
                                                                    custom_resume=True,).to(device, dtype)
            print(f' (3.2) call controlnet')
            controlnet = ControlNetSVDModel.from_pretrained(os.path.join(model_dir, "controlnet")).to(device, dtype)
            print(f' (3.3) call pipeline')
            pipe = StableVideoDiffusionInterpControlPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",
                                                                             unet=unet,
                                                                             controlnet=controlnet,
                                                                             low_cpu_mem_usage=False,
                                                                             torch_dtype=dtype,
                                                                             variant="fp16",
                                                                             local_files_only=False,).to(device)
            vae_scale_factor = pipe.vae_scale_factor # 8
            # ⚡️ (4) 입력 이미지 전처리
            input_image = Image.open(input_image_path).convert('RGB').resize((width, height))
            input_image_end = Image.open(input_image_end_path).convert('RGB').resize((width, height))

            # ⚡️ (5) Trajectory (dummy)
            pred_tracks = torch.zeros((1, model_length, 2))  # (1, num_frames, 2) -> (num_frames, 1,2)
            with_control = True

            # ⚡️ (6) Video 생성
            video_frames = pipe(input_image,     # ------------------------------------------------------------------------------------------- #
                input_image_end, # ------------------------------------------------------------------------------------------- #
                with_control=with_control,
                point_tracks=pred_tracks.permute(1, 0, 2).to(device, dtype),  # (num_frames, num_points, 2)
                point_embedding=None,
                with_id_feature=False,
                controlnet_cond_scale=1.0,
                num_frames=model_length,
                width=width,
                height=height,
                motion_bucket_id=100,
                fps=7,
                num_inference_steps=3,
                do_interpolation = args.do_interpolation
            ).frames[0]
            # ⚡️ (7) GIF/MP4로 저장
            output_gif = os.path.join(output_dir,f"vis_{model_length}.gif")
            if args.do_interpolation :
                output_gif = os.path.join(output_dir, f"vis_{str(uuid.uuid4())[:4]}_with_interpolation.gif")

            duration = 100
            fps = (1000/ duration)
            # fps ?
            #
            video_frames[0].save(output_gif, save_all=True, append_images=video_frames[1:], duration=duration, loop=0)
            print(f"GIF saved to {output_gif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--model", type=str, default="checkpoints/framer_512x320", help="Path to the pretrained model.")
    parser.add_argument("--num_frames", type=int, default=14, help="Number of video frames.")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320) # do_interpolation
    parser.add_argument("--do_interpolation", action="store_true")
    args = parser.parse_args()
    main(args)
