import os
import argparse
import torch
from PIL import Image
from gradio_demo.utils_drag import image2pil
from models_diffusers.controlnet_svd import ControlNetSVDModel
from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from pipelines.pipeline_stable_video_diffusion_interp_control import StableVideoDiffusionInterpControlPipeline
from diffusers.utils.import_utils import is_xformers_available
import torchvision
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
import torchvision.transforms.functional as TF
import torchvision.io as io

def ensure_dirname(dirname):
    os.makedirs(dirname, exist_ok=True)


def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"🛠️ step 1. setup & input check")
    if not os.path.exists(args.first_frame) or not os.path.exists(args.last_frame):
        raise FileNotFoundError("Start or end frame image not found.")
    # 예시: 이름에 따라 고정된 tracking 경로
    track_file = args.track_file
    with open(track_file, 'r') as f:
        lines = f.readlines()
    start_x,start_y  = lines[0].strip().split(',')
    start_x, start_y = float(start_x.strip()), float(start_y.strip())
    end_x,end_y      = lines[1].strip().split(',')
    end_x, end_y = float(end_x.strip()), float(end_y.strip())
    input_all_points = [[[start_x,start_y],[end_x,end_y]]]

    original_width = args.width
    original_height = args.height
    resized_all_points = [tuple(
        [tuple([int(e1[0] * args.width / original_width), int(e1[1] * args.height / original_height)]) for e1 in e]) for
                          e in input_all_points]

    def interpolate_trajectory(points, n_points):
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        t = np.linspace(0, 1, len(points))
        fx = PchipInterpolator(t, x)
        fy = PchipInterpolator(t, y)
        new_t = np.linspace(0, 1, n_points)
        new_x = fx(new_t)
        new_y = fy(new_t)
        new_points = list(zip(new_x, new_y))
        return new_points

    for idx, splited_track in enumerate(resized_all_points):
        if len(splited_track) == 1:  # stationary point
            displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
            splited_track = tuple([splited_track[0], displacement_point])
        splited_track = interpolate_trajectory(splited_track, args.num_frames)
        splited_track = splited_track[:args.num_frames]
        resized_all_points[idx] = splited_track
    pred_tracks = torch.tensor(resized_all_points)  # (num_points, num_frames, 2)
    with_control = True
    pred_tracks = pred_tracks.permute(1, 0, 2).to(device, dtype)  # (num_frames, num_points, 2)
    point_embedding = None
    sift_track_update = False
    anchor_points_flag = None

    ensure_dirname(args.output_dir)

    # step 2. UNet 로드
    print(f"📦 step 2. load UNet")
    if not args.base :
        unet_path = os.path.join(args.model, "unet")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            unet_path, torch_dtype=dtype, low_cpu_mem_usage=True, custom_resume=True
        ).to(device, dtype)
        for module in unet.modules():
            module.to(device=device, dtype=dtype)
        for param in unet.parameters():
            param.data = param.data.to(device, dtype=dtype)
        for buffer in unet.buffers():
            buffer.data = buffer.data.to(device, dtype=dtype)

        # step 3. ControlNet 로드
        print(f"📦 step 3. load ControlNet")
        controlnet_path = os.path.join(args.model, "controlnet")
        controlnet = ControlNetSVDModel.from_pretrained(controlnet_path).to(device, dtype)
        for module in controlnet.modules():
            module.to(device=device, dtype=dtype)
        for param in controlnet.parameters():
            param.data = param.data.to(device, dtype=dtype)
        for buffer in controlnet.buffers():
            buffer.data = buffer.data.to(device, dtype=dtype)
    else :
        model_dir = "./checkpoints/framer_512x320"
        unet = UNetSpatioTemporalConditionModel.from_pretrained(os.path.join(model_dir, "unet"),
                                                                torch_dtype=dtype,
                                                                low_cpu_mem_usage=True,
                                                                # low_cpu_mem_usage=False,
                                                                custom_resume=True, ).to(device, dtype)
        controlnet = ControlNetSVDModel.from_pretrained(os.path.join(model_dir, "controlnet")).to(device, dtype)

    # step 4. xformers 확인
    print(f"🔍 step 4. check xformers")
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise RuntimeError("xformers is not available. Please install it.")

    # step 5. 파이프라인 구성
    print(f"🧪 step 5. build pipeline")
    pipe = StableVideoDiffusionInterpControlPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        unet=unet,
        controlnet=controlnet,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
        variant="fp16",
        local_files_only=True,).to(device)

    # step 6. 이미지 불러오기
    print(f"🖼️ step 6. load input images")
    first_image = Image.open(args.first_frame).convert("RGB").resize((args.width, args.height))
    last_image = Image.open(args.last_frame).convert("RGB").resize((args.width, args.height))

    # step 7. 추론 실행
    print(f"🚀 step 7. run inference")
    video_frames = pipe(first_image,
                        last_image,
                        with_control=with_control,
                        point_tracks=pred_tracks,
                        point_embedding=point_embedding,
                        with_id_feature=False,
                        controlnet_cond_scale=args.controlnet_cond_scale,
                        num_frames=args.num_frames,
                        width=args.width,
                        height=args.height,
                        motion_bucket_id=args.motion_bucket_id,
                        fps=7,
                        num_inference_steps=30,
                        sift_track_update=sift_track_update,
                        anchor_points_flag=anchor_points_flag).frames[0]

    print(f"💾 step 8. save frames")
    for i, frame in enumerate(video_frames):
        save_path = os.path.join(args.output_dir, f"{args.name}_frame_{i}.png")
        frame.save(save_path)
        print(f"✅ Saved frame {i} to {save_path}")

    # step 9. 비디오로 저장 (GIF)
    print(f"🎞️ step 9. save video")
    gif_path = os.path.join(args.output_dir, f"{args.name}.gif")
    duration = 100
    video_frames[0].save(gif_path, save_all=True, append_images=video_frames[1:], loop=0, duration=duration)


if __name__ == "__main__":
    # step 0. argparse 설정
    parser = argparse.ArgumentParser(description="Framer: Interactive Frame Interpolation")

    parser.add_argument("--model", type=str, default="./checkpoints/framer_512x320")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--controlnet_cond_scale", type=float, default=1.0)
    parser.add_argument("--motion_bucket_id", type=int, default=100)
    parser.add_argument("--first_frame", type=str, default="./assets/start.png")
    parser.add_argument("--last_frame", type=str, default="./assets/end.png")
    parser.add_argument("--track_file", type=str, default="./assets/start.txt")
    parser.add_argument("--with_control", type=int, default=1) #
    parser.add_argument("--name", type=str)
    parser.add_argument("--base", action="store_true")
    args = parser.parse_args()
    main(args)
