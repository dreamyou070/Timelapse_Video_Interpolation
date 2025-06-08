# 1024 dimension 중에서 필요한 dimension 을 고르는 것도 가능할거 같은데
# 불필요한 dimension 은 무엇일까??
# bash train_with_newembedding.sh
import json
import argparse
import logging
import math
import os
import cv2
import shutil
from diffusers.utils import USE_PEFT_BACKEND
from pathlib import Path
import accelerate
import numpy as np
import torch.nn as nn
from typing import Optional
from PIL import Image, ImageDraw
import torch
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange
import datetime
import diffusers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from torch.utils.data import Dataset
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from models.controlnet_svd import ControlNetSVDModel
import torch.multiprocessing as mp
from pipelines.pipeline_stable_video_diffusion_efficient_interp import StableVideoDiffusionInterpControlPipeline

mp.set_start_method("spawn", force=True)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")
logger = get_logger(__name__, log_level="INFO")


# i should make a utility function file
def validate_and_convert_image(image, target_size=(256, 256)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None

    return image


def create_image_grid(images, rows, cols, target_size=(256, 256)):
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]

    if not valid_images:
        print("No valid images to create a grid")
        return None

    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid


def save_combined_frames(batch_output, validation_images, validation_control_images, output_folder):
    # Flatten batch_output, which is a list of lists of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # Combine frames into a list without converting (since they are already PIL Images)
    combined_frames = validation_images + validation_control_images + flattened_batch_output

    # Calculate rows and columns for the grid
    num_images = len(combined_frames)
    cols = 3  # adjust number of columns as needed
    rows = (num_images + cols - 1) // cols
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    filename = f"combined_frames_{timestamp}.png"
    # Create and save the grid image
    grid = create_image_grid(combined_frames, rows, cols)
    output_folder = os.path.join(output_folder, "validation_images")
    os.makedirs(output_folder, exist_ok=True)

    # Now define the full path for the file
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"combined_frames_{timestamp}.png"
    output_loc = os.path.join(output_folder, filename)

    if grid is not None:
        grid.save(output_loc)
    else:
        print("Failed to create image grid")


def load_images_from_folder(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        parts = filename.split('_')
        if len(parts) > 1 and parts[0] == 'frame':
            try:
                return int(parts[1].split('.')[0])  # Extracting the number part
            except ValueError:
                return float('inf')  # In case of non-integer part, place this file at the end
        return float('inf')  # Non-frame files are placed at the end

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)

    return images


# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3,
                             device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def make_folder_with_permission(path, exist_ok=True):
    os.makedirs(path, exist_ok=exist_ok)
    os.chmod(path, 0o777)  # 폴더 권한을 777로 변경


def main(args):
    print(f"🚀 Step 1: Training setup")
    argument_dict = vars(args)
    # 저장할 경로
    make_folder_with_permission(args.output_dir)
    argument_save_dir = os.path.join(args.output_dir, 'args.json')
    # JSON으로 저장
    with open(argument_save_dir, 'w') as f:
        json.dump(argument_dict, f, indent=4)  # indent=4로 보기 좋게 저장

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. "
                "Please make sure to use `--variant=non_ema` instead."
            ),
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    generator = torch.Generator(device=accelerator.device).manual_seed(23123134)

    if args.report_to == "wandb":
        print(f"🚀 Step 2: Initializing wandb")
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

        if accelerator.is_main_process:
            wandb.init(project=args.wandb_project_name,
                       name=args.wandb_name)

    print(f"🚀 Step 3: Logging setup")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        print(f"🚀 Step 4: Setting seed")
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            print(f"🚀 Step 5: Pushing to HuggingFace Hub")
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    print(f"🚀 Step 6: Loading models and scheduler")
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path,
                                                           subfolder="feature_extractor", revision=args.revision)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path,
                                                                  subfolder="image_encoder", revision=args.revision,
                                                                  variant="fp16")

    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae",
                                                       revision=args.revision, variant="fp16")

    weight_dtype = torch.float32  # final weight_dtype
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print(f"🚀 Step 7: Loading UNet and ControlNet")
    model_dir = "./checkpoints/framer_512x320"
    global_step = 0
    if args.resume_from_checkpoint is not None:
        model_dir = args.resume_from_checkpoint
        check_point_name = os.path.split(model_dir)[-1]
        global_step = int(check_point_name.split('-')[-1])

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        os.path.join(model_dir, "unet"),
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
        custom_resume=True)

    class SimpleEncoder(torch.nn.Module):
        def __init__(self, output_dim=1024):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1))
            )
            self.proj = torch.nn.Linear(64, output_dim)

        def forward(self, x):
            x = self.encoder(x)
            x = x.view(x.size(0), -1)
            return self.proj(x)

    motion_encoder = SimpleEncoder(output_dim=1024)

    if not args.without_controlnet:
        controlnet = ControlNetSVDModel.from_pretrained(os.path.join(model_dir, "controlnet"))

    print(f"🚀 Step 7: set controller")

    def register_attention_control(model, controller):

        def ca_temporal_forward(self, controller, module_name) -> torch.FloatTensor:

            def forward(hidden_states: torch.FloatTensor,
                        encoder_hidden_states: Optional[torch.FloatTensor] = None,
                        attention_mask: Optional[torch.FloatTensor] = None,
                        temb: Optional[torch.FloatTensor] = None,
                        scale: float = 1.0, ):
                do_crossattn = True
                if encoder_hidden_states is None:
                    do_crossattn = False
                residual = hidden_states
                input_ndim = hidden_states.ndim  # [batch*frame_num, h*w = image_len, dim]
                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
                if not do_crossattn:
                    batch_size, sequence_length, _ = hidden_states.shape  # 5120, 14, 320

                if do_crossattn:
                    batch_size, sequence_length, _ = encoder_hidden_states.shape
                args = () if USE_PEFT_BACKEND else (scale,)
                query = self.to_q(hidden_states, *args)
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states, *args)  # frame_num+frame_num / 2 / new_dim
                value = self.to_v(encoder_hidden_states, *args)  # frame_num+frame_num / 2 / new_dim

                inner_dim = key.shape[-1]
                head_dim = inner_dim // self.heads

                query = query.view(batch_size, -1, self.heads, head_dim).transpose(1,
                                                                                   2)  # batch / h*w*frame_num / head / dim1 -> batch / h*w*frame_num /  dim1 / head
                # batch*frame*h*w / frame_num / head / dim1
                # batch*frame*h*w / head / frame_num / dim1

                key = key.view(batch_size, -1, self.heads, head_dim).transpose(1,
                                                                               2)  # batch / 2*frame_num / head / dim2 -> batch / 2*frame_num / dim2 / head
                # batch*frame*h*w / 1 / head / dim2 -> self attention 에서는 batch*frame*h*w / frame_num / head / dim2
                # batch*frame*h*w / head / 1 / dim2 -> self attention 에서는 batch*frame*h*w / head / frame_num / dim2
                # batch*frame*h*w / head / dim2 / 1 -> self attention 에서는 batch*frame*h*w / head / dim2 / frame_num

                # 즉, cross attention 으로 올 때, frame_num 을 무시하고 k 와 v 가 들어가고 있다.
                # 만약 frame_num 개의 image_token 이 가능했다면, 각 frame 은 모든 frame 에 대한 condition 을 받게 된다.
                # 현재는 first frame 와 end frame 만 가능하므로, 2 frame 에 대해서 attention 을 받게 된다.
                # 여기서 attention mask 를 씌우는건 어떨까????

                value = value.view(batch_size, -1, self.heads, head_dim).transpose(1,
                                                                                   2)  # batch / 2*frame_num / head / dim3 -> batch / 2*frame_num / dim3 / head
                # batch*frame*h*w / 1 / head / dim2
                # batch*frame*h*w / head / 1 / dim2
                # batch*frame*h*w / head / dim2 / 1

                import math

                def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                                 is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:

                    def build_temporal_attention_mask(num_query_frames, num_kv_frames, device):

                        assert num_kv_frames == 2, "Expecting only start and end frames as KV"
                        weights = torch.linspace(0, 1, steps=num_query_frames, device=device)  # 0~1
                        mask = torch.stack([1 - weights, weights], dim=1)  # (T, 2)
                        return mask  # This can be added to QKᵀ before softmax

                    L, S = query.size(-2), key.size(-2)
                    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
                    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
                    attn_weight = query @ key.transpose(-2, -1) * scale_factor

                    # -------------------------------------------------------------------------------------
                    #
                    # -------------------------------------------------------------------------------------
                    # mask: (Q_len, K_len)
                    if controller.use_attentionmask and do_crossattn:
                        bias_mask = build_temporal_attention_mask(L, S, device=attn_weight.device)
                        bias_mask = bias_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, Q_len, K_len)

                        attn_weight = attn_weight + bias_mask  # Additive bias

                    # 분석 분기
                    if do_crossattn:
                        controller.temporal_cross_repeat += 1
                        frame_num = attn_weight.shape[-2]
                        senlen = attn_weight.shape[-1]

                        first_frame_sim = attn_weight[:, :, :, :int(senlen / 2)]  # shape: [B, H, F]
                        end_frame_sim = attn_weight[:, :, :, int(senlen / 2):]  # shape: [B, H, F]
                        for f in range(frame_num):
                            first_sim_val = first_frame_sim[:, :, f, :].mean().item()
                            end_sim_val = end_frame_sim[:, :, f, :].mean().item()
                            controller.set_startframe_attention(f, first_sim_val)
                            controller.set_endframe_attention(f, end_sim_val)

                    attn_weight += attn_bias
                    attn_weight = torch.softmax(attn_weight, dim=-1)
                    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
                    return attn_weight @ value

                hidden_states = scaled_dot_product_attention(query, key, value,
                                                             attn_mask=attention_mask,
                                                             dropout_p=0.0, is_causal=False)

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)
                # linear proj
                hidden_states = self.to_out[0](hidden_states, *args)
                # dropout
                hidden_states = self.to_out[1](hidden_states)
                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
                if self.residual_connection:
                    hidden_states = hidden_states + residual
                hidden_states = hidden_states / self.rescale_output_factor
                return hidden_states

            return forward

        class DummyController:

            def __call__(self, *args):
                return args[0]

            def __init__(self):
                self.num_att_layers = 0

        if controller is None:
            controller = DummyController()

        cross_att_count = 0
        # sub_nets = model.unet.named_children()
        sub_nets = model.named_modules()
        for net_name, net_class in sub_nets:
            module_class = net_class.__class__.__name__
            if module_class == 'Attention':
                if 'temporal' in net_name:
                    net_class.forward = ca_temporal_forward(net_class, controller, net_name)
        controller.num_att_layers = cross_att_count

    class AttnController:

        def __init__(self, name, use_attentionmask):
            self.temporal_self_repeat = 0
            self.temporal_cross_repeat = 0
            self.name = name
            self.startframe_attntion = {}
            self.endframe_attention = {}
            self.use_attentionmask = use_attentionmask

        def set_startframe_attention(self, frame_idx, value):
            if frame_idx not in self.startframe_attntion:
                self.startframe_attntion[frame_idx] = []
            self.startframe_attntion[frame_idx].append(value)

        def set_endframe_attention(self, frame_idx, value):
            if frame_idx not in self.endframe_attention:
                self.endframe_attention[frame_idx] = []
            self.endframe_attention[frame_idx].append(value)

        def reset(self):
            self.temporal_self_repeat = 0
            self.temporal_cross_repeat = 0
            self.startframe_attntion = {}
            self.endframe_attention = {}

    unet_controller = AttnController(name='unet_controller', use_attentionmask=args.use_attentionmask)
    register_attention_control(unet, unet_controller)

    class VAEFeatureProjector(nn.Module):

        def __init__(self, input_dim=1024,
                     output_dim=1024, apply_norm=True):
            super().__init__()

            self.linear = torch.nn.Linear(input_dim, output_dim)
            self.norm = torch.nn.LayerNorm(output_dim) if apply_norm else torch.nn.Identity()

            # 🟩 여기 추가
            self.config = {
                "input_dim": input_dim,
                "output_dim": output_dim,
                "apply_norm": apply_norm
            }

        def forward(self, x):
            out = self.linear(x)
            out = self.norm(out)
            return out

        def save_pretrained(self, save_directory):
            os.makedirs(save_directory, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
            torch.save(self.config, os.path.join(save_directory, "config.json"))

        @classmethod
        def from_pretrained(cls, load_directory):
            config_path = os.path.join(load_directory, "config.json")
            weights_path = os.path.join(load_directory, "pytorch_model.bin")

            config = torch.load(config_path)
            model = cls(**config)
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            return model

        def register_to_config(self, **kwargs):
            self.config.update(kwargs)

    print(f' when making projector with {args.projector_input_dim} dim')
    projector = VAEFeatureProjector(args.projector_input_dim)

    print(f"🚀 Step 8: Freezing non-trainable models")
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    unet.requires_grad_(True)

    def convert_module_dtype(module, dtype):
        for name, param in module.named_parameters(recurse=True):
            param.data = param.data.to(dtype)
        for name, buffer in module.named_buffers(recurse=True):
            buffer.data = buffer.data.to(dtype)
        return module

    vae = convert_module_dtype(vae, weight_dtype)

    if args.use_ema:
        print(f"🚀 Step 10: Setting up EMA")
        ema_unet = EMAModel(unet.parameters(),
                            model_cls=UNetSpatioTemporalConditionModel,
                            model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        print(f"🚀 Step 11: Enabling xformers memory efficient attention")
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "Please update xFormers to at least 0.0.17."
                )
            # unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly.")

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        print(f"🚀 Step 12: Registering custom hooks for saving/loading")

        def save_model_hook(models, weights, output_dir):
            model_name_dict = {
                id(accelerator.unwrap_model(unet)): "unet",
                id(accelerator.unwrap_model(projector)): "projector",
            }

            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for model in models:
                unwrapped_model = accelerator.unwrap_model(model)
                model_id = id(unwrapped_model)
                model_name = model_name_dict.get(model_id, None)

                if model_name:
                    save_path = os.path.join(output_dir, model_name)
                    unwrapped_model.save_pretrained(save_path)
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                ema_unet_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"),
                    UNetSpatioTemporalConditionModel,
                    custom_resume=True,
                )
                ema_unet.load_state_dict(ema_unet_model.state_dict())
                ema_unet.to(accelerator.device)
                del ema_unet_model

            for _ in range(len(models)):
                model = models.pop()
                unwrapped_model = accelerator.unwrap_model(model)
                model_id = id(unwrapped_model)

                if model_id == id(accelerator.unwrap_model(unet)):
                    load_model = UNetSpatioTemporalConditionModel.from_pretrained(os.path.join(input_dir, "unet"))
                elif model_id == id(accelerator.unwrap_model(projector)):
                    load_model = VAEFeatureProjector.from_pretrained(os.path.join(input_dir, "projector"))
                else:
                    continue

                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        print(f"🚀 Step 13: Enabling gradient checkpointing")
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        print(f"🚀 Step 14: Enabling TF32 support (Ampere GPUs)")
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        print(f"🚀 Step 15: Scaling learning rate")
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps *
                args.per_gpu_batch_size * accelerator.num_processes
        )

    print(f"🚀 Step 16: Initializing optimizer")
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam: pip install bitsandbytes")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    parameters_list = []
    # UNet 내부 temporal_transformer_block 파라미터만 학습 대상으로 지정

    for name, param in unet.named_parameters():
        if 'temporal_transformer_block' in name:
            parameters_list.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False

    # projector 파라미터도 학습 대상으로 지정
    for name, param in projector.named_parameters():
        parameters_list.append(param)
        param.requires_grad = True
    for name, param in motion_encoder.named_parameters():
        parameters_list.append(param)
        param.requires_grad = True

    # projector
    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    print(f"🚀 Step 17: Final training setup completed!")
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if not args.without_controlnet:
        controlnet.requires_grad_(False)
        controlnet.to(accelerator.device, dtype=weight_dtype)
        controlnet = convert_module_dtype(controlnet, weight_dtype)

    print(f"🚀 Step 18: Dataset")
    # Dataset & Dataloader 생성
    from data.motion_embedding import FrameInterpolationDataset
    train_dataset = FrameInterpolationDataset(sample_size=256,
                                              sample_n_frames=args.sample_n_frames,
                                              device=accelerator.device,
                                              dtype=weight_dtype,
                                              image_processor=feature_extractor,
                                              image_encoder=image_encoder,
                                              motion_encoder = motion_encoder,
                                              args=args)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=sampler,
                                                   batch_size=args.per_gpu_batch_size,
                                                   num_workers=args.num_workers, )
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Prepare everything with our `accelerator`.
    mask_token = unet.mask_token
    motion_encoder, projector, unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        motion_encoder, projector, unet, optimizer, lr_scheduler, train_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
                       accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def tensor_to_vae_latent(t, vae):
        video_length = t.shape[1]

        t = rearrange(t, "b f c h w -> (b f) c h w")
        latents = vae.encode(t).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
        latents = latents * vae.config.scaling_factor

        return latents

    def _encode_vae_image(image: torch.Tensor,
                          device, ):
        image = image.to(device=device)
        image_latents = vae.encode(image).latent_dist.mode()
        return image_latents

    model = torch.nn.Module()
    # model.unet = unet
    model.projector = projector
    # model.controlnet = controlnet

    for epoch in range(first_epoch, args.num_train_epochs):
        # unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            # with accelerator.accumulate(unet):
            with accelerator.accumulate(model):

                device = accelerator.device

                # --------------------------------------------------------------------------------------------------------------------
                # 1. input noisy latents
                # --------------------------------------------------------------------------------------------------------------------
                pixel_values = batch["video_pixel_values"].to(weight_dtype).to(accelerator.device,
                                                                               non_blocking=True)
                latents = tensor_to_vae_latent(pixel_values, vae)

                # ------------------------------------------------------------------------------------------------
                # 8.1 MSE loss
                # ------------------------------------------------------------------------------------------------
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                sigmas = rand_cosine_interpolated(shape=[bsz, ], image_d=image_d, noise_d_low=noise_d_low,
                                                  noise_d_high=noise_d_high,
                                                  sigma_data=sigma_data, min_value=min_value, max_value=max_value).to(
                    latents.device)
                sigmas_reshaped = sigmas.clone()

                while len(sigmas_reshaped.shape) < len(latents.shape):
                    sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)
                noisy_latents = latents + noise * sigmas_reshaped
                timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(latents.device)
                inp_noisy_latents = noisy_latents / ((sigmas_reshaped ** 2 + 1) ** 0.5)
                bsz, num_frames, _, latent_h, latent_w = inp_noisy_latents.shape
                dtype = latents.dtype

                # -------------------------------------- ------------------------------------------------------------------------------
                # 2. condition
                # --------------------------------------------------------------------------------------------------------------------
                # 2.1 image condition
                images = batch[
                    'image']  # batch,1(first),3,512,370 (infer 에서는 1,3,512,370) -> train 에서는 batch/1/3/512/370
                image_latents = torch.stack([_encode_vae_image(image, device) for image in images]).to(dtype)

                # also for image_end
                image_ends = batch['image_end']
                image_end_latents = torch.stack([_encode_vae_image(image, device) for image in image_ends]).to(dtype)

                # conditional_latents_mask = mask_token.repeat(bsz, num_frames - 2, 1, latent_h, latent_w).to(device,
                #                                                                                            dtype)
                conditional_latents_mask = torch.zeros((bsz, num_frames - 2, 4, latent_h, latent_w)).to(device, dtype)
                image_latents = torch.cat([image_latents, conditional_latents_mask, image_end_latents],
                                          dim=1)  # batch, total_frames, 4

                # 2.2 mask condition
                mask_channel = torch.ones_like(image_latents[:, :, 0:1, :, :])
                mask_channel[:, 0:1, :, :, :] = 0
                mask_channel[:, -1:, :, :, :] = 0
                # 2.3 total image condition
                image_latents = torch.cat([image_latents, mask_channel], dim=2)  # batch, total_frames, 1

                # --------------------------------------------------------------------------------------------------------------------
                # 3.1 Encode input image
                # --------------------------------------------------------------------------------------------------------------------
                image_embeddings = batch['image_embeddings']  # batch, length=1,1024
                image_end_embeddings = batch['image_end_embeddings']  # batch, length=1,1024
                spatial_image_embeddings = torch.cat([image_embeddings, image_end_embeddings],
                                                     dim=1)  # batch / 2 / 1024

                # --------------------------------------------------------------------------------------------------------------------
                # 3.2 Encode input image
                # --------------------------------------------------------------------------------------------------------------------
                image_embeddings_motion = batch['image_embeddings_motion']
                image_end_embeddings_motion = batch['image_end_embeddings_motion']
                image_embeddings_motion = motion_encoder(image_embeddings_motion)
                image_end_embeddings_motion = motion_encoder(image_end_embeddings_motion)
                temporal_image_embeddings = torch.cat([image_embeddings_motion, image_end_embeddings_motion],dim=-1)  # batch / 1 / 2048
                temporal_image_embeddings = projector(temporal_image_embeddings)

                image_embeddings = (spatial_image_embeddings, temporal_image_embeddings)

                # --------------------------------------------------------------------------------------------------------------------
                # 4. Controlnet Condition
                # --------------------------------------------------------------------------------------------------------------------
                controlnet_image = None  # heatmap image (batch, frame_num, 3, H, W)
                # batch_size, frames, channels, height, width
                point_embedding = None
                point_tracks = None  #

                # --------------------------------------------------------------------------------------------------------------------
                # * Motion Super Vision Loss
                # controlnet_image 를 frame 별로 모두 보게 되면, motion 의 움직이는 방향이 white 가 된다.
                # 즉, Frame1의 white 부분와 Frame2 의 white 부분의 semantic 이 일치해야 한다.
                # 생성 이미지의 크기는 현재 controlnet_image 보다 1/8씩 작다.
                # controlnet_image 의 size 를 downsize 시켜서 unet 의 결과인 model_pred 에서 결과와 곱해줌으로써 해당 부분을 얻을 수 있지 않을까???
                # --------------------------------------------------------------------------------------------------------------------

                # --------------------------------------------------------------------------------------------------------------------
                # 5. Get Added Time IDs
                # --------------------------------------------------------------------------------------------------------------------
                fps = 6
                motion_bucket_id = 100
                noise_aug_strength = 0.02

                def _get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, dtype, batch_size):
                    add_time_ids = torch.tensor([[fps, motion_bucket_id, noise_aug_strength]], dtype=dtype)
                    add_time_ids = add_time_ids.repeat(batch_size, 1)
                    return add_time_ids

                added_time_ids = _get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, weight_dtype, bsz)
                added_time_ids = added_time_ids.to(device)

                latent_model_input = torch.cat([inp_noisy_latents, image_latents], dim=2)
                controlnet_cond_scale = 1
                if args.without_controlnet:
                    kwargs = {}
                    output = unet(latent_model_input,
                                  timesteps,
                                  encoder_hidden_states=image_embeddings,
                                  down_block_additional_residuals=None,
                                  mid_block_additional_residual=None,
                                  added_time_ids=added_time_ids.to(dtype=weight_dtype),
                                  return_dict=False,
                                  **kwargs, )

                else:
                    down_block_res_samples, mid_block_res_sample = controlnet(latent_model_input,
                                                                              timesteps,  # problem here ! (batch 크기 만큼)
                                                                              encoder_hidden_states=image_embeddings,
                                                                              controlnet_cond=controlnet_image,
                                                                              added_time_ids=added_time_ids.to(
                                                                                  dtype=weight_dtype),
                                                                              conditioning_scale=controlnet_cond_scale,
                                                                              point_embedding=point_embedding,
                                                                              point_tracks=point_tracks,
                                                                              guess_mode=False,
                                                                              return_dict=False)
                    kwargs = {}
                    output = unet(latent_model_input,
                                  timesteps,
                                  encoder_hidden_states=image_embeddings,
                                  down_block_additional_residuals=down_block_res_samples,
                                  mid_block_additional_residual=mid_block_res_sample,
                                  added_time_ids=added_time_ids.to(dtype=weight_dtype),
                                  return_dict=False,
                                  **kwargs, )
                # why it is not noise_prediction ?
                model_pred, intermediate_features = output
                target = latents
                sigmas = sigmas_reshaped
                # Denoise the latents
                c_out = -sigmas / ((sigmas ** 2 + 1) ** 0.5)
                c_skip = 1 / (sigmas ** 2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = ((1 + sigmas ** 2) * (sigmas ** -2.0)).float()
                # ------------------------------------------------------------------------------------------------
                # 8.1 MSE loss
                # ------------------------------------------------------------------------------------------------
                target_diff = (denoised_latents.float() - target.float()) ** 2
                # loss mean, denoised_latents should be latent
                # denoised_latents -> shape = [batch, frame_nun, 4, h, w]
                loss = torch.mean((weighing * target_diff).reshape(target.shape[0], -1), dim=1, )  # batch shape loss
                loss = loss.mean()
                # ------------------------------------------------------------------------------------------------
                # 8.3 FrameMatching
                # ------------------------------------------------------------------------------------------------
                if args.frame_matching:
                    frame_matching_loss = 0.0
                    for i in range(num_frames):
                        model_frame = denoised_latents[:, i].float()
                        target_frame = target[:, i].float()

                        # 프레임당 MSE loss 사용 (원하면 L1로도 변경 가능)
                        frame_loss = torch.nn.functional.mse_loss(model_frame, target_frame)
                        frame_matching_loss += frame_loss

                    frame_matching_loss = frame_matching_loss / num_frames

                    # 기존 loss에 더함 (가중치 곱할 수도 있음: e.g., 0.1 * frame_matching_loss)
                    loss = loss + frame_matching_loss

                # ------------------------------------------------------------------------------------------------
                # 8.2 attention weight
                # ------------------------------------------------------------------------------------------------
                startframe_attntion_dict = unet_controller.startframe_attntion
                endframe_attention_dict = unet_controller.endframe_attention
                unet_controller.reset()

                start_attn_list = []
                end_attn_list = []

                for frame_idx in sorted(startframe_attntion_dict.keys()):
                    # float 리스트일 경우 -> tensor 리스트로 변환
                    start_vals = startframe_attntion_dict[frame_idx]
                    end_vals = endframe_attention_dict[frame_idx]

                    if isinstance(start_vals[0], float):
                        start_vals = [torch.tensor(v, device='cuda') for v in start_vals]
                        end_vals = [torch.tensor(v, device='cuda') for v in end_vals]

                    start_attn = torch.stack(start_vals).sum()
                    end_attn = torch.stack(end_vals).sum()

                    start_attn_list.append(start_attn)
                    end_attn_list.append(end_attn)

                if len(start_attn_list) > 1:
                    # 프레임 간 변화량 계산
                    start_diffs = torch.stack([
                        start_attn_list[i] - start_attn_list[i + 1]
                        for i in range(len(start_attn_list) - 1)
                    ])
                    end_diffs = torch.stack([
                        end_attn_list[i + 1] - end_attn_list[i]
                        for i in range(len(end_attn_list) - 1)
                    ])

                    # 꾸준한 감소: start attention
                    # → 평균 감소폭이 작으면 penalty, 변화량 std는 작을수록 좋음
                    start_mean_loss = torch.relu(0.05 - start_diffs.mean())  # 감소폭이 너무 작으면 손실
                    start_var_loss = start_diffs.std()

                    # 꾸준한 증가: end attention
                    end_mean_loss = torch.relu(0.05 - end_diffs.mean())  # 증가폭이 너무 작으면 손실
                    end_var_loss = end_diffs.std()

                    # 최종 Loss
                    lambda_attn = 1.0  # 조정 가능
                    attn_loss = lambda_attn * (start_mean_loss + start_var_loss + end_mean_loss + end_var_loss)

                    # 기존 loss와 합산
                    loss = loss + attn_loss

                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    wandb.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                # ------------------------------------------------------------------------------------------------------------ #
                # save checkpoints!
                # ------------------------------------------------------------------------------------------------------------ #
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(
                                checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                    # ------------------------------------------------------------------------------------------------------------ #
                    # Evaluation
                    # ------------------------------------------------------------------------------------------------------------ #
                    if accelerator.is_main_process and (
                            (global_step % args.validation_steps == 0) or (global_step == 1)):
                        logger.info(f"Running validation... \n Generating {args.num_validation_images} videos."
                                    )
                        if args.use_ema:
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        pipeline = StableVideoDiffusionInterpControlPipeline.from_pretrained(
                            "stabilityai/stable-video-diffusion-img2vid-xt",
                            unet=accelerator.unwrap_model(unet),
                            controlnet=accelerator.unwrap_model(controlnet),
                            image_encoder=accelerator.unwrap_model(image_encoder),
                            vae=accelerator.unwrap_model(vae),
                            revision=args.revision,
                            torch_dtype=weight_dtype, )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images")

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        with torch.no_grad():
                            with torch.autocast(str(accelerator.device).replace(":0", ""),
                                                enabled=accelerator.mixed_precision == "fp16"):
                                folders = ['008_flood', '009_flood', '010_flood']
                                for i, folder in enumerate(folders):
                                    base_image_dir = f'./assets/{folder}/input_frames'
                                    input_image_path = os.path.join(base_image_dir, 'flood_0.png')
                                    input_image_end_path = os.path.join(base_image_dir, 'flood_1.png')
                                    model_lengths = [14, 40]  # 생성할 프레임 수
                                    width, height = 512, 320
                                    device = "cuda" if torch.cuda.is_available() else "cpu"
                                    dtype = torch.float16

                                    # ⚡️ (4) 입력 이미지 전처리
                                    input_image = Image.open(input_image_path).convert('RGB').resize((width, height))
                                    input_image_end = Image.open(input_image_end_path).convert('RGB').resize(
                                        (width, height))

                                    for model_length in model_lengths:
                                        # ⚡️ (5) Trajectory (dummy)
                                        pred_tracks = torch.zeros(
                                            (1, model_length, 2))  # (1, num_frames, 2) -> (num_frames, 1,2)
                                        with_control = True

                                        # ⚡️ (6) Video 생성

                                        video_frames = pipeline(
                                            input_image,
                                            # ------------------------------------------------------------------------------------------- #
                                            input_image_end,
                                            # ------------------------------------------------------------------------------------------- #
                                            with_control=with_control,
                                            point_tracks=pred_tracks.permute(1, 0, 2).to(device, dtype),
                                            # (num_frames, num_points, 2)
                                            point_embedding=None,
                                            with_id_feature=False,
                                            controlnet_cond_scale=1.0,
                                            num_frames=model_length,
                                            width=width,
                                            height=height,
                                            motion_bucket_id=100,
                                            fps=7,
                                            num_inference_steps=30,
                                        ).frames[0]

                                        # ⚡️ (7) GIF/MP4로 저장
                                        validation_folder = os.path.join(args.output_dir, 'validation_videos')

                                        make_folder_with_permission(validation_folder)
                                        output_gif = os.path.join(validation_folder,
                                                                  f"sample_{i}_length_{model_length}_global_step_{global_step}.gif")
                                        video_frames[0].save(output_gif, save_all=True, append_images=video_frames[1:],
                                                             duration=100, loop=0)
                                        # log on wandb
                                        wandb.log({f"validation/sample_{i}": wandb.Video(output_gif,
                                                                                         caption=f"sample_{i}_length_{model_length}_global_step_{global_step}")},
                                                  step=global_step)
                        if args.use_ema:
                            ema_controlnet.restore(controlnet.parameters())
                        del pipeline
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:

        pipeline = StableVideoDiffusionInterpControlPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            controlnet=controlnet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    # wandb.init(project=args.wandb_project_name,
    #                        name=args.wandb_name)

    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument('--wandb_project_name', type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )  #
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--sample_n_frames",
        type=int,
        default=14,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
             " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=(
            "path to the dataset csv"
        ),
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help=(
            "path to the video folder"
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image_folder",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_control_folder",
        type=str,
        default=None,
        help=(
            "the validation control image"
        ),
    )  # use_attentionmask

    parser.add_argument('--use_attentionmask', action='store_true')
    parser.add_argument('--use_slerp_loss', action='store_true', help='Enable SLERP-based frame interpolation loss')
    parser.add_argument('--lambda_slerp', type=float, default=1.0, help='Weight for SLERP consistency loss')  #
    parser.add_argument('--no_point_tracks', action='store_true', help='disasble using point tracks')
    parser.add_argument('--frame_matching', action='store_true', help='frame image matching')  #
    parser.add_argument('--firstframe_conditioned', action='store_true')
    parser.add_argument('--endframe_conditioned', action='store_true')
    parser.add_argument('--without_controlnet', action='store_true')
    parser.add_argument("--projector_input_dim", type=int, default=2048)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    args = parser.parse_args()
    main(args)