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
from models.feature_projector import FeatureProjector
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
from data import FrameInterpolationDataset
import torch.multiprocessing as mp
from pipelines.pipeline_stable_video_diffusion_efficient_interp import StableVideoDiffusionEfficientInterpPipeline
from utils import make_folder_with_permission

mp.set_start_method("spawn", force=True)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")
logger = get_logger(__name__, log_level="INFO")

min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5

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

def main(args):
    print("🚀 Step 1: Training setup")

    # (1) 인자 저장
    argument_dict = vars(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(argument_dict, f, indent=4)

    # (2) 비권장 옵션 경고
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. "
                "Please use `--variant=non_ema` instead."
            ),
        )

    # (3) Accelerator 초기화
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=logging_dir
        ),
    )

    # (4) Weights & Biases 설정
    if args.report_to == "wandb":
        print("🚀 Step 2: Initializing wandb")
        if not is_wandb_available():
            raise ImportError("Install wandb to use it for logging.")
        import wandb
        if accelerator.is_main_process:
            wandb.init(
                project=args.wandb_project_name,
                name=args.wandb_name,
            )

    # (5) 로깅 설정
    print("🚀 Step 3: Logging setup")
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

    # (6) 시드 설정
    print("🚀 Step 4: Setting seed")
    set_seed(args.seed)

    # (7) HuggingFace Hub에 푸시
    if accelerator.is_main_process and args.push_to_hub:
        print("🚀 Step 5: Pushing to HuggingFace Hub")
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name,
            exist_ok=True,
            token=args.hub_token,
        ).repo_id

    print("🚀 Step 6: Loading models and scheduler")

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="image_encoder",
        revision=args.revision,
        variant="fp16"
    )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant="fp16"
    )

    # Set weight dtype depending on precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print("🚀 Step 7: Loading UNet")
    model_dir = args.resume_from_checkpoint or "./checkpoints/framer_512x320"
    global_step = 0
    if args.resume_from_checkpoint:
        check_point_name = os.path.basename(model_dir)
        global_step = int(check_point_name.split('-')[-1])

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        os.path.join(model_dir, "unet"),
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
        custom_resume=True
    )

    # Load projector
    projector = FeatureProjector(args.projector_input_dim)

    print("🚀 Step 8: Freezing non-trainable models")
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder = image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(True)

    # Utility: convert parameters/buffers to correct dtype
    def convert_module_dtype(module, dtype):
        for _, param in module.named_parameters(recurse=True):
            param.data = param.data.to(dtype)
        for _, buffer in module.named_buffers(recurse=True):
            buffer.data = buffer.data.to(dtype)
        return module

    vae = convert_module_dtype(vae, weight_dtype)

    if args.use_ema:
        print("🚀 Step 10: Setting up EMA")
        ema_unet = EMAModel(
            unet.parameters(),
            model_cls=UNetSpatioTemporalConditionModel,
            model_config=unet.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        print("🚀 Step 11: Enabling xformers memory efficient attention")
        if not is_xformers_available():
            raise ValueError("xformers is not available. Please install it.")
        import xformers
        if version.parse(xformers.__version__) == version.parse("0.0.16"):
            logger.warning(
                "xFormers 0.0.16 has known issues on some GPUs. Consider upgrading to 0.0.17 or later."
            )
        # unet.enable_xformers_memory_efficient_attention()  # Uncomment if support exists

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        print("🚀 Step 12: Registering custom hooks for saving/loading")

        def save_model_hook(models, weights, output_dir):
            model_name_dict = {id(accelerator.unwrap_model(unet)): "unet",id(accelerator.unwrap_model(projector)): "projector",
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
                    load_model = FeatureProjector.from_pretrained(os.path.join(input_dir, "projector"))
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
                args.per_gpu_batch_size * accelerator.num_processes)

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
    for name, param in unet.named_parameters():
        if 'temporal_transformer_block' in name:
            parameters_list.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, param in projector.named_parameters():
        parameters_list.append(param)
        param.requires_grad = True

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

    print(f"🚀 Step 18: Dataset")
    train_dataset = FrameInterpolationDataset(sample_size=256,
                                              sample_n_frames=args.sample_n_frames,
                                              device=accelerator.device,
                                              dtype=weight_dtype,
                                              image_processor=feature_extractor,
                                              image_encoder=image_encoder,
                                              args=args)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=sampler,
                                                   batch_size=args.per_gpu_batch_size,
                                                   num_workers=args.num_workers, )
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Prepare everything with our `accelerator`.
    mask_token = unet.mask_token
    projector, unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(projector, unet, optimizer, lr_scheduler, train_dataloader)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
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
    logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    first_epoch = 0
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
    model.unet = unet
    model.projector = projector

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
                sigmas = rand_cosine_interpolated(shape=[bsz, ], image_d=image_d, noise_d_low=noise_d_low, noise_d_high=noise_d_high,
                                                  sigma_data=sigma_data, min_value=min_value, max_value=max_value).to(latents.device)
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
                images = batch['image']  # batch,1(first),3,512,370 (infer 에서는 1,3,512,370)
                image_latents = torch.stack([_encode_vae_image(image, device) for image in images]).to(dtype)

                # also for image_end
                image_ends = batch['image_end']
                image_end_latents = torch.stack([_encode_vae_image(image, device) for image in image_ends]).to(dtype)

                conditional_latents_mask = mask_token.repeat(bsz, num_frames - 2, 1, latent_h, latent_w).to(device,
                                                                                                            dtype)
                # conditional_latents_mask =torch.zeros((bsz, num_frames-2, 4, latent_h, latent_w)).to(device, dtype)
                image_latents = torch.cat([image_latents, conditional_latents_mask, image_end_latents],
                                          dim=1)  # batch, total_frames, 4

                # 2.2 mask condition
                mask_channel = torch.ones_like(image_latents[:, :, 0:1, :, :])
                mask_channel[:, 0:1, :, :, :] = 0
                mask_channel[:, -1:, :, :, :] = 0
                # 2.3 total image condition
                image_latents = torch.cat([image_latents, mask_channel], dim=2)  # batch, total_frames, 1
                # --------------------------------------------------------------------------------------------------------------------
                # 3. Encode input image
                # --------------------------------------------------------------------------------------------------------------------
                image_embeddings = batch['image_embeddings']  # batch, length=1,1024
                image_end_embeddings = batch['image_end_embeddings']  # batch, length=1,1024
                spatial_image_embeddings = torch.cat([image_embeddings, image_end_embeddings],dim=1)  # batch / 2 / 1024
                temporal_image_embeddings = torch.cat([image_embeddings, image_end_embeddings],dim=-1)  # batch / 1 / 2048
                temporal_image_embeddings = projector(temporal_image_embeddings)
                image_embeddings = (spatial_image_embeddings, temporal_image_embeddings)

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
                kwargs = {}
                output = unet(latent_model_input,
                              timesteps,
                              encoder_hidden_states=image_embeddings,
                              down_block_additional_residuals=None,
                              mid_block_additional_residual=None,
                              added_time_ids=added_time_ids.to(dtype=weight_dtype),
                              return_dict=False,
                              **kwargs, )

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
                """
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
                """
                avg_loss = accelerator.gather(loss.repeat(args.per_gpu_batch_size)).mean()
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
                        pipeline = StableVideoDiffusionEfficientInterpPipeline.from_pretrained(
                            "stabilityai/stable-video-diffusion-img2vid-xt",
                            unet=accelerator.unwrap_model(unet),
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
                        del pipeline
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = StableVideoDiffusionEfficientInterpPipeline.from_pretrained(args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            revision=args.revision,)
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