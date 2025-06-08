import os
import argparse
import torch
from PIL import Image
from models.controlnet_svd import ControlNetSVDModel
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from pipelines.pipeline_stable_video_diffusion_efficient_interp import StableVideoDiffusionInterpControlPipeline
from diffusers.utils.import_utils import is_xformers_available
import torchvision
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
import torch
import math
from typing import Any, Dict, Optional
from diffusers.utils import USE_PEFT_BACKEND

def make_folder_with_permission(path, exist_ok=True):
    os.makedirs(path, exist_ok=exist_ok)
    os.chmod(path, 0o777)  # 폴더 권한을 777로 변경

def linear_interpolate(img1, img2, alpha):
    img1_np = np.array(img1).astype(np.float32) / 255.0
    img2_np = np.array(img2).astype(np.float32) / 255.0
    blended = (1 - alpha) * img1_np + alpha * img2_np
    blended_uint8 = (blended * 255).astype(np.uint8)
    return Image.fromarray(blended_uint8)

def ensure_dirname(dirname):
    os.makedirs(dirname, exist_ok=True)


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


def main(args):

    torch.cuda.init()
    dummy = torch.randn(1).to("cuda")
    # ------------------------------------------------------------------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------------------------------------------------------------------
    print(f"🛠️ step 0. setup")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device = {device}')
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, }
    dtype = dtype_map[args.dtype]
    make_folder_with_permission(args.output_dir)

    # ------------------------------------------------------------------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------------------------------------------------------------------
    print(f"📦 step 2. load UNet")
    if not args.base:
        unet_path = os.path.join(args.model, "unet")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_path, torch_dtype=dtype,
                                                                low_cpu_mem_usage=True, custom_resume=True)
        print(f' loading model finish')

        #for module in unet.modules():
        #    module.to(device=device, dtype=dtype)
        #for param in unet.parameters():
        #    param.data = param.data.to(device, dtype=dtype)
        #for buffer in unet.buffers():
        #    buffer.data = buffer.data.to(device, dtype=dtype)

        # step 3. ControlNet 로드
        print(f"📦 step 3. load ControlNet")
        controlnet_path = os.path.join(args.model, "controlnet")
        if os.path.exists(controlnet_path):
            controlnet = ControlNetSVDModel.from_pretrained(controlnet_path)
        else :
            model_dir = "./checkpoints/framer_512x320"
            controlnet = ControlNetSVDModel.from_pretrained(os.path.join(model_dir, "controlnet"))
    else:
        model_dir = "./checkpoints/framer_512x320"
        unet = UNetSpatioTemporalConditionModel.from_pretrained(os.path.join(model_dir, "unet"),
                                                                torch_dtype=dtype,
                                                                low_cpu_mem_usage=True,
                                                                custom_resume=True, ).to(device, dtype)
        model_dir = "./checkpoints/framer_512x320"
        controlnet = ControlNetSVDModel.from_pretrained(os.path.join(model_dir, "controlnet"))


    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise RuntimeError("xformers is not available. Please install it.")

    print(f"📦 step 3. set pipeline")
    pipe = StableVideoDiffusionInterpControlPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",
                                                                     unet=unet,
                                                                     controlnet=controlnet,
                                                                     low_cpu_mem_usage=False,
                                                                     torch_dtype=dtype,
                                                                     variant="fp16",
                                                                     local_files_only=True,).to(device,dtype=dtype)
    if args.controlnet_attentioncontrol:
        base_output_dir = args.output_dir
        print(f"🚀 step 7. run inference")
        make_folder_with_permission(base_output_dir)

        def register_attention_control(model, controller):

            def ca_forward(self, controller) -> torch.FloatTensor:

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
                    """
                    if do_crossattn:
                        uncon_hidden_states, con_hidden_states = encoder_hidden_states.chunk(2, dim=0)

                        end_frame_condition = controller.end_frame_condition
                        first_frame_condition = controller.first_frame_condition
                        #print(f'first_frame_condition: {first_frame_condition} | end_frame_condition: {end_frame_condition}')
                        con_first_states = con_hidden_states[:,0, :] # 14 frame, 1, dim
                        con_end_states = con_hidden_states[:,1, :]
                        if con_first_states.ndim == 2:
                            con_first_states = con_first_states.unsqueeze(1)  # shape: [batch, 1, dim]
                            con_end_states = con_end_states.unsqueeze(1)
                        con_first_states = con_first_states[:first_frame_condition, :, :]  # [alpha,      1,dim]
                        con_end_states = con_end_states[:end_frame_condition, :, :]        # [batch-alpha,1,dim]
                        con_hidden_states = torch.cat([con_first_states, con_end_states], dim=0) # [batch,1,dim]
                        uncon_hidden_states = torch.zeros_like(con_hidden_states).to(device, dtype=dtype)
                        encoder_hidden_states = torch.cat([uncon_hidden_states, con_hidden_states], dim=0)
                        #print(f' [cross] new made encoder_hidden_states (28, 1, dim) = {encoder_hidden_states.shape}')
                    """
                    batch_size, sequence_length, _ = (
                        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)

                    if attention_mask is not None:
                        print(f'NOT HERE')
                        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                        attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

                    if self.group_norm is not None:
                        print(f'NOT HERE')
                        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

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

                    query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

                    key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                    value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

                    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                                     is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
                        L, S = query.size(-2), key.size(-2)
                        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
                        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
                        attn_weight = query @ key.transpose(-2, -1) * scale_factor

                        if do_crossattn:
                            print(f' Spatial Crossattention attn_weight = {attn_weight.shape}')
                            """
                            # k shape = batch / len=2 / dimension
                            # [batch, frame_num, dimension ] [batch, dimension, len=2]
                            controller.temporal_cross_repeat += 1
                            frame_num = attn_weight.shape[-2]
                            condition_len = attn_weight.shape[-1]


                            first_frame_sim = attn_weight[:, :, :, 0]  # shape: [B, H, F]
                            if condition_len == 2 :
                                end_frame_sim = attn_weight[:, :, :, 1]  # shape: [B, H, F]
                            for f in range(frame_num):
                                first_sim_val = first_frame_sim[:, :, f].mean().item()
                                if condition_len == 2:
                                    end_sim_val = end_frame_sim[:, :, f].mean().item()
                                    print(f"🧠 Frame {f:02d} similarity w.r.t First Frame: {first_sim_val:.4f} and  End Frame:   {end_sim_val:.4f}")
                                else :
                                    print(f"🧠 Frame {f:02d} similarity w.r.t First Frame: {first_sim_val:.4f}")

                                if 'controlnet' in controller.name :
                                    file = os.path.join(base_output_dir, 'controlnet_cross_anal.csv')
                                else :
                                    file = os.path.join(base_output_dir, 'unet_cross_anal.csv')
                                if not os.path.exists(file):
                                    with open(file, 'w') as f :
                                        f.write('frame_idx,FirstFrame_sim,EndFrame_sim\n')
                                else :
                                    if condition_len == 2:
                                        with open(file, 'a') as f:
                                            f.write(f'{f},{first_sim_val:.4f},{end_sim_val:.4f}\n')
                                    else :
                                        with open(file, 'a') as f:
                                            f.write(f'{f},{first_sim_val:.4f}\n')

                            # Cross Attention: 마지막 dim이 2 (first_frame, end_frame)일 때의 유사도 추출
                            first_frame_sim = attn_weight[:, :, :, 0]  # shape: [B, heads, L]
                            if condition_len == 2:
                                end_frame_sim = attn_weight[:, :, :, 1]  # shape: [B, heads, L]
                                print(f"🧠 Total First frame similarity: {first_frame_sim.mean().item()} End frame similarity: {end_frame_sim.mean().item()}", )
                            if 'controlnet' in controller.name:
                                file = os.path.join(base_output_dir, 'controlnet_cross_anal_total.csv')
                            else:
                                file = os.path.join(base_output_dir, 'unet_cross_anal_total.csv')

                            if not os.path.exists(file):
                                with open(file, 'w') as f :
                                    f.write('module_name,FirstFrame_sim,EndFrame_sim\n')
                            with open(file, 'a') as f:
                                if condition_len == 2:
                                    f.write(f'{module_name},{first_frame_sim},{end_frame_sim}\n')
                                else :
                                    f.write(f'{module_name},{first_frame_sim}\n')
                            """
                        attn_weight += attn_bias
                        attn_weight = torch.softmax(attn_weight, dim=-1)
                        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
                        return attn_weight @ value

                    hidden_states = scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                    )
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

            def ca_temporal_forward(self, controller, module_name) -> torch.FloatTensor:

                def forward(hidden_states: torch.FloatTensor,
                            encoder_hidden_states: Optional[torch.FloatTensor] = None,
                            attention_mask: Optional[torch.FloatTensor] = None,
                            temb: Optional[torch.FloatTensor] = None,
                            scale: float = 1.0, ):
                    do_crossattn = True
                    if encoder_hidden_states is None:
                        do_crossattn = False

                    # hidden_states = [batch*h*w / frame_num / dim]
                    print(f' start ca_temporal_forward, hidden_states : {hidden_states.shape}')
                    residual = hidden_states
                    input_ndim = hidden_states.ndim
                    if input_ndim == 4:
                        batch_size, channel, height, width = hidden_states.shape
                        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                    if not do_crossattn:
                        """ Temporal Self Attention """
                        batch_size, sequence_length, _ = hidden_states.shape  # 5120, 14, 320

                    if do_crossattn:
                        batch_size, sequence_length, _ = encoder_hidden_states.shape  # batch*h*w / img_len / dim

                    args = () if USE_PEFT_BACKEND else (scale,)
                    query = self.to_q(hidden_states, *args)  # batch*h*w / frame_num / dim1
                    if encoder_hidden_states is None:
                        encoder_hidden_states = hidden_states
                    elif self.norm_cross:
                        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                    key = self.to_k(encoder_hidden_states, *args)  # batch*h*w / 2 / new_dim
                    value = self.to_v(encoder_hidden_states, *args)  # batch*h*w / 2 / new_dim

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

                        # ------------------------------------------------------------------------------------------
                        # ------------------------------------------------------------------------------------------
                        # 분석 분기

                        if do_crossattn:

                            controller.temporal_cross_repeat += 1

                            frame_num = attn_weight.shape[-2]
                            condition_len = attn_weight.shape[-1]
                            first_frame_sim = attn_weight[:, :, :, 0]  # shape: [B, H, F]

                            if condition_len == 2:
                                end_frame_sim = attn_weight[:, :, :, 1]  # shape: [B, H, F]
                            for f in range(frame_num):
                                first_sim_val = first_frame_sim[:, :, f].mean().item()
                                if condition_len == 2:
                                    end_sim_val = end_frame_sim[:, :, f].mean().item()
                                    print(
                                        f"🧠 Frame {f:02d} similarity w.r.t First Frame: {first_sim_val:.4f} and  End Frame:   {end_sim_val:.4f}")
                                else:
                                    print(f"🧠 Frame {f:02d} similarity w.r.t First Frame: {first_sim_val:.4f}")

                                if 'controlnet' in controller.name:
                                    file = os.path.join(base_output_dir, 'controlnet_cross_anal.csv')
                                else:
                                    file = os.path.join(base_output_dir, 'unet_cross_anal.csv')
                                if not os.path.exists(file):
                                    with open(file, 'w') as f:
                                        f.write('frame_idx,FirstFrame_sim,EndFrame_sim\n')
                                else:
                                    if condition_len == 2:
                                        with open(file, 'a') as f:
                                            f.write(f'{f},{first_sim_val:.4f},{end_sim_val:.4f}\n')
                                    else:
                                        with open(file, 'a') as f:
                                            f.write(f'{f},{first_sim_val:.4f}\n')

                            # Cross Attention: 마지막 dim이 2 (first_frame, end_frame)일 때의 유사도 추출
                            first_frame_sim = attn_weight[:, :, :, 0]  # shape: [B, heads, L]
                            if condition_len == 2:
                                end_frame_sim = attn_weight[:, :, :, 1]  # shape: [B, heads, L]
                                print(
                                    f"🧠 Total First frame similarity: {first_frame_sim.mean().item()} End frame similarity: {end_frame_sim.mean().item()}", )
                            if 'controlnet' in controller.name:
                                file = os.path.join(base_output_dir, 'controlnet_cross_anal_total.csv')
                            else:
                                file = os.path.join(base_output_dir, 'unet_cross_anal_total.csv')

                            if not os.path.exists(file):
                                with open(file, 'w') as f:
                                    f.write('module_name,FirstFrame_sim,EndFrame_sim\n')
                            with open(file, 'a') as f:
                                if condition_len == 2:
                                    f.write(f'{module_name},{first_frame_sim},{end_frame_sim}\n')
                                else:
                                    f.write(f'{module_name},{first_frame_sim}\n')
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
                    # if 'temporal' not in net_name :
                    #    net_class.forward = ca_forward(net_class, controller)
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

        unet_controller = AttnController('unet_controller', args.use_attentionmask)
        # controlnet_controller = AttnController('controlnet_controller', False)
        # register_attention_control(controlnet, controlnet_controller)
        register_attention_control(unet, unet_controller)

        pipe.unet = unet
        pipe.controlnet = controlnet

    from torch import nn

    class VAEFeatureProjector(nn.Module):

        def __init__(self, input_dim=1024, output_dim=1024, apply_norm=True):
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
            """
            Args:
                x: Tensor of shape (batch, N, 4)
            Returns:
                Tensor of shape (batch, N, 1024)
            """
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
    if args.use_projector :
        projector = VAEFeatureProjector.from_pretrained(os.path.join(args.model, "projector")).to(device, dtype)
    else :
        projector = None

    # ------------------------------------------------------------------------------------------------------------------------------
    # 2. Image Loading
    # ------------------------------------------------------------------------------------------------------------------------------
    base_folder = './assets'
    folders = os.listdir(base_folder)
    for folder in folders:

        output_dir = os.path.join(args.output_dir, folder)
        make_folder_with_permission(output_dir)

        input_folder = os.path.join(base_folder, folder, 'input_frames')
        first_frame_path = os.path.join(input_folder, 'image_0.png')
        end_frame_path = os.path.join(input_folder, 'image_1.png')
        first_image = Image.open(first_frame_path).convert("RGB").resize((args.width, args.height))
        last_image = Image.open(end_frame_path).convert("RGB").resize((args.width, args.height))

        print(f"🛠️ step 1. setup & input check")
        if not args.with_no_track :
            track_file = os.path.join(input_folder, 'track.txt')
            with open(track_file, 'r') as f:
                lines = f.readlines()
            start_x,start_y  = lines[0].strip().split(',')
            start_x, start_y = float(start_x.strip()), float(start_y.strip())
            end_x,end_y      = lines[1].strip().split(',')
            end_x, end_y = float(end_x.strip()), float(end_y.strip())
            input_all_points = [[[start_x, start_y], [end_x, end_y]]]
            if len(lines) > 2:
                start_x2, start_y2 = lines[2].strip().split(',')
                start_x2, start_y2 = float(start_x2.strip()), float(start_y2.strip())
                end_x2, end_y2 = lines[3].strip().split(',')
                end_x2, end_y2 = float(end_x2.strip()), float(end_y2.strip())
                input_all_points = [[[start_x, start_y], [end_x, end_y]],
                                    [[start_x2, start_y2], [end_x2, end_y2]]]
            elif len(lines) > 2:
                args.output_dir = f'{args.output_dir}_multi_track_condition'
            original_width = args.width
            original_height = args.height
            resized_all_points = [tuple([tuple([int(e1[0] * args.width / original_width), int(e1[1] * args.height / original_height)]) for e1 in e]) for
                                  e in input_all_points]
            for idx, splited_track in enumerate(resized_all_points):
                if len(splited_track) == 1:  # stationary point
                    displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
                    splited_track = tuple([splited_track[0], displacement_point])
                splited_track = interpolate_trajectory(splited_track, args.num_frames)
                splited_track = splited_track[:args.num_frames]
                resized_all_points[idx] = splited_track
            pred_tracks = torch.tensor(resized_all_points)  # [1,frame_num,2]
            pred_tracks = pred_tracks.permute(1, 0, 2).to(device, dtype)  # [1,frame_num,2]

        else :
            pred_tracks = None

        with_control = not args.without_controlnet
        point_embedding = None
        sift_track_update = False
        anchor_points_flag = None

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
            firstframe_conditioned=args.without_end_embedding,  # True일 때 end embedding 사용
            endframe_conditioned=args.without_front_embedding,  # True일 때 front embedding 사용
            without_front_channel=args.without_front_channel,  # True일 때 front channel 사용
            without_end_channel= args.without_end_channel,  # True일 때 end channel 사용
            anchor_points_flag=anchor_points_flag,
            last_channel_lastframe_condition = args.last_channel_lastframe_condition,
            seoncd_channel_firstframe_condition =args.seoncd_channel_firstframe_condition,
            channelwise_all_front = args.channelwise_all_front,
            channelwise_all_end = args.channelwise_all_end,
                            projector = projector,
                            do_motion_prompt = args.do_motion_prompt
        ).frames[0]

        for i, frame in enumerate(video_frames):
            save_path = os.path.join(output_dir, f"{folder}_frame_{i}.png")
            frame.save(save_path)
        duration = 100
        video_frames[0].save(os.path.join(output_dir, f"{folder}.gif"), save_all=True, append_images=video_frames[1:], loop=0, duration=duration)
        if args.controlnet_attentioncontrol:
            unet_controller.reset()


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
    parser.add_argument("--firstframe_conditioned", action="store_true")
    parser.add_argument("--endframe_conditioned", action="store_true")
    parser.add_argument("--slerp_latent_mask", action="store_true") # slerp_latent_crossattn
    parser.add_argument("--slerp_latent_crossattn", action="store_true")  #
    parser.add_argument("--with_no_track", action="store_true")
    parser.add_argument("--use_attentionmask", action="store_true")
    parser.add_argument("--controlnet_attentioncontrol", action="store_true")
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--use_edge", action="store_true")
    parser.add_argument('--without_front_embedding', action='store_true', help="Disable front embedding condition")
    parser.add_argument('--without_end_embedding', action='store_true', help="Disable end embedding condition")
    parser.add_argument('--without_front_channel', action='store_true', help="Disable front channel condition")
    parser.add_argument('--without_end_channel', action='store_true', help="Disable end channel condition")
    parser.add_argument('--seoncd_channel_firstframe_condition', action='store_true', help="Disable front channel condition")
    parser.add_argument('--last_channel_lastframe_condition', action='store_true', help="Disable end channel condition")
    parser.add_argument('--channelwise_all_front', action='store_true')
    parser.add_argument('--channelwise_all_end', action='store_true')
    parser.add_argument('--without_controlnet', action='store_true') # do_motion_prompt
    parser.add_argument('--do_motion_prompt', action='store_true')  # use_projector
    parser.add_argument('--use_projector', action='store_true')  #
    args = parser.parse_args()
    main(args)
