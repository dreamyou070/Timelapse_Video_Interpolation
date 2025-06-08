import os
import numpy as np
from torch import nn
import torch

class FeatureProjector(nn.Module):

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