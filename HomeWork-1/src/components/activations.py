import torch
import torch.nn as nn


class SiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # TODO: create 2 linear layers
  
    def forward(self, x):
        # TODO: implement SiGLU W * x * sigma (W_g * x)
        raise NotImplemented