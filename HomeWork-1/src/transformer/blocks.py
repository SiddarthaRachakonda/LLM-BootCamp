import torch.nn as nn

from components.norm_layers import RMSNorm
from components.attentions import EfficientSlidingWindowMultiheadAttention
from components.moe import MoeLayer


class TransformerBlock(nn.Module):
    def __init__(
          self, 
          hidden_size, 
          num_heads, 
          window_size, 
          d_ff, 
          num_experts, 
          n_experts_per_token,
          rotation_matrix
        ) -> None:
        super().__init__()

        # TODO: instantiate the different components

    def forward(self, x):
        # TODO: implement for the forward logic
        raise NotImplemented



