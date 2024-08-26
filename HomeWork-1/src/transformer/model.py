import torch.nn as nn

from transformer.blocks import TransformerBlock
from components.rope import get_rotation_matrix


class Transformer(nn.Module):
    def __init__(
            self,
            vocabulary_size,
            hidden_size, 
            num_heads, 
            window_size, 
            d_ff, 
            num_experts, 
            n_experts_per_token, 
            n_blocks,
            max_seq_len
        ):

        super().__init__()

        head_dim = hidden_size // num_heads
        period = 10000.0
        self.rotation_matrix = get_rotation_matrix(head_dim, max_seq_len, period)

        # TODO: instantiate the components
        self.embedding = None
        self.blocks = None
        self.out = None

    def forward(self, x):
        # TODO: implement for the forward method
        raise NotImplemented