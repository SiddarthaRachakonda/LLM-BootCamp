import torch
import torch.nn as nn
from typing import Tuple


def get_rotation_matrix(dim: int, context_size: int, period: float) -> torch.Tensor:
    # TODO: compute a tensor of frequencies
    freqs = None
    # TODO: compute a tensor of token indexes
    token_indexes = None
    # TODO: compute the matrix thetas
    thetas = None
    # TODO: create the rotation matrix
    rotation_matrix = None
    return rotation_matrix


class RoPE(nn.Module):
    def __init__(self, rotation_matrix):
        super().__init__()
        self.rotation_matrix = rotation_matrix

    def forward(self, queries, keys):
        batch_size, num_heads, seq_length, head_dim = queries.size()

        # TODO: reshape to [batch_size, num_heads, seq_length, head_dim // 2 , 2]
        queries = None
        keys = None

        # TODO: transform into a complex tensor
        queries_complex = None
        keys_complex = None

        # TODO: rotate the queries and keys
        queries_rotated = None
        keys_rotated = None

        # TODO: conver to read and reshape back to [batch_size, num_heads, seq_length, head_dim]
        new_queries = None
        new_keys = None

        return new_queries, new_keys











def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)