import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff):
        super().__init__()
        # TODO: instantiate 3 linear layers

    def forward(self, x) -> torch.Tensor:
        # TODO: implement the expert logic
        raise NotImplemented


class MoeLayer(nn.Module):
    def __init__(self, hidden_size, d_ff, num_experts, n_experts_per_token):
        super().__init__()

        self.num_experts = num_experts
        self.n_experts_per_token = n_experts_per_token

        # TODO: instantiate the experts and the gate
        self.experts = None 
        self.gate = None 

    def forward(self, x):
        # TODO: pass the input x to the gate
        # TODO: use torch.topk to get the topk values and indexes
        # TODO: pass the topk values to F.softmax to get the weights for each expert 
  
        out = torch.zeros_like(x, device=x.device)
        for i, expert in enumerate(self.experts):
            # TODO: find the indexes of the hidden states that should be routed to the current expert
            # TODO: update the out tensor
            pass
        return out