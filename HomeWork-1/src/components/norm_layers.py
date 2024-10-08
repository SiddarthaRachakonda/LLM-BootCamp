import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        # TODO: implement the normalization
        raise NotImplemented

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight