import torch.nn as nn
from foundation_lab.architecture.registry import NORMS

@NORMS.register("rmsnorm")
def rmsnorm(dim: int) -> nn.Module:
    return nn.RMSNorm(dim, eps=1e-6)
