import torch
from torch import Tensor

__all__ = ["forwardPSBS", "forwardPS", "forwardmodReLU"]


def forwardPSBS(input: Tensor, angleAB: Tensor, indexAB: Tensor, split: Tensor, atten: Tensor) -> Tensor:
    return torch.ops.mzi_onn_sim_zo.forwardPSBS(input, angleAB, indexAB, split, atten)


def forwardPS(input: Tensor, angle: Tensor, atten: Tensor) -> Tensor:
    return torch.ops.mzi_onn_sim_zo.forwardPS(input, angle, atten)


def forwardmodReLU(input: Tensor, bias: Tensor) -> Tensor:
    return torch.ops.mzi_onn_sim_zo.forwardmodReLU(input, bias)
