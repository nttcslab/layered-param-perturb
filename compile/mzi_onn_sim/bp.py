import torch
from torch import Tensor

__all__ = ["forwardAD_PSBS", "forwardPSBS", "backwardPSBS", "forwardAD_PS_",
           "forwardPS", "backwardPS", "forwardmodReLU", "backwardmodReLU"]


def forwardAD_PSBS(input: Tensor, angleAB: Tensor, tangent: Tensor, indexAB: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.mzi_onn_sim_bp.forwardAD_PSBS(input, angleAB, tangent, indexAB)


def forwardPSBS(input: Tensor, angleAB: Tensor, indexAB: Tensor, split: Tensor, atten: Tensor) -> Tensor:
    return torch.ops.mzi_onn_sim_bp.forwardPSBS(input, angleAB, indexAB, split, atten)


def backwardPSBS(grad_output: Tensor, outputs: Tensor, input: Tensor, angleAB: Tensor,
                 indexAB: Tensor, split: Tensor, atten: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return torch.ops.mzi_onn_sim_bp.backwardPSBS(grad_output, outputs, input, angleAB, indexAB, split, atten)


def forwardAD_PS_(input: Tensor, angle: Tensor, tangent_input: Tensor, tangent_params: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.mzi_onn_sim_bp.forwardAD_PS_(input, angle, tangent_input, tangent_params)


def forwardPS(input: Tensor, angle: Tensor) -> Tensor:
    return torch.ops.mzi_onn_sim_bp.forwardPS(input, angle)


def backwardPS(grad_output: Tensor, input: Tensor, angle: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.mzi_onn_sim_bp.backwardPS(grad_output, input, angle)


def forwardmodReLU(input: Tensor, bias: Tensor) -> Tensor:
    return torch.ops.mzi_onn_sim_bp.forwardmodReLU(input, bias)


def backwardmodReLU(grad_output: Tensor, input: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    return torch.ops.mzi_onn_sim_bp.backwardmodReLU(grad_output, input, bias)
