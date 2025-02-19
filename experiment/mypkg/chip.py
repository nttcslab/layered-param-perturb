import torch
import numpy as np
import mzi_onn_sim
from dataclasses import dataclass


class ModulePS(torch.nn.Module):
    def __init__(self, nFeatures, deficient):
        super().__init__()
        self.params = torch.nn.Parameter(torch.zeros(1, nFeatures))
        radius = 1.0 - deficient.eps_radius * torch.rand(nFeatures)
        phi = deficient.eps_angle * (2.0 * torch.rand(nFeatures) - 1)
        self.register_buffer('atten', radius*torch.exp(1.j*phi), persistent=True)

    def forward(self, input):
        return mzi_onn_sim.zo.forwardPS(input, self.params, self.atten)


class ModulePSBS(torch.nn.Module):
    def __init__(self, nFeatures, index, deficient):
        super().__init__()
        self.nFeatures = nFeatures
        nLayers, nAngles, _ = index.shape
        self.params = torch.nn.Parameter(torch.randn(1, nLayers, nAngles))
        self.register_buffer('index', index, persistent=True)
        self.register_buffer('split', torch.randn(nLayers, nAngles, dtype=torch.float)*deficient.split, persistent=True)
        radius = 1.0 - deficient.eps_radius * torch.rand(nLayers, nFeatures)
        phi = deficient.eps_angle * (2.0 * torch.rand(nLayers, nFeatures) - 1)
        self.register_buffer('atten', radius*torch.exp(1.j*phi), persistent=True)

    def forward(self, input):
        return mzi_onn_sim.zo.forwardPSBS(input, self.params, self.index, self.split, self.atten)


class ClementsPSBS(ModulePSBS):
    def __init__(self, nFeatures, nLayers, deficient):
        self.nLayers = nLayers
        indexA, indexB = _make_indexAB(nFeatures)
        indexAB = ([indexA] + [indexB]) * int(nLayers/2)
        if nLayers % 2 == 1:
            indexAB += [indexA]
        indexAB = torch.tensor(indexAB, dtype=torch.int32)
        super().__init__(nFeatures, indexAB, deficient)


class ClementsMZI(ModulePSBS):
    def __init__(self, nFeatures, nLayers, deficient):
        self.nLayers = nLayers
        indexA, indexB = _make_indexAB(nFeatures)
        indexAB = ([indexA] * 2 + [indexB] * 2) * int(nLayers/2)
        if nLayers % 2 == 1:
            indexAB += [indexA] * 2
        indexAB = torch.tensor(indexAB, dtype=torch.int32)
        super().__init__(nFeatures, indexAB, deficient)


class ClementsMZI_PS(torch.nn.Module):
    def __init__(self, nFeatures, nLayers, deficient):
        super().__init__()
        self.nFeatures = nFeatures
        self.clements = ClementsMZI(nFeatures, nLayers, deficient)
        self.ps = ModulePS(nFeatures, deficient)

    def forward(self, input):
        output = self.clements(input)
        output = self.ps(output)
        return output


class CmodReLU(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros((1, input_size), dtype=torch.float))

    def forward(self, input):
        return mzi_onn_sim.zo.forwardmodReLU(input, self.bias)


@dataclass
class Deficient:
    split: float = 0
    eps_radius: float = 0
    eps_angle: float = 0

    def set_deficient(self, cfg_deficient, cfg_defmul):
        self.split = cfg_defmul * cfg_deficient.split
        self.eps_angle = cfg_defmul * cfg_deficient.eps_angle
        self.eps_radius = cfg_defmul * cfg_deficient.eps_radius


def _make_indexAB(nFeatures):
    if nFeatures % 2 == 0:
        nAnglesAB = int(nFeatures/2)
        indexA = [[i*2, i*2+1] for i in range(nAnglesAB)]
        indexB = [[i*2+1, i*2+2] for i in range(nAnglesAB-1)] + [[np.invert(0), np.invert(nFeatures-1)]]
    else:
        nAnglesAB = int(nFeatures/2) + 1
        indexA = [[i*2, i*2+1] for i in range(nAnglesAB-1)] + [[np.invert(nFeatures-1)] * 2]
        indexB = [[i*2+1, i*2+2] for i in range(nAnglesAB-1)] + [[np.invert(0)] * 2]
    return indexA, indexB
