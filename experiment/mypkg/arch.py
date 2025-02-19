import numpy as np
import torch
from . import chip_bp, chip


def get_model(cfg):
    deficient = chip.Deficient()
    deficient.set_deficient(cfg.deficient, cfg.defmul)
    nFeatures = cfg.net.num_features
    nLayers = cfg.net.num_layers
    preprocess = PreprocessFFT(cfg.net.num_features)
    net_model = FF3layersClements('ClementsMZI', nFeatures, nLayers)
    net_chip = FF3layersClements('ClementsMZI', nFeatures, nLayers, deficient)
    return net_model, net_chip, preprocess


def target_parameters(net):
    for p in net.parameters():
        if p.requires_grad:
            yield p


def copy_parameters(net_model, net_chip):
    for pm, pc in zip(target_parameters(net_model), target_parameters(net_chip)):
        pc.data[0] = pm


def copy_deficient(net_chip, net_model):
    for modc, modm in zip(net_chip.modules(), net_model.modules()):
        if isinstance(modm, chip_bp.ModulePSBS):
            modm.split.data = modc.split.data
            modm.atten.data = modc.atten.data


class NetBase(torch.nn.Module):
    def __init__(self, nFeatures, variation=False):
        super().__init__()
        nEffectiveOutputs = 10
        out_begin = int((nFeatures - nEffectiveOutputs)/2)
        self.output_select = np.arange(out_begin, out_begin + nEffectiveOutputs)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = NetBase.calc_accuracy
        self.output_scale = 1
        self.variation = variation

    def forward(self, x):
        for mod in self.module_list:
            x = mod(x)
        z = self.output_scale * x.abs()**2
        if self.variation:
            z_sel = z[:, :, self.output_select]
        else:
            z_sel = z[:, self.output_select]
        return z_sel

    @staticmethod
    def calc_accuracy(output, label):
        return torch.eq(torch.argmax(output, dim=1), label).float().mean()


class FF3layersClements(NetBase):
    def __init__(self, name, nFeatures, nLayers, deficient=None):
        if deficient is not None:
            super().__init__(nFeatures, variation=True)
            self.module_list = torch.nn.ModuleList([
                getattr(chip, name)(nFeatures, nLayers, deficient),
                chip.CmodReLU(nFeatures),
                getattr(chip, name)(nFeatures, nLayers, deficient),
                chip.CmodReLU(nFeatures),
                getattr(chip, name)(nFeatures, nLayers, deficient)
            ])
        else:
            super().__init__(nFeatures)
            self.module_list = torch.nn.ModuleList([
                getattr(chip_bp, name)(nFeatures, nLayers),
                chip_bp.CmodReLU(nFeatures),
                getattr(chip_bp, name)(nFeatures, nLayers),
                chip_bp.CmodReLU(nFeatures),
                getattr(chip_bp, name)(nFeatures, nLayers)
            ])


class PreprocessFFT(torch.nn.Module):
    def __init__(self, nSelect):
        super().__init__()
        self.input_select = permute_idx(torch.arange(1, nSelect+1))

    def forward(self, x):
        batch_size = x.shape[0]
        y = torch.fft.rfft(x.view(batch_size, -1), norm="ortho")
        return y[:, self.input_select]


def permute_idx(sort_idx):
    even_idx = sort_idx[::2]
    odd_idx = sort_idx[1::2]
    reverse_odd_idx = torch.flip(odd_idx, dims=[0])
    selected = torch.cat((reverse_odd_idx, even_idx), dim=0)
    return selected
