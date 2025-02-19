import torch
import mzi_onn_sim
from . import chip


class ModulePS(torch.nn.Module):
    def __init__(self, nFeatures):
        super().__init__()
        self.params = torch.nn.Parameter(torch.zeros(nFeatures,))

    def forward(self, incmplx):
        return funcPS.apply(incmplx, self.params)

    def jvp(self, input, tangent_input, tangent_params):
        return mzi_onn_sim.bp.forwardAD_PS_(input, self.params, tangent_input, tangent_params[0])


class funcPS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, params):
        output = mzi_onn_sim.bp.forwardPS(input, params)
        ctx.save_for_backward(input, params)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, params = ctx.saved_tensors
        grad_input, grad_params = mzi_onn_sim.bp.backwardPS(grad_output, input, params)
        return grad_input, grad_params.sum(dim=0)


class ModulePSBS(torch.nn.Module):
    def __init__(self, nFeatures, index):
        super().__init__()
        self.nFeatures = nFeatures
        nLayers, nAngles, _ = index.shape
        self.params = torch.nn.Parameter(torch.randn(nLayers, nAngles))
        self.register_buffer('index', index, persistent=True)
        self.register_buffer('split', torch.zeros(nLayers, nAngles), persistent=True)
        self.register_buffer('atten', torch.ones(nLayers, nFeatures, dtype=torch.cfloat), persistent=True)

    def forward(self, cmplx):
        if hasattr(self, 'split_mask'):
            split = self.split * self.split_mask
        else:
            split = self.split
        return funcPSBS.apply(cmplx, self.params, self.index, split, self.atten)

    def jvp(self, input, tangent):
        return mzi_onn_sim.bp.forwardAD_PSBS(input, self.params, tangent[0], self.index)

    def calc_module_grads(self, input, nRandom, sample_wise=False):
        batch_size = input.shape[0]
        outputs = mzi_onn_sim.bp.forwardPSBS(input, self.params, self.index, self.split, self.atten)
        output = outputs[:, -1]
        grad_params = []
        for i in range(nRandom):
            dout = torch.randn_like(output)
            din, grad_param, grad_split, grad_atten = mzi_onn_sim.bp.backwardPSBS(
                dout, outputs, input, self.params, self.index, self.split, self.atten
            )
            grad_params.append(grad_param[None])
        grad_params = torch.cat(grad_params)
        effective_indexes = self.index[:, :, 0] >= 0
        grad_params = grad_params[:, :, effective_indexes]
        if sample_wise:
            correlation = torch.einsum('rta,rtb->tab', grad_params, grad_params) / nRandom
        else:
            correlation = torch.einsum('rta,rtb->ab', grad_params, grad_params) / nRandom / batch_size
        return correlation, effective_indexes.flatten()


class funcPSBS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, params, index, split, atten):
        outputs = mzi_onn_sim.bp.forwardPSBS(input, params, index, split, atten)
        ctx.save_for_backward(outputs, input, params, index, split, atten)
        return outputs[:, -1]

    @staticmethod
    def backward(ctx, grad_output):
        outputs, input, params, index, split, atten = ctx.saved_tensors
        grad_input, grad_params, _, _ = mzi_onn_sim.bp.backwardPSBS(
            grad_output, outputs, input, params, index, split, atten
        )
        return grad_input, grad_params.sum(dim=0), None, None, None


class ClementsPSBS(ModulePSBS):
    def __init__(self, nFeatures, nLayers):
        self.nLayers = nLayers
        indexA, indexB = chip._make_indexAB(nFeatures)
        indexAB = ([indexA] + [indexB]) * int(nLayers/2)
        if nLayers % 2 == 1:
            indexAB += [indexA]
        indexAB = torch.tensor(indexAB, dtype=torch.int32)
        super().__init__(nFeatures, indexAB)


class ClementsMZI(ModulePSBS):
    def __init__(self, nFeatures, nLayers):
        self.nLayers = nLayers
        indexA, indexB = chip._make_indexAB(nFeatures)
        indexAB = ([indexA] * 2 + [indexB] * 2) * int(nLayers/2)
        if nLayers % 2 == 1:
            indexAB += [indexA] * 2
        indexAB = torch.tensor(indexAB, dtype=torch.int32)
        super().__init__(nFeatures, indexAB)


class CmodReLU(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(input_size, dtype=torch.float))

    def forward(self, incmplx):
        return funcCmodReLU.apply(incmplx, self.bias)


class funcCmodReLU(torch.autograd.Function):
    ''' modReLU activation function '''
    @staticmethod
    def forward(ctx, incmplx, bias):
        ctx.save_for_backward(incmplx, bias)
        output = mzi_onn_sim.bp.forwardmodReLU(incmplx, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        incmplx, bias = ctx.saved_tensors
        grad_input, grad_bias = mzi_onn_sim.bp.backwardmodReLU(grad_output, incmplx, bias)
        return grad_input, grad_bias.sum(dim=0)


class ClementsMZI_PS(torch.nn.Module):
    def __init__(self, nFeatures, nLayers):
        super().__init__()
        self.nFeatures = nFeatures
        self.clements = ClementsMZI(nFeatures, nLayers)
        self.ps = ModulePS(nFeatures)

    def forward(self, input):
        output = self.clements(input)
        output = self.ps(output)
        return output

    def jvp(self, input, tangents):
        v_out, v_jvp = self.clements.jvp(input, (tangents[0],))
        v_out, v_jvp = self.ps.jvp(v_out, v_jvp, (tangents[1],))
        return v_out, v_jvp

    def calc_module_grads(self, input, nRandom, sample_wise=False):
        batch_size = input.shape[0]
        outputs = mzi_onn_sim.bp.forwardPSBS(
            input, self.clements.params, self.clements.index, self.clements.split, self.clements.atten
        )
        clements_output = outputs[:, -1]
        ps_output = mzi_onn_sim.bp.forwardPS(clements_output, self.ps.params)
        grad_params = []
        clements_indexes = self.clements.index[:, :, 0] >= 0
        ps_indexes = torch.full_like(self.ps.params, True, dtype=torch.bool)
        effective_indexes = torch.cat([clements_indexes.flatten(), ps_indexes])
        for i in range(nRandom):
            dout = torch.randn_like(ps_output)
            dout, grad_ps = mzi_onn_sim.bp.backwardPS(dout, clements_output, self.ps.params)
            _, grad_clements, _, _ = mzi_onn_sim.bp.backwardPSBS(
                dout, outputs, input, self.clements.params,
                self.clements.index, self.clements.split, self.clements.atten
            )
            grad_clements = grad_clements[:, clements_indexes]
            grad_tensor = torch.cat([grad_clements, grad_ps], dim=1)
            grad_params.append(grad_tensor[None])
        grad_params = torch.cat(grad_params)
        if sample_wise:
            correlation = torch.einsum('rta,rtb->tab', grad_params, grad_params) / nRandom
        else:
            correlation = torch.einsum('rta,rtb->ab', grad_params, grad_params) / nRandom / batch_size
        return correlation, effective_indexes
