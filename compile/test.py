# A very simple script for checking mzi_onn_sim
import torch
import mzi_onn_sim

num_features = 2
num_samples = 3
num_variations = 4


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


def test_bp(device):
    input = torch.randn((num_samples, num_features), dtype=torch.cfloat, device=device)
    param = torch.nn.Parameter(torch.randn((num_features), dtype=torch.float, device=device))
    output = funcPS.apply(input, param)
    target = torch.randn_like(output)
    error = output - target
    loss = (error * error.conj()).real.sum()
    loss.backward()
    print(loss)
    print(param.grad)


def test_zo(device):
    input = torch.randn((num_variations, num_samples, num_features), dtype=torch.cfloat, device=device)
    param = torch.nn.Parameter(torch.randn((num_variations, num_features), dtype=torch.float, device=device))
    atten = torch.randn((num_features), dtype=torch.cfloat, device=device)
    output = mzi_onn_sim.zo.forwardPS(input, param, atten)
    print(output)


test_bp('cpu')
test_bp('cuda')
test_zo('cpu')
test_zo('cuda')
