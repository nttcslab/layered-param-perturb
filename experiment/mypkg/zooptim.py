''' Zeroth-order Optimization '''
import math
import numpy as np
import torch
from . import arch, chip_bp


class ZerothOrderOptimization():
    def __init__(self, cfg, net_model, net_chip, preprocess):
        self.cfg = cfg.zoo
        self.net = net_model.train()
        self.net_chip = net_chip
        self.preprocess = preprocess
        self.device = list(self.net.parameters())[0].device
        self.optimizer = getattr(torch.optim, cfg.optimizer)(net_model.parameters(), lr=cfg.net.lr_zo)
        self.cost_pmodify = self.cost_forward = 0.0
        self._parse_net()

    def _parse_net(self):
        'Assuming a flat (non-recursive) structure listed in net.module_list'
        self.target_modules = []
        self.num_params = []
        self.eff_idxes = []
        self.cov_matrices = []
        self.chol_matrices = []
        self.eye_matrices = []
        self.nDim = 0
        self.nMod = 0
        for mod in self.net.module_list:
            num_param = sum([math.prod(p.shape) for p in arch.target_parameters(mod)])
            if num_param == 0:
                continue
            elif isinstance(mod, chip_bp.ModulePSBS) or isinstance(mod, chip_bp.ClementsMZI_PS):
                batch_size, nPerturb = 1, 1
                input = normalized_input(batch_size, mod.nFeatures, self.device)
                grads, eff_idx = mod.calc_module_grads(input, nPerturb)
                num_param_eff = grads.shape[0]
                cov_mat = torch.eye(num_param_eff, device=self.device)
                chol_mat = torch.eye(num_param_eff, device=self.device)
                eye_mat = torch.eye(num_param_eff, device=self.device)
            else:
                eff_idx, cov_mat, chol_mat, eye_mat, = None, None, None, None
            self.target_modules.append(mod)
            self.num_params.append(num_param)
            self.eff_idxes.append(eff_idx)
            self.cov_matrices.append(cov_mat)
            self.chol_matrices.append(chol_mat)
            self.eye_matrices.append(eye_mat)
            self.nDim += num_param
            self.nMod += 1

    def initialize_current_batch(self, input, label):
        self.input = self.preprocess(input)
        self.label = label
        self.deltaWeights = None

    def _add_deltaWeights(self, delta, normalize=True):
        if normalize is True:
            norm = delta.norm(dim=1)
            delta /= norm[:, None]
            delta *= math.sqrt(self.nDim)
        if self.deltaWeights is None:
            self.deltaWeights = delta
        else:
            self.deltaWeights = torch.cat((self.deltaWeights, delta))

    def delta_random(self, num_random):
        random = torch.randn((num_random, self.nDim), device=self.device)
        self._add_deltaWeights(random)

    def delta_coordinate(self, num_random):
        random = torch.zeros((num_random, self.nDim), device=self.device)
        idxs = torch.randint(self.nDim, (num_random,))
        for i, idx in enumerate(idxs):
            random[i][idx] = 1
        self._add_deltaWeights(random)

    def delta_lpp(self, num_random):  # line 9, Algorithm 1
        rand_list = []
        for i, (num_param, eff_idx, chol) in enumerate(zip(self.num_params, self.eff_idxes, self.chol_matrices)):
            rand = torch.randn((num_random, num_param), device=self.device)
            if chol is not None:
                rand[:, eff_idx] = torch.einsum('mn,sn->sm', chol, rand[:, eff_idx])
                rand /= (rand**2).mean(dim=1).sqrt()[:, None]  # scale back
            rand_list.append(rand)
        random = torch.cat(rand_list, dim=1)
        self._add_deltaWeights(random)

    def update_cov(self):  # line 6, Algorithm 1
        for mod, cov in zip(self.target_modules, self.cov_matrices):
            if cov is not None:
                inputs = normalized_input(self.cfg.lpp.Rin, mod.nFeatures, self.device)
                corr, _ = mod.calc_module_grads(inputs, self.cfg.lpp.Rout)
                cov.data = (1 - self.cfg.lpp.alpha) * cov + self.cfg.lpp.alpha * corr

    def update_chol(self):  # line 7, Algorithm 1
        for cov, chol, eye in zip(self.cov_matrices, self.chol_matrices, self.eye_matrices):
            if cov is not None:
                scale = torch.tensor(1 + self.cfg.lpp.rho).sqrt()
                cov2 = cov + self.cfg.lpp.rho * eye
                L = torch.linalg.cholesky(cov2)
                chol.data = scale * torch.linalg.solve_triangular(L.T, eye, upper=True)

    def evaluate_all(self, mu):  # line 15, Algorithm 1
        mu /= math.sqrt(self.nDim)
        parameters_vector = torch.cat([p.flatten() for p in arch.target_parameters(self.net)])
        parameters_tensor = parameters_vector[None] + mu * self.deltaWeights
        parameters_tensor = torch.cat((parameters_vector[None], parameters_tensor), dim=0)
        nVariation = parameters_tensor.shape[0]
        batch_size = self.input.shape[0]
        self.cost_pmodify += nVariation
        self.cost_forward += nVariation * batch_size
        outputs_all = self._chip_forward(parameters_tensor, self.input)
        self.output0 = outputs_all[0].to(self.device)
        outputs = outputs_all[1:].to(self.device)
        loss0 = self.net.criterion(self.output0, self.label).detach()
        correct = self.net.accuracy(self.output0, self.label).detach()
        self.deltaLosses, self.deltaOutputs = [], []
        for i in range(self.deltaWeights.shape[0]):
            output = outputs[i].detach()
            loss = self.net.criterion(output, self.label)
            deltaLoss = (loss - loss0) / mu
            self.deltaLosses.append(deltaLoss.view(1))
        return loss0, correct

    def _chip_forward(self, param, inputs):
        nVariation, nParams = param.shape
        idx_start = 0
        for p in arch.target_parameters(self.net_chip):
            psh = p[0].shape
            length = math.prod(psh)
            extracted = param[:, idx_start:idx_start+length]
            extracted = extracted.reshape((nVariation,) + psh)
            p.data = extracted.data
            idx_start += length
        expanded_inputs = torch.cat((inputs[None], ) * nVariation)
        outputs = self.net_chip(expanded_inputs)
        return outputs

    def get_grad(self):  # line 16, Algorithm 1
        deltaLosses = torch.cat(self.deltaLosses)
        grad = torch.matmul(deltaLosses, self.deltaWeights)
        grad *= self.cfg.lambdaS / self.nDim / deltaLosses.shape[0]
        return grad

    def step(self, grad):  # line 17, Algorithm 1
        self.optimizer.zero_grad()
        idx_start = 0
        for p in arch.target_parameters(self.net):
            p_shape = p.shape
            length = np.prod(p_shape)
            p.grad = grad[idx_start:idx_start+length].reshape(p_shape)
            idx_start += length
        self.optimizer.step()
        return

    def perform_test(self, test_loader):
        correct_test_list = []
        for i, (input, label) in enumerate(test_loader):
            input = self.preprocess(input)
            parameters_vector = torch.cat([p.flatten() for p in arch.target_parameters(self.net)])
            parameters_tensor = parameters_vector[None]
            outputs_all = self._chip_forward(parameters_tensor, input)
            output_test = outputs_all[0]  # only the first variation
            correct_test = self.net.accuracy(output_test, label)
            correct_test_list.append(correct_test.item())
        return np.mean(correct_test_list)


def normalized_input(nSamples, nFeatures, device):
    input = torch.randn((nSamples, nFeatures), dtype=torch.cfloat, device=device)
    input /= torch.linalg.vector_norm(input, dim=1)[:, None]
    input *= np.sqrt(nFeatures)
    return input
