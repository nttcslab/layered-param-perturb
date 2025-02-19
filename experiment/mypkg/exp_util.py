import torch
from . import arch


def regularize_scale(corr_matrix, eye_matrix, rho, scale=False):
    corr_matrix += rho * eye_matrix
    if scale:
        num_param = corr_matrix.shape[0]
        corr_matrix /= corr_matrix.trace()
        corr_matrix *= num_param
    return corr_matrix


def _tangents_from_vector(params, vector):
    idx_start = 0
    tangents = []
    for p in params:
        length = len(p.flatten())
        t = vector[idx_start:idx_start+length].reshape(p.shape)
        tangents.append(t)
        idx_start += length
    return tangents


def _perturb_params(mod, input, perturbation):
    params = arch.target_parameters(mod)
    tangents = _tangents_from_vector(params, perturbation)
    value, grad = mod.jvp(input, tangents)
    return value, grad


def get_outcorr(mod, input, random):
    nPerturb = random.shape[0]
    deltaOutputs = []
    for i in range(nPerturb):
        value, deltaOutput = _perturb_params(mod, input, random[i])
        deltaOutputs.append(deltaOutput[None])
    deltaOutputs = torch.cat(deltaOutputs)
    outcorr = torch.einsum('psa,psb->sab', deltaOutputs, deltaOutputs.conj()) / nPerturb
    return outcorr, deltaOutputs, value


def is_divergence_identity_alpha(A):
    nSample, nOut, nOut = A.shape
    trace = torch.einsum('smm->s', A).real
    alpha = trace / nOut
    det = torch.linalg.det(A).real
    logdet = torch.log(det)
    isd = (trace / alpha).sum() - logdet.sum() - nOut * (1 - torch.log(alpha)).sum()
    return isd
