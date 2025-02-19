''' Motivating example, 2-dimensional case '''
import os
import hydra
import torch
import matplotlib.pyplot as plt
from mypkg import exp_util
import mypkg


def plot_delout(delout1, value, target, sample_idx):
    delout1 = delout1.detach().cpu()[:, sample_idx].T + value[sample_idx].detach().cpu()[:, None]
    target = target[sample_idx].detach().cpu()
    plt.figure(figsize=[10, 4.2])
    plt.subplot(121)
    plt.plot(delout1[0].real, delout1[1].real, '.', color='lightblue')
    plt.plot(delout1[0, 0].real, delout1[1, 0].real, 's', color='blue', markersize=7)
    plt.plot(target[0].real, target[1].real, '*', color='red', markersize=10)
    plt.axis('square'), plt.axis([-3, 1, -2, 2])
    plt.subplot(122)
    plt.plot(delout1[0].imag, delout1[1].imag, '.', color='lightblue')
    plt.plot(delout1[0, 0].imag, delout1[1, 0].imag, 's', color='blue', markersize=7)
    plt.plot(target[0].imag, target[1].imag, '*', color='red', markersize=10)
    plt.axis('square'), plt.axis([-2, 2, -1, 3])


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cfg.cuda_id}'
    print('CUDA_VISIBLE_DEVICES = ', os.environ['CUDA_VISIBLE_DEVICES'])
    seed = 1
    torch.manual_seed(seed)
    plt.rcParams["font.size"] = 16

    nFeatures, nLayers, nParam = 2, 1, 4
    batch_size = 1000
    sample_idx = 0
    mod_orig = mypkg.ClementsMZI_PS(nFeatures, nLayers).to(cfg.device)
    mod_task = mypkg.ClementsMZI_PS(nFeatures, nLayers).to(cfg.device)
    mod_orig.clements.params.data[0] = - torch.pi / 2
    mod_orig.clements.params.data[1] = torch.pi / 4
    mod_orig.ps.params.data[0] = torch.pi / 8
    mod_orig.ps.params.data[1] = 0
    mod_task.clements.params.data[0] = torch.pi / 4
    mod_task.clements.params.data[1] = - torch.pi / 4
    mod_task.ps.params.data[0] = 0
    mod_task.ps.params.data[1] = 0
    inputs = mypkg.normalized_input(batch_size, nFeatures, cfg.device)
    # replace the 1st input
    input_angle0 = - torch.pi / 4
    input_angle1 = torch.pi / 2
    input_abs0, input_abs1 = torch.sqrt(torch.tensor(1)), torch.sqrt(torch.tensor(1))
    inputs[sample_idx, 0] = input_abs0 * torch.exp(torch.tensor(input_angle0 * 1.j))
    inputs[sample_idx, 1] = input_abs1 * torch.exp(torch.tensor(input_angle1 * 1.j))
    outputs = mod_orig(inputs)

    rand_scale = 0.5
    nPerturb, nOutRandom = 1000, 10
    rand = rand_scale * torch.randn((nPerturb, nParam), device=cfg.device)
    rand[0] = 0  # make the first one no-perturb
    outcorr1, delout1, value = exp_util.get_outcorr(mod_task, inputs, rand)
    eval1, _ = torch.linalg.eigh(outcorr1)

    corr, eff_idx = mod_task.calc_module_grads(inputs, nOutRandom, sample_wise=True)
    corr = corr.mean(dim=0)
    for row in corr:
        for item in row:
            print(f'{item:.3f} & ', end='')
        print('\\\\')
    eye = torch.eye(corr.shape[0], device=cfg.device)
    lambda_K = 1e-6
    corr = exp_util.regularize_scale(corr, eye, lambda_K)
    inv_corr = torch.linalg.inv(corr)
    for row in inv_corr:
        for item in row:
            print(f'{item:.3f} & ', end='')
        print('\\\\')
    L = torch.linalg.cholesky(corr)
    chol = torch.linalg.solve_triangular(L.T, eye, upper=True)

    rand[:, eff_idx] = torch.einsum('mn,sn->sm', chol, rand[:, eff_idx])
    outcorr2, delout2, value = exp_util.get_outcorr(mod_task, inputs, rand)
    eval2, _ = torch.linalg.eigh(outcorr2)

    print(f'eval1=[{eval1[sample_idx,0]:.3f}, {eval1[sample_idx,1]:.3f}]')
    print(f'eval2=[{eval2[sample_idx,0]:.3f}, {eval2[sample_idx,1]:.3f}]')
    plot_delout(delout1, value, outputs, sample_idx)
    plt.savefig('../fig/exp1b.png')
    plt.show()
    plot_delout(delout2, value, outputs, sample_idx)
    plt.savefig('../fig/exp1a.png')
    plt.show()


if __name__ == '__main__':
    main()
