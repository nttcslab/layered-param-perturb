''' Motivating example, Clements mesh and its truncated version '''
import os
import hydra
import torch
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from mypkg import exp_util
import mypkg


def get_results(nFeatures, mod, device, lambda_K=1e-1):
    nParam = mod.params.flatten().shape[0]
    batch_size = 100
    inputs = mypkg.normalized_input(batch_size, nFeatures, device)

    nPerturb = 100
    rand = torch.randn((nPerturb, nParam), device=device)
    outcorr1, _, _ = exp_util.get_outcorr(mod, inputs, rand)
    eval1, _ = torch.linalg.eigh(outcorr1)
    isd1 = exp_util.is_divergence_identity_alpha(outcorr1)

    nOutRandom = 10
    corr, eff_idx = mod.calc_module_grads(inputs, nOutRandom, sample_wise=True)
    corr = corr.mean(dim=0)
    eye = torch.eye(corr.shape[0], device=device)
    corr = exp_util.regularize_scale(corr, eye, lambda_K)
    inv_corr = torch.linalg.inv(corr)
    L = torch.linalg.cholesky(corr)
    chol = torch.linalg.solve_triangular(L.T, eye, upper=True)
    eval, _ = torch.linalg.eigh(corr)

    rand[:, eff_idx] = torch.einsum('mn,sn->sm', chol, rand[:, eff_idx])
    outcorr2, _, _ = exp_util.get_outcorr(mod, inputs, rand)
    eval2, _ = torch.linalg.eigh(outcorr2)
    isd2 = exp_util.is_divergence_identity_alpha(outcorr2)
    return nParam, isd1, isd2, eval, eval1, eval2, corr, inv_corr


def make_dataframe(eval1_1, eval1_2, eval2_1, eval2_2):
    to_concat = []
    data = [eval1_1, eval1_2, eval2_1, eval2_2]
    labels = ['Clements(8,8), $\\mathbf{I}_{56}$', 'Clements(8,8), $\\mathbf{\\Sigma}_u$',
              'Clements(8,4), $\\mathbf{I}_{28}$', 'Clements(8,4), $\\mathbf{\\Sigma}_u$']
    for dat, lab in zip(data, labels):
        df = pd.DataFrame(dat.detach().cpu())
        df_melt = pd.melt(df, var_name='Sorted eigenvalue index', value_name='Eigenvalue')
        df_melt['Structure, Covariance'] = lab
        to_concat.append(df_melt)
    return pd.concat(to_concat, axis=0)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cfg.cuda_id}'
    print('CUDA_VISIBLE_DEVICES = ', os.environ['CUDA_VISIBLE_DEVICES'])
    torch.manual_seed(cfg.seed)

    nFeatures = 8  # cfg.net.num_features  # works when <= 16
    nLayers1, nLayers2 = 8, 4    # cfg.net.num_layers
    mod1 = mypkg.ClementsMZI(nFeatures, nLayers1).to(cfg.device)
    mod2 = mypkg.ClementsMZI(nFeatures, nLayers2).to(cfg.device)

    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['font.size'] = 13
    plt.rcParams['text.usetex'] = False

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5.8, 4.5))
    cmap = plt.get_cmap('seismic')
    normalizer = Normalize(-5.9, 5.9)
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    nParam, isd1, isd2, eval, eval1_1, eval1_2, corr, inv_corr = get_results(nFeatures, mod1, cfg.device)
    axes[0, 0].imshow(corr.detach().cpu(), cmap=cmap, norm=normalizer, interpolation=None)
    axes[0, 0].set_xticks([]), axes[0, 0].set_yticks([])
    axes[0, 0].set_title('Averaged Fisher')
    axes[0, 0].set_ylabel('Clements(8,8)', fontsize=15)
    axes[0, 1].imshow(inv_corr.detach().cpu(), cmap=cmap, norm=normalizer, interpolation=None)
    axes[0, 1].set_xticks([]), axes[0, 1].set_yticks([])
    axes[0, 1].set_title('Covariance $\\mathbf{\\Sigma}_u$')
    nParam, isd1, isd2, eval, eval2_1, eval2_2, corr, inv_corr = get_results(nFeatures, mod2, cfg.device)
    axes[1, 0].imshow(corr.detach().cpu(), cmap=cmap, norm=normalizer, interpolation=None)
    axes[1, 0].set_xticks([]), axes[1, 0].set_yticks([])
    axes[1, 0].set_ylabel('Clements(8,4)', fontsize=15)
    axes[1, 0].set_xticks([])
    axes[1, 1].imshow(inv_corr.detach().cpu(), cmap=cmap, norm=normalizer, interpolation=None)
    axes[1, 1].set_xticks([]), axes[1, 1].set_yticks([])
    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), cmap=cmap)
    plt.savefig('../fig/exp2cov.eps', bbox_inches="tight", dpi=500)
    plt.savefig('../fig/exp2cov.png', bbox_inches="tight", dpi=500)
    plt.show()

    sns.set_theme(style="whitegrid", font_scale=1.4)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap("Paired").colors[6:])
    fig = plt.figure(figsize=(4, 5.5))
    df = make_dataframe(eval1_1, eval1_2, eval2_1, eval2_2)
    ax = sns.pointplot(x='Sorted eigenvalue index', y='Eigenvalue', data=df,
                       hue='Structure, Covariance', errorbar='sd', dodge=True, markers='.')
    ax.set_xlabel('Sorted eigenvalue index', fontsize='19')
    ax.set_ylabel('Eigenvalue', fontsize='19')
    ax.set(yticks=[0, 5, 10, 15, 20])
    ax.legend(loc='lower center', fontsize=18, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    plt.savefig('../fig/exp2eig.eps', bbox_inches="tight")
    plt.savefig('../fig/exp2eig.png', bbox_inches="tight", dpi=500)
    plt.show()


if __name__ == '__main__':
    main()
