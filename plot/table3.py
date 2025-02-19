import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import read_dir_main
import read_dir_main_cma
import read_dir_main_bp


def data_of_test_accuracies(lastdf, vector_methods, datasets, num_features_list):
    data = [
        [
            [
                lastdf[(lastdf.vec == vec) & (lastdf.K == nf) & (lastdf.L == nf) & (lastdf.Q >= nf) & (lastdf.dataset == ds)]['te_acc']
                for vec in vector_methods
            ]
            for nf in num_features_list
        ]
        for ds in datasets
    ]
    return data


def best_mean_values_and_pvalues(data):
    # identify best mean values
    best_idx_list = [
        [
            np.array(da).mean(axis=1).argmax()
            for da in dat
        ]
        for dat in data
    ]
    # p-values by Mann-Whitney U test
    pvalue_list = [
        [
            [
                # None: best mean value, otherwise p-value
                # None if i == idx else wilcoxon(da[idx], data1)[1]
                None if i == idx else mannwhitneyu(da[idx], data1)[1]
                for i, data1 in enumerate(da)
            ]
            for idx, da in zip(idxs, dat)
        ]
        for idxs, dat in zip(best_idx_list, data)
    ]
    # convert the pvalue_list into a numpy array and permute
    pvalues = np.array(pvalue_list).transpose(2, 0, 1)
    return pvalues


def print_results_ZO_CMA():
    # read results
    _, lastdf = read_dir_main.read_dir()
    _, lastdf_cma = read_dir_main_cma.read_dir()
    lastdf = pd.concat([lastdf, lastdf_cma])

    # create data of 'te_acc' test accuracies
    names = ['ZO-$\\mb{I}$', 'ZO-$\\mathbf{co}$', 'ZO-$\\pcov_u$', 'CMA']
    vector_methods = ['fromI', 'coordinate', 'lpp', 'CMA']
    datasets = ['MNIST', 'FMNIST']
    num_features_list = [16, 32, 64]
    data = data_of_test_accuracies(lastdf, vector_methods, datasets, num_features_list)

    # Compensation for not running CMA, K=64
    data[0][2][3] = [0, 0, 0, 0, 0, 0, 0, 0]
    data[1][2][3] = [0, 0, 0, 0, 0, 0, 0, 0]

    # identify best mean values, p-values by Mann-Whitney U test
    pvalues = best_mean_values_and_pvalues(data)

    # produce the results in a latex table format
    alpha = 0.05  # significance level
    data = np.array(data).transpose(2, 0, 1, 3)
    for nam, dat, pva in zip(names, data, pvalues):
        print(f'\\multirow{{2}}{{*}}{{{nam}}} & ', end='')
        for ds, da, pv in zip(datasets, dat, pva):
            for nf, d, p in zip(num_features_list, da, pv):
                mean = d.mean()
                if p is None:
                    print(f'\\textbf{{{mean*100:.2f}}}\\% & ', end='')
                elif p < alpha:
                    print(f'\\textit{{{mean*100:.2f}}}\\% & ', end='')
                else:
                    print(f'{mean*100:.2f}\\% & ', end='')
        print('\b\b\\\\')
        print('& ', end='')
        for da in dat:
            for d in da:
                std = d.std()
                print(f'$\\pm {std*100:.2f}$ & ', end='')
        print('\b\b\\\\\\hline')


def print_results_BP_w_error_info():
    # read results
    _, lastdf = read_dir_main_bp.read_dir()

    # create data of 'te_acc' test accuracies
    names = ['BP w error info', ]
    vector_methods = ['fromI', ]
    datasets = ['MNIST', 'FMNIST']
    num_features_list = [16, 32, 64]
    data = data_of_test_accuracies(lastdf, vector_methods, datasets, num_features_list)

    # produce the results in a latex table format
    data = np.array(data).transpose(2, 0, 1, 3)
    for nam, dat in zip(names, data):
        print(f'\\multirow{{2}}{{*}}{{{nam}}} & ', end='')
        for ds, da in zip(datasets, dat):
            for nf, d in zip(num_features_list, da):
                mean = d.mean()
                print(f'{mean*100:.2f}\\% & ', end='')
        print('\b\b\\\\')
        print('& ', end='')
        for da in dat:
            for d in da:
                std = d.std()
                print(f'$\\pm {std*100:.2f}$ & ', end='')
        print('\b\b\\\\\\hline')


if __name__ == '__main__':
    print_results_ZO_CMA()
    print_results_BP_w_error_info()
