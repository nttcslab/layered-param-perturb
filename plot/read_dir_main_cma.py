import pandas as pd
import read_log


def read_dir():
    dir = 'main-cma'
    alldf = pd.DataFrame()
    lastdf = pd.DataFrame()
    cuda_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    ds_list = ['MNIST', 'MNIST', 'MNIST', 'MNIST', 'FMNIST', 'FMNIST', 'FMNIST', 'FMNIST']
    numf_list = [16, 32, 32, 32, 16, 32, 32, 32]
    nums_list = [17, 33, 33, 33, 17, 33, 33, 33]
    seeds_list = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3], [4, 5, 6], [7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3], [4, 5, 6], [7, 8]]
    epochs = 100

    for ci, ds, nf, nums, seeds in zip(cuda_ids, ds_list, numf_list, nums_list, seeds_list):
        for s in seeds:
            logfile = f'../log/{dir}/{ds}{nf}_{nums}_{s}_{ci}.log'
            cfg, df = read_log.read_log(logfile, epochs)
            df = df.assign(seed=s)
            df = df.assign(method='CMA')
            df = df.assign(vec='CMA')
            df = df.assign(dataset=cfg.dataset)
            df = df.assign(K=cfg.net.num_features)
            df = df.assign(L=cfg.net.num_layers)
            df = df.assign(Q=cfg.net.num_solutions)
            alldf = pd.concat([alldf, df])
            lastdf = pd.concat([lastdf, df.iloc[[-1]]])
    return alldf, lastdf


datasets = ['MNIST', 'FMNIST']


def main():
    _, bldf = read_dir()

    for ds in datasets:
        for nf in [16, 32, 64]:
            mean = bldf[(bldf.K==nf) & (bldf.dataset==ds)]['te_acc'].mean()
            print(f'{mean:.4f} & ', end='')
    print()


if __name__ == '__main__':
    main()
