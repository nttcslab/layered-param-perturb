import pandas as pd
import read_log


def read_dir():
    dir = 'main'
    alldf = pd.DataFrame()
    lastdf = pd.DataFrame()
    vectors = ['fromI', 'coordinate', 'lpp']
    cuda_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    dataset_list = ['MNIST', 'MNIST', 'MNIST', 'MNIST', 'FMNIST', 'FMNIST', 'FMNIST', 'FMNIST']
    numf_list = [16, 32, 64, 64, 16, 32, 64, 64, 16, 32, 64, 64]
    numl_list = [[8, 12, 16], [16, 24, 32], [64,], [32, 48], [8, 12, 16], [16, 24, 32], [64,], [32, 48]]
    numz_list = [[16], [32], [64], [64], [16], [32], [64], [64]]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    epochs = 100

    for vec in vectors:
        for ci, ds, nf, layers, numz in zip(cuda_ids, dataset_list, numf_list, numl_list, numz_list):
            for layer in layers:
                for nz in numz:
                    for s in seeds:
                        logfile = f'../log/{dir}/{ds}{nf}_{layer}_{nz}_{vec}{ci}_{s}.log'
                        cfg, df = read_log.read_log(logfile, epochs)
                        df = df.assign(vec=vec)
                        match vec:
                            case 'fromI':
                                df = df.assign(method='ZO-$\\mathbf{0}$'.format('{I}'))
                            case 'coordinate':
                                df = df.assign(method='ZO-$\\mathbf{co}$')
                            case 'lpp':
                                df = df.assign(method='ZO-$\\mathbf{0}$'.format('{\\Sigma}_u'))
                        df = df.assign(minute=(df.time / 60))
                        df = df.assign(seed=s)
                        df = df.assign(dataset=cfg.dataset)
                        df = df.assign(K=cfg.net.num_features)
                        df = df.assign(L=cfg.net.num_layers)
                        df = df.assign(Q=cfg.net.num_zoo_vectors)
                        alldf = pd.concat([alldf, df])
                        lastdf = pd.concat([lastdf, df.iloc[[-1]]])
    return alldf, lastdf
