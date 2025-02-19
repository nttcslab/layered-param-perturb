''' Image classification task (MNIST, FashionMNIST) '''
import os
import hydra
import numpy as np
import torch
import mypkg


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cfg.cuda_id}'
    print('CUDA_VISIBLE_DEVICES = ', os.environ['CUDA_VISIBLE_DEVICES'])
    torch.set_num_threads(cfg.num_threads)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cfg.epochs_bp = 100
    cfg.epochs_zo = 1
    cfg.net.lr_zo = 0

    # ONN
    net_model, net_chip, preprocess = mypkg.get_model(cfg)
    mypkg.copy_deficient(net_chip, net_model)

    # dataset
    train_loader, test_loader = mypkg.prepare_data_loader(cfg)

    # report setting
    outfile = f'/{cfg.suffix}{cfg.dataset}{cfg.net.num_features}_{cfg.net.num_layers}'\
              f'_{cfg.net.num_zoo_vectors}_{cfg.zoo.vectors}{cfg.cuda_id}_{cfg.seed}.log'
    progr = mypkg.ReportProgress(cfg, outfile)

    # rough training by backpropagation
    tbp = mypkg.TrainBackprop(cfg, net_model, preprocess, progr, train_loader, test_loader)
    try:
        for epoch in range(cfg.epochs_bp):
            tbp.one_epoch(epoch, cfg)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    # copy the parameters from net_model to net_chip
    mypkg.copy_parameters(net_model, net_chip)

    # training by ZO optimization
    progr.report_msg('# ZO optim start')
    tnz = mypkg.TrainNetZoo(cfg, net_model, net_chip, preprocess, progr, train_loader, test_loader)
    try:
        for epoch in range(cfg.epochs_zo):
            tnz.one_epoch(epoch, cfg)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')


if __name__ == '__main__':
    main()
