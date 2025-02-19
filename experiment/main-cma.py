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

    # ONN
    net_model, net_chip, preprocess = mypkg.get_model(cfg)

    # dataset
    train_loader, test_loader = mypkg.prepare_data_loader(cfg)

    # report setting
    outfile = f'/{cfg.suffix}{cfg.dataset}{cfg.net.num_features}_{cfg.net.num_solutions}_{cfg.seed}_{cfg.cuda_id}.log'
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

    # training by CMA-ES
    progr.report_msg('# CMA-ES start')
    tcma = mypkg.TrainCMA(cfg, net_chip, preprocess, progr, train_loader, test_loader)
    try:
        tcma.run()
    except KeyboardInterrupt:
        print('KeyboardInterrupt')


if __name__ == '__main__':
    main()
