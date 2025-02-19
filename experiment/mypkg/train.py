import time
import datetime
import math
import pickle
import numpy as np
import pandas as pd
import torch
from torchvision import datasets
import cma
from . import zooptim, arch


def prepare_data_loader(cfg):
    def preprocess(dataset, name):
        maxval_pixel = 255.0
        image_list = []
        for image in dataset.data:
            if name == 'MNIST':
                image = image / maxval_pixel
            image_list.append(image[None])
        dataset.data = torch.cat(image_list)
        return dataset

    match cfg.dataset:
        case 'MNIST':
            train_data = datasets.MNIST(root='data', train=True, download=True)
            test_data = datasets.MNIST(root='data', train=False, download=True)
            train_data = preprocess(train_data, name='MNIST')
            test_data = preprocess(test_data, name='MNIST')
        case 'FMNIST':
            train_data = datasets.FashionMNIST(root='data', train=True, download=True)
            test_data = datasets.FashionMNIST(root='data', train=False, download=True)
            train_data = preprocess(train_data, name='MNIST')
            test_data = preprocess(test_data, name='MNIST')
    batch_size = cfg.batch_size
    match cfg.dataset:
        case 'MNIST' | 'FMNIST':
            batch_size_test = 10000
            train_loader = TensorsLoader(
                [train_data.data.to(cfg.device), train_data.targets.to(cfg.device)],
                batch_size=batch_size, shuffle=True
            )
            test_loader = TensorsLoader(
                [test_data.data.to(cfg.device), test_data.targets.to(cfg.device)],
                batch_size=batch_size_test, shuffle=False
            )
        case 'random':
            nSamples_train, nSamples_test = 10000, 100
            train_data = torch.randn((nSamples_train, cfg.net.num_features), dtype=torch.cfloat, device=cfg.device)
            train_targets = torch.empty((nSamples_train, 1), device=cfg.device)
            test_data = torch.randn((nSamples_test, cfg.net.num_features), dtype=torch.cfloat, device=cfg.device)
            test_targets = torch.empty((nSamples_test, 1), device=cfg.device)
            train_loader = TensorsLoader([train_data, train_targets], batch_size=batch_size, shuffle=True)
            test_loader = TensorsLoader([test_data, test_targets], batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class TrainBackprop():
    def __init__(self, cfg, net_model, preprocess, progr, train_loader, test_loader):
        self.net = net_model.to(cfg.device)
        self.preprocess = preprocess
        self.optimizer = getattr(torch.optim, cfg.optimizer)(net_model.parameters(), lr=cfg.net.lr_bp)
        msg_append = '\tloss\ttr_acc\tte_acc'
        fmt_append = ('{:.5f}', '{:.5f}', '{:.5f}')
        self.progr = progr.set_header(msg_append, fmt_append)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def one_epoch(self, epoch, cfg):
        self.progr.start_epoch(epoch, ['loss', 'tr_acc'])
        for i, (input, label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            input = self.preprocess(input)
            output = self.net(input)
            tr_acc = self.net.accuracy(output, label)
            loss = self.net.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            self.progr.append_results([loss, tr_acc])
        te_acc = self.perform_test()
        self.progr.progress_report(te_acc)

    def perform_test(self):
        correct_test_list = []
        for i, (input, label) in enumerate(self.test_loader):
            input = self.preprocess(input)
            output_test = self.net(input)
            correct_test = self.net.accuracy(output_test, label)
            correct_test_list.append(correct_test.item())
        return np.mean(correct_test_list)


class TrainNetZoo():
    def __init__(self, cfg, net_model, net_chip, preprocess, progr, train_loader, test_loader):
        self.num_random = cfg.net.num_zoo_vectors
        net_model = net_model.to(cfg.device)
        net_chip = net_chip.to(cfg.device)
        self.zoo = zooptim.ZerothOrderOptimization(cfg, net_model, net_chip, preprocess)
        msg_append = '\tloss\ttr_acc\tgrad_n\tdl_pow\tte_acc\ttr_loss\tn_delta\tpmodify\t\tforward'
        fmt_append = ('{:.5f}', '{:.5f}', '{:.1e}', '{:.1e}', '{:.5f}', '{:.5f}', '{:d}', '{:.3e}', '{:.3e}')
        self.progr = progr.set_header(msg_append, fmt_append)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.prev_loss = 1e+16

    def one_epoch(self, epoch, cfg):
        self.progr.start_epoch(epoch, ['loss', 'tr_acc', 'agrad_norm', 'dloss_pow'])
        agrad = None
        for i, (input, label) in enumerate(self.train_loader):
            self.zoo.initialize_current_batch(input, label)
            match cfg.zoo.vectors:
                case 'fromI':
                    self.zoo.delta_random(self.num_random)
                case 'coordinate':
                    self.zoo.delta_coordinate(self.num_random)
                case 'lpp':
                    if i % cfg.zoo.lpp.Tud == 0:              # line 5, Algorithm 1
                        self.zoo.update_cov()                 # line 6, Algorithm 1
                        self.zoo.update_chol()                # line 7, Algorithm 1
                    self.zoo.delta_lpp(self.num_random)       # line 9, Algorithm 1
            loss, tr_acc = self.zoo.evaluate_all(cfg.zoo.mu)  # line 15, Algorithm 1
            agrad = self.zoo.get_grad()                       # line 16, Algorithm 1
            self.zoo.step(agrad)                              # line 17, Algorithm 1
            dloss_pow = (torch.cat(self.zoo.deltaLosses)**2).mean()
            self.progr.append_results([loss, tr_acc, agrad.norm(), dloss_pow])
        te_loss, te_acc = calc_loss_accuracy(self.zoo.net_chip, self.zoo.preprocess, self.test_loader)
        tr_loss, tr_acc = calc_loss_accuracy(self.zoo.net_chip, self.zoo.preprocess, self.train_loader)
        avg_loss = self.progr.progress_report(
            te_acc, tr_loss, self.zoo.deltaWeights.shape[0], self.zoo.cost_pmodify, self.zoo.cost_forward
        )
        return avg_loss


class TrainCMA():
    def __init__(self, cfg, net_chip, preprocess, progr, train_loader, test_loader):
        self.net = net_chip.to(cfg.device)
        self.device = cfg.device
        self.preprocess = preprocess
        msg_append = '\tloss\ttr_acc\tte_acc\ttr_loss\tn_delta\tpmodify\t\tforward'
        fmt_append = ('{:.5f}', '{:.5f}', '{:.5f}', '{:.5f}', '{:d}', '{:.3e}', '{:.3e}')
        self.progr = progr.set_header(msg_append, fmt_append)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_batches_per_epoch = int(train_loader.data_size / cfg.batch_size)
        self.options = {
            'seed': np.random.randint(1e+7),
            'maxiter': cfg.epochs_cma * self.num_batches_per_epoch,
            'popsize': cfg.net.num_solutions,
            'verbose': cfg.cma.verbose
        }
        if cfg.cma.diagonal:
            self.options['CMA_diagonal'] = True
        self.savefile = f'/tmp/es{cfg.net.num_features}_{cfg.net.num_solutions}{cfg.cma.diagonal}.pickle'
        self.counter = 0
        self.cost_pmodify = self.cost_forward = 0.0
        self.solution = self._init_param()
        self.sigma = cfg.net.cma_sigma
        self.es = cma.CMAEvolutionStrategy(self.solution, self.sigma, options=self.options)

    def run(self):
        self.train_loader.__iter__()
        self.es.optimize(lambda x: self._fitness_func(x), callback=self._my_callback)
        return self.avg_loss

    def one_epoch(self, epoch):
        if epoch == 0:
            print(self.options)
            self.train_loader.__iter__()
        else:
            print(f'loading from {self.savefile}')
            self.es = pickle.loads(open(self.savefile, 'rb').read())
        self.es.optimize(
            lambda x: self._fitness_func(x), iterations=self.num_batches_per_epoch, callback=self._my_callback
        )
        open(self.savefile, 'wb').write(self.es.pickle_dumps())
        return self.avg_loss

    def _my_callback(self, es):
        self.counter = 0
        # es: CMAEvolutionStrategy object
        if es.countiter % self.num_batches_per_epoch == 1:
            epoch = int(es.countiter / self.num_batches_per_epoch)
            self.progr.start_epoch(epoch, ['loss', 'tr_acc'])
        loss = torch.tensor(self._forward(es.mean))
        tr_acc = torch.tensor(0)
        self.progr.append_results([loss, tr_acc])
        if es.countiter % self.num_batches_per_epoch == 0:
            param = torch.tensor(es.mean).to(torch.float32).to(self.device)
            self._set_parameters(param)
            te_loss, te_acc = calc_loss_accuracy(self.net, self.preprocess, self.test_loader)
            tr_loss, tr_acc = calc_loss_accuracy(self.net, self.preprocess, self.train_loader)
            self.avg_loss = self.progr.progress_report(
                te_acc, tr_loss, self.options['popsize'], self.cost_pmodify, self.cost_forward
            )

    def _init_param(self):
        param_list = []
        for mod in self.net.modules():
            if len(list(mod.modules())) > 1:
                continue
            for p in arch.target_parameters(mod):
                param_list.append(p.flatten().detach().cpu())
        return torch.cat(param_list)

    def _set_parameters(self, param):
        nVariation = 1
        idx_start = 0
        for p in arch.target_parameters(self.net):
            psh = p[0].shape
            length = math.prod(psh)
            extracted = param[idx_start:idx_start+length]
            extracted = extracted.reshape((nVariation,) + psh)
            p.data = extracted.data
            idx_start += length

    def perform_test(self):
        correct_test_list = []
        for i, (input, label) in enumerate(self.test_loader):
            input = self.preprocess(input)
            output_test = self.net(input[None])
            correct_test = self.net.accuracy(output_test[0], label)
            correct_test_list.append(correct_test.item())
        return np.mean(correct_test_list)

    def _get_inputs_label(self):
        try:
            self.inputs, self.label = self.train_loader.__next__()
        except StopIteration:
            self.train_loader.__iter__()
            self.inputs, self.label = self.train_loader.__next__()

    def _forward(self, solution):
        input = self.preprocess(self.inputs)
        solution = torch.from_numpy(solution).to(torch.float32).to(self.device)
        self._set_parameters(solution)
        expanded_input = torch.cat((input[None], ))
        output = self.net(expanded_input)
        loss = self.net.criterion(output[0], self.label).item()
        return loss

    def _fitness_func(self, solution):
        if self.counter == 0:
            self._get_inputs_label()
        self.counter += 1
        self.cost_pmodify += 1
        self.cost_forward += self.inputs.shape[0]
        return self._forward(solution)


class ReportProgress():
    def __init__(self, cfg, outfile):
        self.start_time = time.time()
        self.outfile = cfg.logdir + outfile
        dt = datetime.datetime.fromtimestamp(self.start_time)
        self._print(dt)
        self._print(cfg)

    def set_header(self, msg_append, fmt_append):
        msg = 'epoch\ttime'
        msg += msg_append
        self.fmt = ('{:d}', '{:.2f}')
        self.fmt += fmt_append
        self._print(msg)
        return self

    def start_epoch(self, epoch, columns):
        self.epoch = epoch
        self.results = pd.DataFrame(columns=columns)

    def append_results(self, results):
        appending = {k: v.item() for k, v in zip(self.results.columns, results)}
        app_se = pd.Series(appending)
        self.results.loc[len(self.results)] = app_se

    def progress_report(self, *direct_values):
        elapsed_time = time.time() - self.start_time
        mean_results = self.results.mean()
        to_report = [self.epoch+1, elapsed_time]
        to_report.extend([v for v in mean_results])
        to_report.extend(direct_values)
        msg = ''
        for v, f in zip(to_report, self.fmt):
            msg += f.format(v) + '\t'
        self._print(msg[:-1])
        loss_mean = self.results['loss'].mean()
        return loss_mean

    def report_msg(self, msg):
        self._print(msg)

    def _print(self, msg):
        print(msg)
        if self.outfile is not None:
            with open(self.outfile, 'a') as f:
                print(msg, file=f)


class TensorsLoader:
    ''' tensors: a list of tensors '''
    def __init__(self, tensors, batch_size=1, shuffle=False):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = tensors[0].shape[0]

    def __iter__(self):
        self._i = 0
        if self.shuffle:
            index_shuffle = torch.randperm(self.data_size)
            self.tensors = [tensor[index_shuffle] for tensor in self.tensors]
        return self

    def __next__(self):
        i1 = self.batch_size * self._i
        i2 = min(self.batch_size * (self._i + 1), self.data_size)
        if i1 >= self.data_size:
            raise StopIteration()
        self._i += 1
        return [tensor[i1:i2] for tensor in self.tensors]


def calc_loss_accuracy(net_chip, preprocess, data_loader):
    loss_list, correct_list = [], []
    for i, (input, label) in enumerate(data_loader):
        input = preprocess(input)
        output = net_chip(input[None])
        loss = net_chip.criterion(output[0], label)
        loss_list.append(loss.item())
        correct = net_chip.accuracy(output[0], label)
        correct_list.append(correct.item())
        return np.mean(loss_list), np.mean(correct_list)
