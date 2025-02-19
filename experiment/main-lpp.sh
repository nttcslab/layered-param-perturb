#! /bin/bash

# lpp, layered-parameter perturbation (the proposed method)

python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=8,12,16  dataset=MNIST  suffix='main/' net=ff16l cuda_id=0 zoo.vectors=lpp  &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=16,24,32 dataset=MNIST  suffix='main/' net=ff32l cuda_id=1 zoo.vectors=lpp  &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=64       dataset=MNIST  suffix='main/' net=ff64l cuda_id=2 zoo.vectors=lpp  &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=32,48    dataset=MNIST  suffix='main/' net=ff64l cuda_id=3 zoo.vectors=lpp  &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=8,12,16  dataset=FMNIST suffix='main/' net=ff16l cuda_id=4 zoo.vectors=lpp  &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=16,24,32 dataset=FMNIST suffix='main/' net=ff32l cuda_id=5 zoo.vectors=lpp  &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=64       dataset=FMNIST suffix='main/' net=ff64l cuda_id=6 zoo.vectors=lpp  &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=32,48    dataset=FMNIST suffix='main/' net=ff64l cuda_id=7 zoo.vectors=lpp  &
