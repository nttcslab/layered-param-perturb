#! /bin/bash

# fromI, perturbation sampled from a normal distribution with an identity covariance matrix I

python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=8,12,16  dataset=MNIST  suffix='main/' net=ff16f cuda_id=0 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=16,24,32 dataset=MNIST  suffix='main/' net=ff32f cuda_id=1 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=64       dataset=MNIST  suffix='main/' net=ff64f cuda_id=2 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=32,48    dataset=MNIST  suffix='main/' net=ff64f cuda_id=3 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=8,12,16  dataset=FMNIST suffix='main/' net=ff16f cuda_id=4 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=16,24,32 dataset=FMNIST suffix='main/' net=ff32f cuda_id=5 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=64       dataset=FMNIST suffix='main/' net=ff64f cuda_id=6 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=32,48    dataset=FMNIST suffix='main/' net=ff64f cuda_id=7 zoo.vectors=fromI &
