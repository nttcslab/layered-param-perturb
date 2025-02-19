#! /bin/bash

python main-cma.py -m seed=1,2,3,4,5,6,7,8 net=ff16cma cuda_id=0 suffix='main-cma/' dataset=MNIST &
python main-cma.py -m seed=1,2,3           net=ff32cma cuda_id=1 suffix='main-cma/' dataset=MNIST &
python main-cma.py -m seed=4,5,6           net=ff32cma cuda_id=2 suffix='main-cma/' dataset=MNIST &
python main-cma.py -m seed=7,8             net=ff32cma cuda_id=3 suffix='main-cma/' dataset=MNIST &
python main-cma.py -m seed=1,2,3,4,5,6,7,8 net=ff16cma cuda_id=4 suffix='main-cma/' dataset=FMNIST &
python main-cma.py -m seed=1,2,3           net=ff32cma cuda_id=5 suffix='main-cma/' dataset=FMNIST &
python main-cma.py -m seed=4,5,6           net=ff32cma cuda_id=6 suffix='main-cma/' dataset=FMNIST &
python main-cma.py -m seed=7,8             net=ff32cma cuda_id=7 suffix='main-cma/' dataset=FMNIST &
