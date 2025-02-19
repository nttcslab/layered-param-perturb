#! /bin/bash

# bp_now_def: backpropagation with knowing the deficients (theoretical limit)

python main-bp_know_def.py -m seed=1,2,3,4,5,6,7,8 dataset=MNIST  suffix='main-bp_know_def/' net=ff16c cuda_id=1 &
python main-bp_know_def.py -m seed=1,2,3,4,5,6,7,8 dataset=MNIST  suffix='main-bp_know_def/' net=ff32c cuda_id=2 &
python main-bp_know_def.py -m seed=1,2,3,4,5,6,7,8 dataset=MNIST  suffix='main-bp_know_def/' net=ff64c cuda_id=3 &
python main-bp_know_def.py -m seed=1,2,3,4,5,6,7,8 dataset=FMNIST suffix='main-bp_know_def/' net=ff16c cuda_id=4 &
python main-bp_know_def.py -m seed=1,2,3,4,5,6,7,8 dataset=FMNIST suffix='main-bp_know_def/' net=ff32c cuda_id=5 &
python main-bp_know_def.py -m seed=1,2,3,4,5,6,7,8 dataset=FMNIST suffix='main-bp_know_def/' net=ff64c cuda_id=6 &
