# Layered-Parameter Perturbation for Zeroth-Order Optimization of Optical Neural Networks

All source code required for conducting and analyzing the experiments described in the above titled manuscript presented at AAAI 2025.

## Directories
```
.
├─ compile
├─ experiment
│  ├─ mypkg
│  └─ net
├─ fig
├─ log
│  ├─ main
│  ├─ main-bp_know_def
│  └─ main-cma
└─ plot
```
- compile: custom CUDA kernels and C++ code for computationally efficient simulation of ONNs
- experiment: python code and shell scripts for the experiments
- experiment/mypkg: python code for training and simulating ONNs
- experiment/net: YAML configuration files with optimized learning-rate hyperparameters for specific tasks and circuit sizes
- fig: to store figures in *.png and *.eps formats
- log: to store execution log files in subdirectories, e.g., main, main-cma.
- plot: python code for generating figures and table

## Procedure
One can reproduce the experimental results and plots by the following procedure.

1. Prepare a python environment. The versions of python and pip3 we used was 3.11 and 25.0.
2. Install the packages specified in [requirements.txt](./requirements.txt).
   ```bash
   pip3 install -r requirements.txt
   ``` 
3. Install pytorch by the instructions in https://pytorch.org/. The pytorch version we used was 2.6.0+cu118:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
4. Compile CUDA and C++ codes in [compile](./compile/):
   ```bash
   cd compile; pip3 install --no-build-isolation --editable .
   ```
   The built package `mzi_onn_sim` is imported by the python files [experiment/mypkg/chip.py](./experiment/mypkg/chip.py) and [experiment/mypkg/chip_bp.py](./experiment/mypkg/chip_bp.py).
5. Run one of the experiment:
   ```bash
   cd experiment; python exp1.py
   ```
6. To reproduce all the experimental results, one needs to run all the shell scripts in the directory `experiment`. An example of running a shell script is:
   ```bash
   cd experiment; bash main-lpp.sh
   ```
   The shell scripts contained here, e.g.,  [main-lpp.sh](./experiment/main-lpp.sh), generally assume that the environment has 8 GPUs, `cuda_id` from 0 to 7, and takes advantage of parallel execution.
   When the environment does not have that amount of GPUs, please refer to [none or less GPUs](#none-or-less-gpus).
7. To generate figures and table, please go to the directory `plot`, and run the corresponding python script. An example is:
   ```bash
   cd plot; python conv.py
   ```

## Files in experiment
#### Python code for Motivating examples
- [exp1.py](./experiment/exp1.py): 2-dimensional case, Figure 3 (exp1a.png, exp1b.png)
- [exp2.py](./experiment/exp2.py): Clements mesh and its truncated version, Figure 4 (exp2cov.png, exp2eig.png)

#### Python code related to the proposed method in `experiment/mypkg`, with comments such as `# line xx, Algorithm 1`
- [zooptim.py](./experiment/mypkg/zooptim.py): Zeroth-order (ZO) optimization
- [train.py](./experiment/mypkg/train.py): traning neural networks

#### Python code for Experiments
- [main.py](./experiment/main.py): ZO optimization for Image classification task
- [main-cma.py](./experiment/main-cma.py): CMA-ES for Image classification task
- [main-bp_know_def.py](./experiment/main-bp_know_def.py): Backpropagation with perfect error information for Image classification task

#### Shell script for Experiments
- [main-fromI.sh](./experiment/main-fromI.sh), [main-co.sh](./experiment/main-co.sh), [main-lpp.sh](./experiment/main-lpp.sh): Table 3, Figure 6
- [main-cma.sh](./experiment/main-cma.sh): Table 3
- [main-bp_know_def.sh](./experiment/main-bp_know_def.sh): Table 3

## Files in plot
- [table3.py](./plot/table3.py): Table 3
- [loss.py](./plot/loss.py): Figure 6 (loss.png), run in a separate python environment for `statannotations`
- [conv.py](./plot/conv.py): Figure 7 (conv.png)

## Trouble shooting
#### none or less GPUs
If the environment does not have any GPU, i.e., without CUDA, please run a python program as
```bash
python main.py device=cpu
```
If the environment does not have 8 GPUs, please modify a shell script to accommodate the number of GPUs.
For example, if there are 3 GPUs, [main-lpp.sh](./experiment/main-lpp.sh) should be modified to
```bash
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=8,12,16  dataset=MNIST,FMNIST  suffix='main/' net=ff16l cuda_id=0 zoo.vectors=lpp  &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=16,24,32 dataset=MNIST,FMNIST  suffix='main/' net=ff32l cuda_id=1 zoo.vectors=lpp  &
python main.py -m seed=1,2,3,4,5,6,7,8 net.num_layers=32,48,64 dataset=MNIST,FMNIST  suffix='main/' net=ff64l cuda_id=2 zoo.vectors=lpp  &
```
If a shell script is modified and therefore the mapping from `cuda_id` to the experimental condition changes, one should also modify the corresponding plot file in [Files in plot](#files-in-plot).

## Citation

```
@inproceedings{sawada2025layered,
  title = {Layered-Parameter Perturbation for Zeroth-Order Optimization of Optical Neural Networks},
  author = {Hiroshi Sawada and Kazuo Aoyama and Masaya Notomi},
  booktitle = {Proc. AAAI Conf. Artificial Intelligence},
  year = {2025}
}
```