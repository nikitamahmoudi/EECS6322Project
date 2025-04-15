# EECS 6322 Course Project

By Niki Mahmoudi and Kevin Y. H. Hui

Attempt to reproduce ["Adversarial Neuron Pruning Purifies Backdoored Deep Models"](https://openreview.net/pdf?id=4cEapqXfP30) by Dongxian Wu and Yisen Wang \
Paper originally presented in [NeurIPS 2021](https://neurips.cc/virtual/2021/poster/27055)

The paper further utilizes ["BadNets: Evaluating Backdooring Attacks on Deep Neural Networks"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8685687)

How to use this repository:
- `instruction.txt` for installing conda environment
- `backdoor_resnet18.ipynb` trains the ResNet18 model backdoored with badnets
  - [Trained model here](https://drive.google.com/file/d/183irbv-bkvaGoFktraJ8NzFfBocP-Vya/view?usp=sharing), download and put in `saved_models`
- `pruning.ipynb`, `pruning_2.ipynb`, `pruning_2a.ipynb`, `pruning_3.ipynb`, `pruning_3a.ipynb` are the test/trial experiments
- `pruning_experiments.py`, `pruning_experiments_2.py`, `pruning_experiments_3.py` runs the experiments prescribed in the paper in one go
  - `experiment_stats_data-*.json` are our results, you might have slightly different results if you run the experiments yourself
- `experiment_data_plotting.ipynb` plot the graphs