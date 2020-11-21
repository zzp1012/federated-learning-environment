# Federated Averaging

## Requirements

1. Download the repo on Github [FedML](https://github.com/FedML-AI/FedML)
2. Change the root dir in `main_fedavg.py` to the absolute path of `FedML`. For example,
    ```python
        sys.path.insert(0, os.path.abspath("/home/zzp1012/FedML")) # add the root dir of FedML
    ```
3. Follow the instruction or documentation of [FedML](https://github.com/FedML-AI/FedML) to install required package in python environment.

## Experimental Tracking Platform 

1. report real-time result to wandb.com, please change ID to your own
    ```
    wandb login `Your ID`
    ```

## Experiment Scripts

Heterogeneous distribution (Non-IID) experiment:

Before any experiments, remember to kill any process occupying port 8999. Simply, you can run the following script:

```bash
lsof -i:8999 # this instruction could show the PID of the process occupying port 8999
kill PID
```

Frond-end debugging:
``` 
## MNIST
sh run_fedavg_standalone_pytorch.sh 0 10 mnist ./../../../data/mnist lr hetero 50 20 0.03 sgd 0 sch_random -v

## shakespeare (LEAF)
sh run_fedavg_standalone_pytorch.sh 0 10 shakespeare ./../../../data/shakespeare rnn hetero 50 1 0.8 sgd 0 sch_random -v

# fed_shakespeare (Google)
sh run_fedavg_standalone_pytorch.sh 0 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 50 1 0.8 sgd 0 sch_random -v

## Federated EMNIST
sh run_fedavg_standalone_pytorch.sh 0 10 femnist ./../../../data/FederatedEMNIST cnn hetero 50 1 0.03 sgd 0 sch_random -v

## Fed_CIFAR100
sh run_fedavg_standalone_pytorch.sh 0 10 fed_cifar100 ./../../../data/fed_cifar100 resnet18_gn hetero 50 1 0.03 adam 0 sch_random -v

# Stackoverflow_LR
sh run_fedavg_standalone_pytorch.sh 0 10 stackoverflow_lr ./../../../data/stackoverflow lr hetero 50 1 0.03 sgd 0 sch_random -v

# Stackoverflow_NWP
sh run_fedavg_standalone_pytorch.sh 0 10 stackoverflow_nwp ./../../../data/stackoverflow rnn hetero 50 1 0.03 sgd 0 sch_random -v

# CIFAR10
sh run_fedavg_standalone_pytorch.sh 0 10 cifar10 ./../../../data/cifar10 resnet56 hetero 50 1 0.03 sgd 0 sch_random -v
```

Please make sure to run on the background when you start training after debugging. An example to run on the background:
``` 
# MNIST
nohup sh run_fedavg_standalone_pytorch.sh 2 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 sgd 0 sch_random -v > ./fedavg_standalone.txt 2>&1 &
```

For large DNNs (ResNet, Transformer, etc), please use the distributed computing (fedml_api/distributed). 

### Benchmark Results
We publish benchmark experimental results at wanb.com: \
https://app.wandb.ai/automl/fedml
