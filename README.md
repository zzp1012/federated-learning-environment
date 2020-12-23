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

1. Before any experiments, remember to kill any process occupying port 8999. Simply, you can run the following script:
    ```bash
    lsof -i:8999 # this instruction could show the PID of the process occupying port 8999
    kill PID
    ```

2. To test whether program are correctly configured, you can run following commands to see whether training process starts correctly.
    ```
    sh begin.sh
    ```

Or, you can try other heterogeneous distribution (Non-IID) experiment:
``` 
## MNIST (non-i.i.d)
sh ./src/run_fedavg_standalone_pytorch.sh 0 10 mnist /home/zzp1012/FedML/data/mnist lr hetero 50 20 0.03 sgd 0 sch_random -v

## shakespeare (non-i.i.d LEAF)
sh ./src/run_fedavg_standalone_pytorch.sh 0 10 shakespeare /home/zzp1012/FedML/data/shakespeare rnn hetero 50 1 0.8 sgd 0 sch_random -v

# fed_shakespeare (non-i.i.d Google)
sh ./src/run_fedavg_standalone_pytorch.sh 0 10 fed_shakespeare /home/zzp1012/FedML/data/fed_shakespeare rnn hetero 50 1 0.8 sgd 0 sch_random -v

## Federated EMNIST (non-i.i.d)
sh ./src/run_fedavg_standalone_pytorch.sh 0 10 femnist /home/zzp1012/FedML/data/FederatedEMNIST cnn hetero 50 1 0.03 sgd 0 sch_random -v

## Fed_CIFAR100 (non-i.i.d)
sh ./src/run_fedavg_standalone_pytorch.sh 0 10 fed_cifar100 /home/zzp1012/FedML/data/fed_cifar100 resnet18_gn hetero 50 1 0.03 adam 0 sch_random -v

# Stackoverflow_LR (non-i.i.d)
sh ./src/run_fedavg_standalone_pytorch.sh 0 10 stackoverflow_lr /home/zzp1012/FedML/data/stackoverflow lr hetero 50 1 0.03 sgd 0 sch_random -v

# Stackoverflow_NWP (non-i.i.d)
sh ./src/run_fedavg_standalone_pytorch.sh 0 10 stackoverflow_nwp /home/zzp1012/FedML/data/stackoverflow rnn hetero 50 1 0.03 sgd 0 sch_random -v
 
# CIFAR10 (non-i.i.d) 
sh ./src/run_fedavg_standalone_pytorch.sh 0 10 cifar10 /home/zzp1012/FedML/data/cifar10 resnet56 hetero 50 1 0.03 sgd 0 sch_random -v
```

All above datasets are heterogeneous non-i.i.d distributed.

### Benchmark Results
We publish benchmark experimental results at wanb.com: \
https://app.wandb.ai/automl/fedml
