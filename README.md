# Flatland-MARL

This repository expands on various multi-agent reinforcement learning (MARL) algorithms and applies them to the Flatland environment. The Flatland environment is a grid-world environment where agents must navigate to their respective target stations while avoiding collisions with other agents. The environment is based on the [Flatland challenge](https://www.aicrowd.com/challenges/flatland-challenge) hosted on AIcrowd.

The repository is structured as follows:

- `flatland_rl/`: The Flatland environment, with a starter kit for training agents, taken from the AIcrowd Flatland challenge repository.

- `flatland_cutils/`: C++ for a BFS tree observation builder, taken from ```Multi-Agent Path Finding via Tree LSTM, 2022```

- `src/`: Contains utility code as well as implementations of various policies and training algorithms:
    - `models.py`: The neural network architectures used in the policies.
    - `deep_model_policy.py`: Wrapper for neural networks into DDQN policies.
    - `training.py`: Training loop and evaluation for the DDQN policies.
    - `impl_config.py`: Configuration for some networks.

    - `evolution_algos.py`, `evolution_policy.py`: Implementation of the evolution algorithm for training policies.

    - `rewards.py`: Custom rewards for the Flatland environment.

    - `observation_utils.py`: Utility functions for processing observations, in part taken from the Flatland starter kit.
    - `test_utils.py`: Utility functions for running tests and rendering policies.

Then, various notebooks are provided to demonstrate the training and evaluation of the policies. The notebooks are as follows:

- `DQN_policies_notebook.ipynb`: Demonstrates training a policy using the DDQN algorithm, with neural networks of varying complexity.
- `gnn_notebook.ipynb`: Demonstrates training a policy using a Graph Neural Network (GNN) architecture.
- `rewards_notebook.ipynb`: Demonstrates training a policy using various custom rewards.
- `ES_logreg.ipynb`: Demonstrates CEM and SAES with a logistic regression policy.