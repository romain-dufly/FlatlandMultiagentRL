# Flatland-MARL
Multi-agent reinforcement learning solution to [Flatland3 challenge](https://www.aicrowd.com/challenges/flatland-3).

# Install Dependencies

### Clone the repository.
```shell
$ git clone http://gitlab.parametrix.cn/parametrix/challenge/flatland-marl.git
$ cd flatland-marl
```

Needs Python version between 3.7 and 3.? (tested and sure to work on 3.9)
If you are using conda, you can create a new environment with the following command (optional) :
```shell
$ conda create -n flatland-marl python=3.9
$ conda activate flatland-marl 
```

### Install flatland
```shell
$ cd flatland-rl
$ pip install .
$ cd ..
```

### Install other requirements
```shell
$ pip install -r requirements.txt
```

### Install torch with CUDA capabilities (for training on A5000 GPUs)
```shell
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Build flatland_cutils
**NOTE** : this is the potentially difficult part of the installation (still works in the computer room, though). May be skipped if you don't want to use the feature parser from `flatland_cutils`. The normal parser from `flatland` can be used instead, using TreeObsForRailEnv and the function normalize_observation (to get features from the observation).

IF YOU DID NOT INSTALL CUTILS: comment out the imports and ignore the related cells in the notebook DQN_policies_notebook.ipynb.

```shell
$ cd flatland_cutils
$ pip install .
$ cd ..
```

### For WSL2 users only (Not usually necessary)
The game rendering relies on `OpenGL`. If you are wsl2 user, it is very likely that you don't have OpenGL installed. Please install it.
```shell
$ sudo apt-get update
$ sudo apt-get install freeglut3-dev
```