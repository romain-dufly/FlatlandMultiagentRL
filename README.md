# Flatland-MARL
Multi-agent reinforcement learning solution to [Flatland3 challenge](https://www.aicrowd.com/challenges/flatland-3).

Also see the paper https://arxiv.org/abs/2210.12933.

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
There's a bug in the flatland environment that may lead to performance drop for RL solutions, so we cloned `flatland-rl-3.0.15` and fixed the bug. The bug-free one is provided in folder `flatland-rl/`. Please install this version of flatland.
```shell
$ cd flatland-rl
$ pip install .
$ cd ..
```

### Install other requirements
```shell
$ pip install -r requirements.txt
```

### Build flatland_cutils
**NOTE** : this is the difficult part of the installation. May be skipped if you don't want to use the feature parser from `flatland_cutils`. The normal parser from `flatland` can be used instead, using TreeObsForRailEnv and the function normalize_observation (to get features from the observation).

`flatland_cutils` is a feature parsing package designed to substitute the built-in `flatland.envs.observations.TreeObsForRailEnv`. Our feature parser is developed in c++ language, which is much faster than the built-in `TreeObsForRailEnv`.
```shell
$ cd flatland_cutils
$ pip install .
$ cd ..
```

### For WSL2 users only (Not necessary for me (romain))
The game rendering relies on `OpenGL`. If you are wsl2 user, it is very likely that you don't have OpenGL installed. Please install it.
```shell
$ sudo apt-get update
$ sudo apt-get install freeglut3-dev
```


# Usage

## Romain

See the file testing.ipynb to try and understand how the environnement and agents work.
The first part implements a shortest path algorithm to determine actions, while the latter part implements a DDDQNetwork that is trained on the environment.

The DeepQNetwork relies on files deep_model_policy.py and training.py to define the policy and then train it.
Models are located in models.py.

Current goal : change the model to a TreeLSTM model, and train it on the environment using current functions.
Compare to the results below.


## Quick demo for TreeLSTM
Only if cutils is installed.
Run the solution in random environments:
```shell
$ cd solution/
$ python demo.py
```

In a terminal without GUI, you may disable real-time rendering and save the replay as a video.
```shell
$ python demo.py --no-render --save-video replay.mp4
```

<!-- 
## Our results

| Test Stage |     Model     | #agents | Map Size  | #cities| Arrival%| Normalized<br>Reward|
|:----------:|:------------- | -------:|:---------:| ------:| -------:| -------------------:|
|  Test_00   | Phase-III-50  |       7 |  30 x 30  |      2 |    94.3 |                .957 |
|  Test_01   | Phase-III-50  |      10 |  30 x 30  |      2 |    92.0 |                .947 |
|  Test_02   | Phase-III-50  |      20 |  30 x 30  |      3 |    87.0 |                .934 |
|  Test_03   | Phase-III-50  |      50 |  30 x 35  |      3 |    86.2 |                .922 |
|  Test_04   | Phase-III-80  |      80 |  35 x 30  |      5 |    62.6 |                .812 |
|  Test_05   | Phase-III-80  |      80 |  45 x 35  |      7 |    62.9 |                .824 |
|  Test_06   | Phase-III-80  |      80 |  40 x 60  |      9 |    70.6 |                .859 |
|  Test_07   | Phase-III-80  |      80 |  60 x 40  |     13 |    65.4 |                .833 |
|  Test_08   | Phase-III-80  |      80 |  60 x 60  |     17 |    74.3 |                .877 |
|  Test_09   | Phase-III-100 |     100 |  80 x 120 |     21 |    59.7 |                .795 |
|  Test_10   | Phase-III-100 |     100 | 100 x 80  |     25 |    57.6 |                .779 |
|  Test_11   | Phase-III-200 |     200 | 100 x 100 |     29 |    52.8 |                .790 |
|  Test_12   | Phase-III-200 |     200 | 150 x 150 |     33 |    57.3 |                .777 |
|  Test_13   | Phase-III-200 |     400 | 150 x 150 |     37 |    34.9 |                .704 |
|  Test_14   | Phase-III-200 |     425 | 158 x 158 |     41 |    39.3 |                .721 |
 -->

# Bibtex
If you use this repo in your research, please cite our paper.
```bib
@article{jiang2022multi,
  title={Multi-Agent Path Finding via Tree LSTM},
  author={Jiang, Yuhao and Zhang, Kunjie and Li, Qimai and Chen, Jiaxin and Zhu, Xiaolong},
  journal={arXiv preprint arXiv:2210.12933},
  year={2022}
}
```