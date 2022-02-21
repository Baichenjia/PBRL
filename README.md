# PBRL

## Introduction

This is a Pytorch implementation for our paper on 

**Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning, ICLR 2022**.

## Prerequisites

- Python3.6 or 3.7 with pytorch 1.8
- [D4RL](https://github.com/rail-berkeley/d4rl) with v2 [dataset](http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2_old/) 
- OpenAI [Gym](http://gym.openai.com/) with [mujoco-py](https://github.com/openai/mujoco-py)

## Installation and Usage

Install the package of `rlkit` with
```
cd d4rl
pip install -e .
```

For running PBRL on the MuJoCo environments, run:

```
python examples/pevi_mujoco.py --env walker2d-medium-v2 --gpu 0
```

For running PBRL-Prior on the MuJoCo environments, run:

```
python examples/pevi_mujoco.py --env walker2d-medium-v2 --prior --gpu 0
```

For running PBRL on the Adroit environments, run:

```
python examples/pevi_adroit.py --env pen-cloned-v0 --gpu 0
```

For running PBRL-Prior on the Adroit environments, run:

```
python examples/pevi_adroit.py --env pen-cloned-v0 --prior --gpu 0
```

The core implementation is given in `d4rl/rlkit/torch/sac/pevi.py`

## Execution

The data for separate runs is stored on disk under the result directory with filename `<env-id>-<timestamp>/<seed>/`. Each run directory contains

- `debug.log` Record the epoch, Q-value, Uncertainty-value, scores.
- `progress.csv` Same data as `debug.log` but with csv format.
- `variant.json` The hyper-parameters in training.
- `models` The final actor-critic network.

The `evaluation/d4rl score` in `debug.log` or `progress.csv` records the normalized score in our paper.

In case of any questions, bugs, suggestions or improvements, please feel free to open an issue.
