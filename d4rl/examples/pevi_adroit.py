import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.pevi_adroit import PEVITrainer
from rlkit.torch.networks import FlattenMlp, RandomPrior
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import argparse
import os
import numpy as np
import datetime
import h5py
import d4rl
import gym
import random
import torch


def load_hdf5(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)
    replay_buffer._size = dataset['terminals'].shape[0]
    print('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size


def experiment(variant, log_dir=None):
    eval_env = gym.make(variant['env_name'])  # e.g. <TimeLimitNormalized: <OfflineWalker2dEnv instance>>
    expl_env = eval_env

    obs_dim = expl_env.observation_space.low.size  
    action_dim = eval_env.action_space.low.size    

    M = variant['layer_size']  # 256

    qfs, target_qfs = [], []
    for _ in range(variant['trainer_kwargs']['ensemble']):
        if variant['prior']:  # random prior
            print("***\n Applying Bootstrapped Networks with Random Prior Functions \n*******")
            qfn_main = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M, M])
            target_qfn_main = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M, M])

            qfn_prior = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M, M])
            target_qfn_prior = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M, M])
            qfs.append(RandomPrior(qfn_prior, qfn_main, variant['prior_scale']))
            target_qfs.append(RandomPrior(target_qfn_prior, target_qfn_main, variant['prior_scale']))
        else:
            qfn = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M, M])
            target_qfn = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M, M])
            qfs.append(qfn)
            target_qfs.append(target_qfn)

    policy = TanhGaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=[M, M, M])

    eval_policy = MakeDeterministic(policy)

    eval_path_collector = MdpPathCollector(eval_env, eval_policy)
    expl_path_collector = CustomMDPPathCollector(eval_env)

    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']
    replay_buffer = EnvReplayBuffer(variant['replay_buffer_size'], expl_env)

    if variant['load_buffer'] and buffer_filename is not None:
        replay_buffer.load_buffer(buffer_filename)
    elif 'random-expert' in variant['env_name']:
        load_hdf5(d4rl.basic_dataset(eval_env), replay_buffer)
    else:
        print("Load offline data ", os.path.split(eval_env.dataset_url)[-1])
        load_hdf5(d4rl.qlearning_dataset(eval_env), replay_buffer)

    trainer = PEVITrainer(
        env=eval_env,
        policy=policy,
        qfs=qfs,
        target_qfs=target_qfs,
        **variant['trainer_kwargs'])

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=True,
        batch_rl=variant['load_buffer'],  # true
        variant=variant,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

    # save model
    os.mkdir(os.path.join(log_dir, "models"))
    for idx in range(variant['trainer_kwargs']['ensemble']):
        torch.save(qfs[idx].state_dict(), os.path.join(log_dir, "models", "critic_"+str(idx)))
        torch.save(target_qfs[idx].state_dict(), os.path.join(log_dir, "models", "Tcritic_" + str(idx)))
    torch.save(policy.state_dict(), os.path.join(log_dir, "models", "policy"))


def enable_gpus(gpu_str):
    if gpu_str is not "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == "__main__":
    variant = dict(
        algorithm="PEVI",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=None,
        sparse_reward=False,
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=40000,
            # min Q
            temp=1.0,
            min_q_version=3,
            min_q_weight=1.0,

            # lagrange
            with_lagrange=True,  # Defaults to true
            lagrange_thresh=10.0,

            # extra params
            num_random=10,
            max_q_backup=False,
            deterministic_backup=False,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='pen-human-v0')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--max_q_backup", type=str, default="False")         # if we want to try max_{a'} backups, set this to true
    parser.add_argument("--deterministic_backup", type=str, default="True")  # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
    parser.add_argument("--policy_eval_start", default=40000, type=int)      # Defaulted to 20000 (40000 or 10000 work similarly)
    parser.add_argument('--min_q_weight', default=10.0, type=float)          # the value of alpha, set to 5.0 or 10.0 if not using lagrange
    parser.add_argument('--policy_lr', default=1e-4, type=float)             # Policy learning rate
    parser.add_argument('--min_q_version', default=3, type=int)              # min_q_version = 3
    parser.add_argument('--lagrange_thresh', default=-1.0, type=float)       # the value of tau, corresponds to the version
    parser.add_argument('--ucb_ratio', default=0.0001, type=float)
    parser.add_argument('--ensemble', default=10, type=int)

    # random prior
    parser.add_argument('--prior', action='store_true', default=False)  # if use randomized prior function
    parser.add_argument('--prior_scale', type=float, default=1.0)  # ratio of randomized prior function

    args = parser.parse_args()
    enable_gpus(args.gpu)

    variant['trainer_kwargs']['max_q_backup'] = (True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (True if args.deterministic_backup == 'True' else False)
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    variant['trainer_kwargs']['ucb_ratio'] = args.ucb_ratio
    variant['trainer_kwargs']['ensemble'] = args.ensemble
    variant['trainer_kwargs']['env_name'] = args.env
    if args.lagrange_thresh < 0.0:
        variant['trainer_kwargs']['with_lagrange'] = False

    variant['buffer_filename'] = None
    variant['load_buffer'] = True
    variant['env_name'] = args.env
    variant['seed'] = random.randint(0, 100000)  # random seeds

    # random prior
    variant['prior'] = args.prior
    variant['prior_scale'] = args.prior_scale

    setup_logger("", variant=variant, base_log_dir=os.path.join("result-PEVI-Adroit-finally",
        args.env+datetime.datetime.now().strftime("-%m-%d-%H-%M-%S")))

    ptu.set_gpu_mode(True)
    experiment(variant)
