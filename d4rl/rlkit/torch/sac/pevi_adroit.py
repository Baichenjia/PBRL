from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd
from rlkit.torch.pytorch_util import PiecewiseSchedule, ConstantSchedule
from rlkit.core import logger
from collections import deque


class PEVITrainer(TorchTrainer):
    def __init__(
            self,
            env,
            env_name,
            policy,
            qfs,
            target_qfs,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            policy_eval_start=0,
            ucb_ratio=0.0001,
            ensemble=10,

            # CQL
            min_q_version=3,
            temp=1.0,
            min_q_weight=1.0,

            # sort of backup
            max_q_backup=False,
            deterministic_backup=True,
            num_random=10,
            with_lagrange=False,
            lagrange_thresh=0.0,
    ):
        super().__init__()
        assert env_name in ["pen-human-v0", "pen-expert-v0", "pen-cloned-v0", "hammer-human-v0",
                "hammer-expert-v0", "hammer-cloned-v0", "door-human-v0", "door-expert-v0",
                "door-cloned-v0", "relocate-human-v0", "relocate-expert-v0", "relocate-cloned-v0",
                "kitchen-partial-v0", "kitchen-mixed-v0", "kitchen-complete-v0"]

        self.env_name = env_name
        self.env = env
        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.soft_target_tau = soft_target_tau  # 0.005

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

        if self.use_automatic_entropy_tuning:  # True
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # -6.
            self.log_alpha = ptu.zeros(1, requires_grad=True)  # [0.]
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)  # policy_lr=0.0001

        self.with_lagrange = with_lagrange  # True or False.
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh  # 5.0
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)  # [0.]
            self.alpha_prime_optimizer = optimizer_class([self.log_alpha_prime], lr=qf_lr)  # qf_lr=0.0003

        self.plotter = plotter  # None
        self.render_eval_paths = render_eval_paths  # False

        self.qf_criterion_all = nn.MSELoss(reduction='none')
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.discount = discount  # 0.99
        self.reward_scale = reward_scale  # 1
        self.eval_statistics = OrderedDict() 
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1

        # min Q
        self.temp = temp  # 1.0
        self.min_q_version = min_q_version  # 3
        self.min_q_weight = min_q_weight  # 10.0

        self.softmax = torch.nn.Softmax(dim=1)  #
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)

        self.max_q_backup = max_q_backup  # False
        self.deterministic_backup = deterministic_backup  # True
        self.num_random = num_random  # 10

        # For implementation on the discrete env
        self.discrete = False

        # ucb
        self.ensemble = ensemble
        self.ucb_ratio = ucb_ratio
        logger.log(f"Ensemble: {self.ensemble}, UCB ratio of offline data: {self.ucb_ratio}")

        self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)
        self.qfs_optimizers = []
        for i in range(self.ensemble):
            self.qfs_optimizers.append(optimizer_class(self.qfs[i].parameters(), lr=qf_lr))

        if self.env_name in ["pen-human-v0"]:
            self.ood_weight1 = 0.001
            self.ood_weight2 = 0.001
        elif "expert" in self.env_name:
            self.ood_weight1 = 0.2 if self.env_name == "relocate-expert-v0" else 0.1
            self.ood_weight2 = 0.2 if self.env_name == "door-expert-v0" else 0.4
        else:
            self.ood_weight1 = 0.01
            self.ood_weight2 = 0.01
        logger.log(f"OOD weight {self.ood_weight1}, {self.ood_weight2}")

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]  # 2560
        obs_shape = obs.shape[0]         # 256
        num_repeat = int(action_shape / obs_shape)  # 10
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])  # （2560, obs_dim）
        preds = network(obs_temp.cuda(), actions.cuda())  # (2560, 1)
        preds = preds.view(obs.shape[0], num_repeat, 1)   # (256, 10, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        # obs.shape=(256, obs_dim), num_actions=10. obs_temp.shape=(2560, 10)
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        # new_obs_actions=(2560, act_dim), new_obs_log_pi.shape=(2560, 1)
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(obs_temp, reparameterize=True, return_log_prob=True)

        if not self.discrete:
            # new_obs_actions.shape=(2560, act_dim), new_obs_log_pi.shape=(256, 10, 1)
            return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)
        else:
            return new_obs_actions

    def ucb_func(self, obs, act, mean=False):
        action_shape = act.shape[0]  # 2560
        obs_shape = obs.shape[0]  # 256
        num_repeat = int(action_shape / obs_shape)  # 10
        if num_repeat != 1:
            obs = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat,
                                                                  obs.shape[1])  # （2560, obs_dim）
        q_pred = []
        for i in range(self.ensemble):
            q_pred.append(self.qfs[i](obs.cuda(), act.cuda()))
        ucb = torch.std(torch.hstack(q_pred), dim=1, keepdim=True)   # (2560, 1)
        assert ucb.size() == (obs.size()[0], 1)
        if mean:
            q_pred = torch.mean(torch.hstack(q_pred), dim=1, keepdim=True)
        return ucb, q_pred

    def ucb_func_target(self, obs_next, act_next):
        action_shape = act_next.shape[0]  # 2560
        obs_shape = obs_next.shape[0]     # 256
        num_repeat = int(action_shape / obs_shape)  # 10
        if num_repeat != 1:
            obs_next = obs_next.unsqueeze(1).repeat(1, num_repeat, 1).view(obs_next.shape[0] * num_repeat, obs_next.shape[1])  # （2560, obs_dim）

        target_q_pred = []
        for i in range(self.ensemble):
            target_q_pred.append(self.target_qfs[i](obs_next.cuda(), act_next.cuda()))
        ucb_t = torch.std(torch.hstack(target_q_pred), dim=1, keepdim=True)
        assert ucb_t.size() == (obs_next.size()[0], 1)
        return ucb_t, target_q_pred

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards = batch['rewards']             # shape=(256, 1)
        terminals = batch['terminals']         # shape=(256, 1)
        obs = batch['observations']            # shape=(256, 17)
        actions = batch['actions']             # shape=(256, 6)
        next_obs = batch['next_observations']  # shape=(256, 17)
        batch_size = rewards.size()[0]
        action_dim = actions.size()[-1]
        """ Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                                        obs, reparameterize=True, return_log_prob=True)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        """
        QF Loss
        """
        new_next_actions, _, _, new_log_pi, *_ = self.policy(next_obs, reparameterize=True, return_log_prob=True)
        new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(obs, reparameterize=True, return_log_prob=True)

        # compute ucb
        ucb_current, q_pred = self.ucb_func(obs, actions)
        ucb_next, target_q_pred = self.ucb_func_target(next_obs, new_next_actions)

        # q-target.
        self._num_q_update_steps += 1

        # For OOD actions
        random_actions_tensor = torch.FloatTensor(batch_size * self.num_random, actions.shape[-1]).uniform_(-1, 1)  # num_actions=10
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random, network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random, network=self.policy)

        ucb_rand, qf_rand_all = self.ucb_func(obs, random_actions_tensor)
        ucb_curr_actions, qf_curr_actions_all = self.ucb_func(obs, curr_actions_tensor)
        ucb_next_actions, qf_next_actions_all = self.ucb_func(next_obs, new_curr_actions_tensor)
        assert ucb_rand.size() == ucb_curr_actions.size() == ucb_next_actions.size() == qf_rand_all[0].size() == \
            qf_curr_actions_all[0].size() == qf_next_actions_all[0].size() == (2560, 1)

        for qf_index in np.arange(self.ensemble):
            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount *\
                       (target_q_pred[qf_index] - self.ucb_ratio * ucb_next)
            q_target = q_target.detach()

            # critic loss
            qf_loss_in = self.qf_criterion(q_pred[qf_index], q_target)  

            # for odd actions
            cat_qf_ood = torch.cat([qf_curr_actions_all[qf_index], qf_next_actions_all[qf_index]], 0)
            assert cat_qf_ood.size() == (2560*2, 1)

            cat_qf_ood_target = torch.cat([
                torch.maximum(qf_curr_actions_all[qf_index] - self.ood_weight2 * ucb_curr_actions, torch.zeros(1).cuda()),
                torch.maximum(qf_next_actions_all[qf_index] - self.ood_weight1 * ucb_next_actions, torch.zeros(1).cuda())], 0)
            cat_qf_ood_target = cat_qf_ood_target.detach()
            assert cat_qf_ood_target.size() == (2560*2, 1)
            qf_loss_ood = self.qf_criterion(cat_qf_ood, cat_qf_ood_target)

            qf_loss = qf_loss_in + qf_loss_ood

            # Update the Q-functions
            self.qfs_optimizers[qf_index].zero_grad()
            qf_loss.backward(retain_graph=True)
            self.qfs_optimizers[qf_index].step()

        q_new_actions_all = []
        for i in range(self.ensemble):
            q_new_actions_all.append(self.qfs[i](obs, new_obs_actions))
        q_new_actions = torch.min(torch.hstack(q_new_actions_all), dim=1, keepdim=True).values
        assert q_new_actions.size() == (batch_size, 1)

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft Updates. 
        for i in np.arange(self.ensemble):
            ptu.soft_update_from_to(self.qfs[i], self.target_qfs[i], self.soft_target_tau)

        # Save some statistics for eval
        if self._need_to_update_eval_statistics:
            with torch.no_grad():
                ucb_rand, q_rand = self.ucb_func(obs, torch.distributions.uniform.Uniform(-1, 1).sample((batch_size, action_dim)), mean=True)
                ucb_noise1, q_noise1 = self.ucb_func(obs, actions + torch.normal(mean=torch.zeros((batch_size, action_dim)), std=torch.ones((batch_size, action_dim)) * 0.1).cuda(), mean=True)
                ucb_noise2, q_noise2 = self.ucb_func(obs, actions + torch.normal(mean=torch.zeros((batch_size, action_dim)), std=torch.ones((batch_size, action_dim)) * 1.).cuda(), mean=True)
                ucb_noise3, q_noise3 = self.ucb_func(obs, actions + torch.normal(mean=torch.zeros((batch_size, action_dim)), std=torch.ones((batch_size, action_dim)) * 2.).cuda(), mean=True)

            self._need_to_update_eval_statistics = False
            self.eval_statistics['Q CurrPolicy'] = np.mean(ptu.get_numpy(qf_curr_actions_all[0]))
            self.eval_statistics['Q RandomAction'] = np.mean(ptu.get_numpy(qf_rand_all[0]))
            self.eval_statistics['Q NextAction'] = np.mean(ptu.get_numpy(qf_next_actions_all[0]))

            self.eval_statistics['UCB CurrPolicy'] = np.mean(ptu.get_numpy(ucb_curr_actions))
            self.eval_statistics['UCB RandomAction'] = np.mean(ptu.get_numpy(ucb_rand))
            self.eval_statistics['UCB NextAction'] = np.mean(ptu.get_numpy(ucb_next_actions))

            self.eval_statistics['Q Offline'] = np.mean(ptu.get_numpy(q_pred[0]))
            self.eval_statistics['Q Noise1'] = np.mean(ptu.get_numpy(q_noise1))
            self.eval_statistics['Q Noise2'] = np.mean(ptu.get_numpy(q_noise2))
            self.eval_statistics['Q Noise3'] = np.mean(ptu.get_numpy(q_noise3))
            self.eval_statistics['Q Rand'] = np.mean(ptu.get_numpy(q_rand))

            self.eval_statistics['UCB Offline'] = np.mean(ptu.get_numpy(ucb_current))
            self.eval_statistics['UCB Next'] = np.mean(ptu.get_numpy(ucb_next))
            self.eval_statistics['UCB Noise1'] = np.mean(ptu.get_numpy(ucb_noise1))
            self.eval_statistics['UCB Noise2'] = np.mean(ptu.get_numpy(ucb_noise2))
            self.eval_statistics['UCB Noise3'] = np.mean(ptu.get_numpy(ucb_noise3))
            self.eval_statistics['UCB Rand'] = np.mean(ptu.get_numpy(ucb_rand))

            self.eval_statistics['QF Loss in'] = np.mean(ptu.get_numpy(qf_loss_in))
            self.eval_statistics['QF Loss ood'] = np.mean(ptu.get_numpy(qf_loss_ood))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))

            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q Targets', ptu.get_numpy(q_target)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis', ptu.get_numpy(log_pi)))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qfs,
            self.target_qfs,
        ]
        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qfs=self.qfs,
            target_qf1=self.target_qfs,
        )

