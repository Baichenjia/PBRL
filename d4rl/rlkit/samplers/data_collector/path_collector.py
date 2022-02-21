from collections import deque, OrderedDict

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.rollout_functions import rollout, multitask_rollout, function_rollout
from rlkit.samplers.data_collector.base import PathCollector
import numpy as np


class MdpPathCollector(PathCollector):    # 这个基类里面基本什么也没写
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            sparse_reward=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._sparse_reward = sparse_reward

    def update_policy(self, new_policy):
        self._policy = new_policy
    
    def collect_new_paths(self, max_path_length, num_steps, discard_incomplete_paths, policy_fn=None):
        paths = []
        num_steps_collected = 0         # 已经保存的steps
        while num_steps_collected < num_steps:

            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,   # 剩余的 steps
            )
            # 调用 rollout 函数执行最多 max_path_length_this_loop 时间步, 如果中途 done, 则提前结束
            # 返回一个dict, key 包含 observations, actions, rewards next_observations, terminals, agent_infos, env_infos 等信息, value 是 numpy
            path = rollout(self._env, self._policy, max_path_length=max_path_length_this_loop)
            path_len = len(path['actions'])    # 记录 path 的长度
            # 如果提前终止, 但是最后一个时间步不是done, 表示已经到整个 collect path 的结束, 且保存的最后一个周期不是一个完整的周期, 此时丢弃最后一个周期
            if path_len != max_path_length and not path['terminals'][-1] and discard_incomplete_paths:
                break
            # 记录
            num_steps_collected += path_len
            # print("**", max_path_length, num_steps, num_steps_collected, path_len)  # 都是 1000

            # Used to sparsify reward
            if self._sparse_reward:
                random_noise = np.random.normal(size=path['rewards'].shape)
                path['rewards'] = path['rewards'] + 1.0*random_noise 

            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        # 日志, 在每个周期结束时会调用这个函数
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),   # 总 steps 数
            ('num paths total', self._num_paths_total)])  # 总周期数
        # 返回 path_lens 的 mean, std, max, min 等统计量
        stats.update(create_stats_ordered_dict(
            "path length", path_lens, always_show_all_stats=True))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )


class CustomMDPPathCollector(PathCollector):
    # 这个类和它的基类是基本是一致的
    def __init__(self, env, max_num_epoch_paths_saved=None, render=False, render_kwargs=None):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(self, policy_fn, max_path_length, num_steps, discard_incomplete_paths):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = function_rollout(
                self._env,
                policy_fn,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths
    
    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
        )


class GoalConditionedPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = multitask_rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                observation_key=self._observation_key,
                desired_goal_key=self._desired_goal_key,
                return_dict_obs=True,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
