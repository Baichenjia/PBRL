import abc
from collections import OrderedDict

import gtimer as gt

from rlkit.core import logger, eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self, epoch):
        # if not self.trainer.discrete:
        #     snapshot = self._get_snapshot()
        #     logger.save_itr_params(epoch, snapshot)
        #     gt.stamp('saving')

        # 这里是关键的部分, 调用产生日志
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)   # 结束时清空 _epoch_paths 中的内容
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)         #

        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:  # 一般不进入该循环
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        # 依次调用各个组成部分的 get_diagnostics 方法来获得 Log 数据
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(self.replay_buffer.get_diagnostics(), prefix='replay_buffer/')

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(self.expl_data_collector.get_diagnostics(), prefix='exploration/')
        expl_paths = self.expl_data_collector.get_epoch_paths()
        # import ipdb; ipdb.set_trace()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',)
        if not self.batch_rl or self.eval_both:
            logger.record_dict(
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/",
            )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        # 返回 _epoch_paths, 里面含有多个 list, 每个 list 是一条路径
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(self.eval_env.get_diagnostics(eval_paths), prefix='evaluation/')

        # 以 evaluation paths 作为输入, 输出评价中的 returns/rewards/actions 等信息
        statistics_evaluation = eval_util.get_generic_path_information(eval_paths)
        logger.record_dict(statistics_evaluation, prefix="evaluation/")

        # d4rl score
        d4rl_score = self.eval_env.get_normalized_score(statistics_evaluation['Average Returns'])
        logger.record_dict({"d4rl score": d4rl_score*100.}, prefix="evaluation/")

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
