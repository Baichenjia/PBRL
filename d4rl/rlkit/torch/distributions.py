import torch
from torch.distributions import Distribution, Normal
import rlkit.torch.pytorch_util as ptu


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)   # 高斯分布
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)     # 从高斯分布中采样
        if return_pre_tanh_value:
            return torch.tanh(z), z     # 将采样结果输入到 tanh 中. (原因是策略的输出有限制)
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1+value) / (1-value)) / 2   # arc-tan 函数, 还原到取 tanh 之前时候
        # log of action - log(1-tanh(a)*tanh(a)+e)
        # 参考 https://garage.readthedocs.io/en/v2020.06.1/_modules/garage/torch/distributions/tanh_normal.html#TanhNormal.log_prob
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.  可微分的采样
        """
        z = (self.normal_mean + self.normal_std *
             Normal(ptu.zeros(self.normal_mean.size()), ptu.ones(self.normal_std.size())).sample())
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
