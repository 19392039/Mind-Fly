from __future__ import annotations

import math
import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor, Parameter
from typing import NamedTuple
from mindspore.nn.probability.distribution import Normal

from mind_encoder import RayEncoder

class PolicyOutput(NamedTuple):
    """
    PPO策略输出，用于封装策略网络输出的所有关键信息
    字段说明：
    - action: 输出动作张量，形状[B, A]
    - logp: 动作的对数概率，形状[B, 1]
    - mu: 策略网络输出的均值，形状[B, A]
    - std: 策略网络输出的标准差，形状[B, A]
    """
    action: Tensor
    logp: Tensor
    mu: Tensor
    std: Tensor


def _tanh_log_det_jac(pre_tanh: Tensor) -> Tensor:
    """
    计算Tanh变换的对数雅可比行列式
    功能：为Tanh Squash变换提供对数概率修正项，保证概率分布的正确性
    核心逻辑：使用数值稳定的公式计算雅可比行列式的对数，避免梯度消失/爆炸
    输入：pre_tanh - Tanh变换前的原始值张量
    输出：对数雅可比行列式值，形状与输入一致
    """
    softplus = ops.Softplus()
    return 2.0 * (math.log(2.0) - pre_tanh - softplus(-2.0 * pre_tanh))

def _squash(mu: Tensor, log_std: Tensor, eps: Tensor, limits: Tensor):
    """
    Tanh Squash变换函数
    功能：将正态分布的采样值通过Tanh变换压缩到[-1,1]区间，并缩放至指定动作范围，同时修正对数概率
    核心逻辑：
    1. 基于均值和标准差生成原始采样值
    2. Tanh变换压缩值域，乘以动作限制缩放至目标范围
    3. 计算雅可比行列式修正对数概率，保证概率积分不变
    输入：
    - mu: 正态分布均值 [B, A]
    - log_std: 正态分布对数标准差 [B, A]
    - eps: 标准正态分布采样值 [B, A]
    - limits: 动作范围限制 [B, A]
    输出：
    - a_scaled: 缩放后的动作 [B, A]
    - logp: 修正后的对数概率 [B, 1]
    - std: 标准差 [B, A]
    """
    std = ops.exp(log_std)
    pre_tanh = mu + std * eps
    
    a = ops.tanh(pre_tanh)
    
    log_det = _tanh_log_det_jac(pre_tanh)

    mu = mu.astype(ms.float32)
    std = std.astype(ms.float32)
    
    dist = Normal(mu, std)
    log_prob_original = dist.log_prob(pre_tanh)
    
    logp = (log_prob_original - log_det).sum(-1, keepdims=True)
    
    a_scaled = a * limits
    return a_scaled, logp, std

def _inverse_squash(action_scaled: Tensor, limits: Tensor) -> Tensor:
    """
    Tanh Squash逆变换函数
    功能：将缩放后的动作反变换回原始正态分布空间，用于动作评估
    核心逻辑：
    1. 反缩放动作到[-1,1]区间，添加数值保护防止log(0)
    2. 使用反双曲正切函数(atanh)还原Tanh变换前的值
    输入：
    - action_scaled: 缩放后的动作 [B, A]
    - limits: 动作范围限制 [B, A]
    输出：原始空间的值 [B, A]
    """
    eps = 1e-12
    a = ops.unstack(action_scaled / ops.maximum(limits, eps))
    a = ops.clip_by_value(action_scaled / ops.maximum(limits, eps), -0.999999, 0.999999)
    
    return 0.5 * (ops.log1p(a) - ops.log1p(-a))

class PPOPolicy(nn.Cell):
    """
    PPO策略网络（核心模块）
    功能：基于RayEncoder的特征提取，实现带Tanh Squash的正态分布策略和价值函数估计
    核心逻辑：
    1. 特征提取：通过RayEncoder提取输入向量的高维特征
    2. 策略头：输出动作分布的均值和可学习的对数标准差，生成受限的动作
    3. 价值头：估计状态价值函数
    4. 动作评估：计算给定动作的对数概率、熵和状态价值
    """
    def __init__(self, 
                 vec_dim: int, 
                 action_dim: int = 3, 
                 hidden: int = 64, 
                 d_model: int = 128, 
                 num_queries: int = 4, 
                 num_heads: int = 4, 
                 learnable_queries: bool = True, 
                 log_std_min: float = -5.0, 
                 log_std_max: float = 2.0):
        super(PPOPolicy, self).__init__()

        self.encoder = RayEncoder(
            vec_dim=vec_dim, 
            hidden=hidden, 
            d_model=d_model, 
            num_queries=num_queries, 
            num_heads=num_heads, 
            learnable_queries=learnable_queries
        )
        
        self.mu_head = nn.Dense(256, action_dim)
        self.log_std = Parameter(ops.zeros(action_dim, ms.float32), name="log_std")
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        self.value_head = nn.SequentialCell([
            nn.Dense(256, 256),
            nn.ReLU(),
            nn.Dense(256, 1)
        ])
        self.softplus = ops.Softplus()
        self.exp = ops.Exp()

    def _core(self, obs_vec: Tensor):
        """
        PPO策略核心计算函数
        功能：封装特征提取、均值/对数标准差计算、价值估计的核心逻辑
        输入：obs_vec - 观测向量 [B, vec_dim]
        输出：
        - mu: 动作分布均值 [B, action_dim]
        - log_std: 裁剪后的对数标准差 [B, action_dim]
        - v: 状态价值估计 [B, 1]
        """
        g, _, _ = self.encoder(obs_vec)
        mu = self.mu_head(g)
        log_std_raw = self.log_std.view(1, -1)
        log_std = ops.clip_by_value(log_std_raw, self._log_std_min, self._log_std_max)
        log_std = ops.broadcast_to(log_std, mu.shape)
        v = self.value_head(g)
        return mu, log_std, v

    def act(self, obs_vec: Tensor, limits: Tensor) -> PolicyOutput:
        """
        动作生成函数
        功能：基于当前观测生成动作，并返回相关的概率信息
        核心逻辑：
        1. 获取策略分布参数（均值、对数标准差）
        2. 采样正态分布并通过Tanh Squash生成受限动作
        3. 停止梯度传播，避免采样过程影响网络训练
        输入：
        - obs_vec: 观测向量 [B, vec_dim]
        - limits: 动作范围限制 [B, action_dim]
        输出：PolicyOutput对象，包含动作、对数概率、均值、标准差
        """
        mu, log_std, _ = self._core(obs_vec)
        eps = ops.standard_normal(mu.shape)
        mu_det = ops.stop_gradient(mu)
        log_std_det = ops.stop_gradient(log_std)
        a_scaled, logp, std = _squash(mu_det, log_std_det, eps, limits)
        
        return PolicyOutput(
            action=a_scaled, 
            logp=logp, 
            mu=mu_det, 
            std=std
        )

    def evaluate_actions(self, obs_vec: Tensor, actions_scaled: Tensor, limits: Tensor):
        """
        动作评估函数
        功能：评估给定动作在当前策略下的对数概率、熵和状态价值
        核心逻辑：
        1. 获取策略分布参数和状态价值
        2. 逆变换将动作还原到原始分布空间
        3. 手动计算对数概率（含Tanh修正）和熵
        输入：
        - obs_vec: 观测向量 [B, vec_dim]
        - actions_scaled: 已缩放的动作 [B, action_dim]
        - limits: 动作范围限制 [B, action_dim]
        输出：
        - logp: 动作的对数概率 [B, 1]
        - ent: 策略分布的熵 [B, 1]
        - v: 状态价值估计 [B, 1]
        """
        mu, log_std, v = self._core(obs_vec)
        mu = mu.astype(ms.float32)
        std = ops.exp(log_std).astype(ms.float32)+1e-6
        
        y = _inverse_squash(actions_scaled, limits)
        
        log_2pi = math.log(2 * math.pi)
        var = ops.pow(std, 2)
        logp_raw = -0.5 * (ops.pow(y - mu, 2) / (var + 1e-8) + log_2pi + 2 * log_std)
        
        log_det = _tanh_log_det_jac(y)
        logp = (logp_raw - log_det).sum(axis=-1, keepdims=True)
        
        ent_raw = 0.5 + 0.5 * log_2pi + log_std
        ent = ent_raw.sum(axis=-1, keepdims=True)
        
        return logp, ent, v