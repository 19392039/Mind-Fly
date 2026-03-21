from __future__ import annotations

from dataclasses import dataclass
import mindspore as ms
from mindspore import Tensor, ops, Parameter

# Rollout数据类：存储一批训练数据的所有相关信息,组织从RolloutBuffer中采样出的mini-batch数据
@dataclass
class RolloutBatch:
    obs: Tensor         # [N, obs_dim] 观测值，N为批次大小，obs_dim为观测维度
    actions: Tensor     # [N, act_dim] 动作值，act_dim为动作维度
    logp: Tensor        # [N, 1] 动作的对数概率
    advantages: Tensor  # [N, 1] 优势函数值（GAE或MC计算）
    returns: Tensor     # [N, 1] 折扣回报值
    values: Tensor      # [N, 1] 价值函数预测值
    limits: Tensor      # [N, act_dim] 动作空间的限制/边界值


class RolloutBuffer:
    """
    强化学习中的经验回放缓冲区（Rollout Buffer）
    核心功能：
    1. 存储轨迹数据（观测、动作、奖励、价值等）
    2. 计算优势函数（GAE/MC两种方式）
    3. 生成打乱顺序的mini-batch用于训练
    """
    def __init__(self, T: int, B: int, obs_dim: int, act_dim: int) -> None:
        """
        初始化RolloutBuffer
        
        参数说明：
        - T: 每个环境的轨迹长度 (time steps per env)
        - B: 并行环境数量 (number of parallel environments)
        - obs_dim: 观测空间维度
        - act_dim: 动作空间维度
        
        核心逻辑：
        1. 计算总数据量N = T*B
        2. 初始化所有存储张量，维度为[N, 对应维度]
        3. 初始化数据写入指针_ptr=0
        """
        N = int(T * B)
        self.T, self.B, self.N = T, B, N
        # 数据存储张量(float32)
        self.obs = Tensor(ops.zeros((N, obs_dim), ms.float32))       # 观测存储
        self.actions = Tensor(ops.zeros((N, act_dim), ms.float32))   # 动作存储
        self.logp = Tensor(ops.zeros((N, 1), ms.float32))            # 动作对数概率存储
        self.rewards = Tensor(ops.zeros((N, 1), ms.float32))         # 奖励存储
        self.dones = Tensor(ops.zeros((N, 1), ms.float32))           # 终止标志存储
        self.values = Tensor(ops.zeros((N, 1), ms.float32))          # 价值预测存储
        self.limits = Tensor(ops.zeros((N, act_dim), ms.float32))    # 动作限制存储
        self._ptr = 0  # 数据写入指针，记录当前已存储的数据位置

    def add(self, obs: Tensor, act: Tensor, logp: Tensor, rew: Tensor,
            done: Tensor, val: Tensor, limits: Tensor) -> None:
        """
        向缓冲区添加一批数据
        
        参数说明：
        - obs: 观测张量 [n, obs_dim]
        - act: 动作张量 [n, act_dim]
        - logp: 动作对数概率张量 [n, 1]
        - rew: 奖励张量 [n,] 或 [n, 1]
        - done: 终止标志张量 [n,] 或 [n, 1]
        - val: 价值预测张量 [n,] 或 [n, 1]
        - limits: 动作限制张量 [n, act_dim]
        
        核心逻辑：
        1. 计算当前要写入的数据范围[i0, i1)
        2. 将输入数据写入对应位置（自动reshape确保维度匹配）
        3. 更新写入指针
        """
        n = obs.shape[0]
        i0, i1 = self._ptr, self._ptr + n
        
        self.obs[i0:i1] = obs
        self.actions[i0:i1] = act
        self.logp[i0:i1] = logp
        self.rewards[i0:i1] = rew.reshape(-1, 1)       
        self.dones[i0:i1] = done.reshape(-1, 1).astype(ms.float32)  
        self.values[i0:i1] = val.reshape(-1, 1)        
        self.limits[i0:i1] = limits
        
        self._ptr = i1

    @ms.jit
    def compute_gae(self, last_value: Tensor, gamma: float, lam: float):
        """
        计算广义优势估计（Generalized Advantage Estimation, GAE）
        GAE是PPO算法的核心，平衡了方差和偏差
        
        参数说明：
        - last_value: 最后一步的价值预测 [B, 1]，用于处理轨迹末端
        - gamma: 折扣因子（0~1），用于计算未来奖励的折扣
        - lam: GAE参数（0~1），控制优势估计的偏差-方差权衡
        
        核心计算逻辑：
        1. 重塑数据维度为[T, B, 1]，便于按时间步和环境并行计算
        2. 计算next_v（下一时间步的价值），最后一步使用last_value
        3. 逆序遍历时间步计算GAE：
           - delta = r_t + γ * V(s_{t+1}) * (1-d_t) - V(s_t) （时序差分误差）
           - gae = delta + γ * λ * (1-d_t) * gae （累积优势）
        4. 计算回报值ret = adv + v
        5. 优势归一化（减去均值除以标准差），提高训练稳定性
        
        返回值：
        - normalized_advantages: 归一化后的优势函数 [N, 1]
        - returns_flat: 折扣回报值 [N, 1]
        """
        T, B = self.T, self.B
        last_value = last_value.astype(ms.float32)
        r = self.rewards.reshape(T, B, 1).astype(ms.float32) 
        d = self.dones.reshape(T, B, 1).astype(ms.float32)    
        v = self.values.reshape(T, B, 1).astype(ms.float32)   
        
        next_v = ops.concat([v[1:], last_value.reshape(1, B, 1)], axis=0)
        adv = ops.zeros((T, B, 1), ms.float32)  
        gae = ops.zeros((B, 1), ms.float32)  
        
        # 核心：逆序计算GAE
        for t in reversed(range(T)):
            not_done = 1.0 - d[t]  # 非终止标志
            delta = r[t] + gamma * next_v[t] * not_done - v[t]
            # 累积计算GAE
            gae = delta + gamma * lam * not_done * gae
            adv[t] = gae  
        
        ret = adv + v
        advantages_flat = adv.reshape(T * B, 1)
        returns_flat = ret.reshape(T * B, 1)

        # 优势归一化
        adv_mean = advantages_flat.mean()
        adv_std = advantages_flat.std()
        adv_std = ops.maximum(adv_std, Tensor(1e-8, ms.float32))  
        
        normalized_advantages = (advantages_flat - adv_mean) / adv_std

        self.advantages = normalized_advantages
        self.returns = returns_flat
        
        return normalized_advantages, returns_flat

    @ms.jit
    def compute_mc_returns(self, gamma: float):
        """
        计算蒙特卡洛（MC）回报值（无偏但高方差的优势估计）
        适用于完整轨迹的优势计算，与GAE相比更简单但方差更大
        
        参数说明：
        - gamma: 折扣因子（0~1）
        
        核心计算逻辑：
        1. 重塑数据维度为[T, B, 1]
        2. 逆序遍历时间步，累加折扣奖励计算回报值：
           - next_ret = r_t + γ * (1-d_t) * next_ret
        3. 优势函数 = 实际回报 - 价值预测
        4. 优势归一化，提高训练稳定性
        
        返回值：
        - advantages: 归一化后的优势函数 [N, 1]
        - returns: 折扣回报值 [N, 1]
        """
        T, B = self.T, self.B
        
        # 维度重塑为 [T, B, 1]
        r = self.rewards.reshape(T, B, 1)
        d = self.dones.reshape(T, B, 1)
        v = self.values.reshape(T, B, 1)


        ret = ms.numpy.zeros((T, B, 1)) 
        next_ret = ops.zeros((B, 1), ms.float32) 

        # 蒙特卡洛回报
        for t in reversed(range(T)):
            not_done = 1.0 - d[t] 
            # 累加折扣奖励：当前奖励 + 折扣*未来回报
            next_ret = r[t] + gamma * not_done * next_ret
            ret[t] = next_ret  # 保存当前时间步的回报

        # 计算优势：实际回报 - 价值预测
        advantages = (ret - v).reshape(T * B, 1)
        returns = ret.reshape(T * B, 1)


        adv_mean = advantages.mean()
        adv_std = advantages.std()
        adv_std = ops.maximum(adv_std, 1e-8) 
        
        advantages = (advantages - adv_mean) / adv_std
        

        self.advantages = advantages
        self.returns = returns
        
        return advantages, returns
    
    def minibatches(self, batch_size: int):
        """
        生成打乱顺序的mini-batch数据迭代器
        用于训练时按批次采样数据，提高训练效率和稳定性
        
        参数说明：
        - batch_size: 每个mini-batch的大小
        
        核心逻辑：
        1. 生成0~N-1的随机排列索引，实现数据打乱
        2. 按batch_size切分索引，生成RolloutBatch对象
        3. 返回迭代器，每次yield一个RolloutBatch
        
        返回值：
        - 迭代器，每次返回一个RolloutBatch对象，包含该批次的所有数据
        """
        idx = ops.randperm(self.N)
        
        for i in range(0, self.N, batch_size):
            j = min(self.N, i + batch_size)  
            sl = idx[i:j]  

            yield RolloutBatch(
                obs=self.obs[sl],
                actions=self.actions[sl],
                logp=self.logp[sl],
                advantages=self.advantages[sl],
                returns=self.returns[sl],
                values=self.values[sl],
                limits=self.limits[sl],
            )