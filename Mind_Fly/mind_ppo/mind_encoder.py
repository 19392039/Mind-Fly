from __future__ import annotations

import mindspore
import math
import mindspore.nn as nn
from mindspore import ops, Tensor, Parameter
from mindspore.common.initializer import Normal, initializer
from mindspore.common import dtype as mstype

def _circular_pad1d(x: Tensor, pad: int) -> Tensor:
    """
    一维循环填充函数
    功能：对输入张量的最后一维进行循环填充，左侧填充最后pad个元素，右侧填充前pad个元素
    输入：
    - x: 输入张量，维度为[B, C, L]或[B, L]等
    - pad: 填充长度，若pad<=0则直接返回原张量
    输出：填充后的张量，最后一维长度增加2*pad
    """
    if pad <= 0:
        return x

    left = x[..., -pad:]
    right = x[..., :pad]
    
    return ops.concat([left, x, right], axis=-1)

class SqueezeExcite1D(nn.Cell):
    """
    一维通道注意力模块
    功能：通过挤压-激励机制自适应调整各通道的权重，增强重要通道特征，抑制次要通道特征
    核心逻辑：
    1. 挤压：对通道维度进行全局平均池化，得到通道级全局特征
    2. 激励：通过两层全连接层学习通道权重
    3. 加权：将学习到的权重乘回原特征图
    """
    def __init__(self, ch: int, r: int = 4):
        super(SqueezeExcite1D, self).__init__()
        hid = max(8, ch // r)
        
        self.fc1 = nn.Dense(ch, hid)
        self.fc2 = nn.Dense(hid, ch)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def construct(self, x: Tensor):
        s = ops.reduce_mean(x, -1)
        s = self.relu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        return x * ops.expand_dims(s, -1)

class DepthwiseSeparable1D(nn.Cell):
    """
    一维深度可分离卷积模块
    功能：将标准卷积拆分为深度卷积和逐点卷积，减少参数量和计算量，同时保持卷积的特征提取能力
    核心逻辑：
    1. 深度卷积：每个输入通道单独卷积，不跨通道交互
    2. 逐点卷积：1x1卷积融合跨通道特征
    3. 结合循环填充、批归一化和GELU激活
    """
    def __init__(self, ch: int, kernel: int = 5, dilation: int = 1):
        super(DepthwiseSeparable1D, self).__init__()
        self.kernel = int(kernel)
        self.dil = int(dilation)
        
        self.dw = nn.Conv1d(ch, ch, kernel_size=self.kernel, 
                           group=ch, has_bias=False, dilation=self.dil,
                           pad_mode='valid')
        
        self.pw = nn.Conv1d(ch, ch, kernel_size=1, has_bias=True)
        
        self.bn = nn.BatchNorm1d(ch)
        self.gelu = nn.GELU()

    def construct(self, x: Tensor):
        pad = ((self.kernel - 1) * self.dil) // 2
        if pad > 0:
            x = _circular_pad1d(x, pad)
        
        out = self.dw(x)
        out = self.gelu(out)
        out = self.pw(out)
        out = self.bn(out)
        return out 
    
class RayBranch(nn.Cell):
    """
    射线特征提取分支
    功能：通过多尺度扩张深度可分离卷积提取射线特征，结合通道注意力增强特征表达
    核心逻辑：
    1. 维度扩展：将输入通道映射到隐藏层维度
    2. 多尺度扩张卷积：使用不同扩张率的深度可分离卷积捕捉多尺度特征
    3. 通道注意力：SE模块增强关键通道特征
    """
    def __init__(self, in_ch: int = 1, hidden: int = 64, layers: int = 4, kernel: int = 5):
        super(RayBranch, self).__init__()
        self.in_ch = int(in_ch)
        
        self.expand = nn.Conv1d(self.in_ch, hidden, kernel_size=1, has_bias=True)
        
        dilations = [1, 2, 4, 8][:layers]
        blocks = []
        for d in dilations:
            blocks.append(DepthwiseSeparable1D(hidden, kernel=kernel, dilation=d))
            blocks.append(nn.GELU())
            blocks.append(SqueezeExcite1D(hidden, r=4))
        
        self.blocks = nn.SequentialCell(blocks)

    def construct(self, x: Tensor):
        if x.ndim == 2:
            x = ops.expand_dims(x, 1)
            
        x = self.expand(x)
        x = self.blocks(x)
        return x

class RayEncoder(nn.Cell):
    """
    射线特征编码器（核心模块）
    功能：融合射线特征提取、位姿编码和多头注意力机制，生成高维特征表示
    核心逻辑：
    1. 数据拆分：将输入向量拆分为射线观测和位姿两部分
    2. 特征提取：通过RayBranch提取射线特征并生成注意力的K/V矩阵
    3. Query生成：基于位姿特征和可学习Query生成注意力查询向量
    4. 多头注意力：计算注意力权重并加权求和，融合多尺度特征
    5. 特征聚合：拼接注意力输出、Query均值和Value均值，通过MLP输出最终特征
    """
    def __init__(self, vec_dim: int, hidden: int = 64, d_model: int = 128, *, 
                 num_queries: int = 1, num_heads: int = 1, learnable_queries: bool = True):
        super(RayEncoder, self).__init__()
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.learnable_queries = bool(learnable_queries)
        
        self.pose_dim = 7
        assert vec_dim >= self.pose_dim, f"vec_dim must be N + {self.pose_dim}, got {vec_dim}"
        self.vec_dim = int(vec_dim)
        self.N = max(0, vec_dim - self.pose_dim)
        self.ray_in_ch = 1
        self.hidden = int(hidden)
        self.d_model = int(d_model)
        assert self.d_model % max(1, self.num_heads) == 0, "d_model must be divisible by num_heads"

        self.br_obs = RayBranch(in_ch=self.ray_in_ch, hidden=hidden)
        self.to_k = nn.Conv1d(hidden, d_model, kernel_size=1, has_bias=True)
        self.to_v = nn.Conv1d(hidden, d_model, kernel_size=1, has_bias=True)

        self.pose_mlp = nn.SequentialCell([
            nn.Dense(self.pose_dim, d_model),
            nn.ReLU(),
            nn.Dense(d_model, d_model)
        ])

        if self.learnable_queries:
            init_scale = 1.0 / math.sqrt(max(1, d_model))
            self.q_params = Parameter(initializer(Normal(init_scale), [self.num_queries, d_model]), name="q_params")
            self.to_q = nn.Identity()
        else:
            self.to_q = nn.Dense(d_model, d_model * self.num_queries) if self.num_queries > 1 else nn.Identity()

        self.post = nn.SequentialCell([
            nn.Dense(d_model * 3, 256), 
            nn.ReLU(),
            nn.Dense(256, 256),
            nn.ReLU()
        ])
        
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def split(self, vec: Tensor):
        """
        输入向量拆分函数
        功能：将输入向量拆分为射线观测部分和位姿部分，并调整射线观测的维度格式
        输入：vec - 输入向量，维度为[B, vec_dim]
        输出：
        - d_obs: 射线观测张量，维度为[B, ray_in_ch, N]
        - pose: 位姿张量，维度为[B, pose_dim]
        """
        d_len = self.N * self.ray_in_ch
        d_obs = vec[:, :d_len]
        pose = vec[:, d_len:d_len + self.pose_dim]
        
        if self.ray_in_ch > 1:
            d_obs = self.reshape(d_obs, (vec.shape[0], self.ray_in_ch, self.N))
        else:
            d_obs = ops.expand_dims(d_obs, 1)
        return d_obs, pose

    def construct(self, vec: Tensor):
        d_obs, pose = self.split(vec)
        
        f_map = self.br_obs(d_obs)
        k = self.transpose(self.to_k(f_map), (0, 2, 1))
        v = self.transpose(self.to_v(f_map), (0, 2, 1))

        q_pose = self.pose_mlp(pose)
        if self.learnable_queries:
            q = ops.expand_dims(self.q_params, 0) + ops.expand_dims(q_pose, 1)
        else:
            if self.num_queries > 1:
                qm = self.to_q(q_pose)
                q = self.reshape(qm, (qm.shape[0], self.num_queries, self.d_model))
            else:
                q = ops.expand_dims(q_pose, 1)

        k = k.astype(mstype.float32)
        v = v.astype(mstype.float32)
        q = q.astype(mstype.float32)

        h = self.num_heads
        dh = self.d_model // h
        b_size = q.shape[0]
        
        k_h = self.reshape(k, (b_size, -1, h, dh))
        v_h = self.reshape(v, (b_size, -1, h, dh))
        q_h = self.reshape(q, (b_size, -1, h, dh))

        q_h_t = ops.transpose(q_h, (0, 2, 1, 3))
        k_h_t = ops.transpose(k_h, (0, 2, 3, 1))
        
        attn_logits = ops.matmul(q_h_t, k_h_t) / math.sqrt(dh)
        attn = self.softmax(attn_logits)
        
        v_h_t = ops.transpose(v_h, (0, 2, 1, 3))
        z_h_t = ops.matmul(attn, v_h_t)          
        
        z_h = ops.transpose(z_h_t, (0, 2, 1, 3))
        z = self.reshape(z_h, (b_size, -1, self.d_model))

        z_mean = ops.reduce_mean(z, 1)
        q_mean = ops.reduce_mean(q, 1)
        gavg = ops.reduce_mean(v, 1)

        g = ops.concat([z_mean, gavg, q_mean], axis=-1)
        g = self.post(g)
        
        return g, k, v