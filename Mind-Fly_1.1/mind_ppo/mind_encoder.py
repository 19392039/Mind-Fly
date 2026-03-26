from __future__ import annotations

import mindspore
import math
import mindspore.nn as nn
from mindspore import ops, Tensor, Parameter
from mindspore.common.initializer import Normal, initializer
from mindspore.common import dtype as mstype

# 循环填充
def _circular_pad1d(x: Tensor, pad: int) -> Tensor:
    if pad <= 0:
        return x

    left = x[..., -pad:]
    right = x[..., :pad]
    
    return ops.concat([left, x, right], axis=-1)

# SE模块
class SqueezeExcite1D(nn.Cell):
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

# 深度可分离卷积
class DepthwiseSeparable1D(nn.Cell):
    def __init__(self, ch: int, kernel: int = 5, dilation: int = 1):
        super(DepthwiseSeparable1D, self).__init__()
        self.kernel = int(kernel)
        self.dil = int(dilation)
        
        # Depthwise Conv: group 数等于通道数
        self.dw = nn.Conv1d(ch, ch, kernel_size=self.kernel, 
                           group=ch, has_bias=False, dilation=self.dil,
                           pad_mode='valid') # 因为我们要手动做循环填充，所以设为 valid
        
        # Pointwise Conv: 1x1 卷积
        self.pw = nn.Conv1d(ch, ch, kernel_size=1, has_bias=True)
        
        self.bn = nn.BatchNorm1d(ch)
        self.gelu = nn.GELU()

    def construct(self, x: Tensor):
        # 1. 计算并执行循环填充
        pad = ((self.kernel - 1) * self.dil) // 2
        if pad > 0:
            # 调用你之前定义的 _circular_pad1d 函数
            x = _circular_pad1d(x, pad)
        
        # 2. 前向计算
        out = self.dw(x)
        out = self.gelu(out)
        out = self.pw(out)
        out = self.bn(out)
        return out 
    
#扩张卷积残差分支
class RayBranch(nn.Cell):
    def __init__(self, in_ch: int = 1, hidden: int = 64, layers: int = 4, kernel: int = 5):
        super(RayBranch, self).__init__()
        self.in_ch = int(in_ch)
        
        # 1. 维度扩展：从输入通道映射到隐藏层维度
        self.expand = nn.Conv1d(self.in_ch, hidden, kernel_size=1, has_bias=True)
        
        # 2. 构建空洞卷积块
        dilations = [1, 2, 4, 8][:layers]
        blocks = []
        for d in dilations:
            # 依次放入：深度可分离卷积 -> 激活 -> 通道注意力
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
    def __init__(self, vec_dim: int, hidden: int = 64, d_model: int = 128, *, 
                 num_queries: int = 1, num_heads: int = 1, learnable_queries: bool = True):
        super(RayEncoder, self).__init__()
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.learnable_queries = bool(learnable_queries)
        
        # 维度计算
        self.pose_dim = 7
        assert vec_dim >= self.pose_dim, f"vec_dim must be N + {self.pose_dim}, got {vec_dim}"
        self.vec_dim = int(vec_dim)
        self.N = max(0, vec_dim - self.pose_dim)
        self.ray_in_ch = 1
        self.hidden = int(hidden)
        self.d_model = int(d_model)
        assert self.d_model % max(1, self.num_heads) == 0, "d_model must be divisible by num_heads"

        # 1. 射线特征提取分支
        self.br_obs = RayBranch(in_ch=self.ray_in_ch, hidden=hidden)
        self.to_k = nn.Conv1d(hidden, d_model, kernel_size=1, has_bias=True)
        self.to_v = nn.Conv1d(hidden, d_model, kernel_size=1, has_bias=True)

        # 2. 位姿 MLP
        self.pose_mlp = nn.SequentialCell([
            nn.Dense(self.pose_dim, d_model),
            nn.ReLU(),
            nn.Dense(d_model, d_model)
        ])

        # 3. Query 处理
        if self.learnable_queries:
            init_scale = 1.0 / math.sqrt(max(1, d_model))
            self.q_params = Parameter(initializer(Normal(init_scale), [self.num_queries, d_model]), name="q_params")
            self.to_q = nn.Identity()
        else:
            self.to_q = nn.Dense(d_model, d_model * self.num_queries) if self.num_queries > 1 else nn.Identity()

        # 4. 后处理 MLP
        self.post = nn.SequentialCell([
            nn.Dense(d_model * 3, 256), 
            nn.ReLU(),
            nn.Dense(256, 256),
            nn.ReLU()
        ])
        
        # 算子实例化
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def split(self, vec: Tensor):
        d_len = self.N * self.ray_in_ch
        d_obs = vec[:, :d_len]
        pose = vec[:, d_len:d_len + self.pose_dim]
        
        if self.ray_in_ch > 1:
            d_obs = self.reshape(d_obs, (vec.shape[0], self.ray_in_ch, self.N))
        else:
            # 确保即使 ray_in_ch 为 1 也有通道维度 (B, 1, L)
            d_obs = ops.expand_dims(d_obs, 1)
        return d_obs, pose

    def construct(self, vec: Tensor):
        # 数据拆分
        d_obs, pose = self.split(vec)
        
        # 提取 K, V (Conv1d 输出通常是 float16/float32 取决于 AMP)
        f_map = self.br_obs(d_obs)
        k = self.transpose(self.to_k(f_map), (0, 2, 1)) # (B, L, D)
        v = self.transpose(self.to_v(f_map), (0, 2, 1)) # (B, L, D)

        # 生成 Q
        q_pose = self.pose_mlp(pose)
        if self.learnable_queries:
            q = ops.expand_dims(self.q_params, 0) + ops.expand_dims(q_pose, 1)
        else:
            if self.num_queries > 1:
                qm = self.to_q(q_pose)
                q = self.reshape(qm, (qm.shape[0], self.num_queries, self.d_model))
            else:
                q = ops.expand_dims(q_pose, 1)

        # 【核心修改 1】：强制转换为一致的 float32 以兼容 Ascend MatMul
        # 这步防止了 [w]:Tensor[Float16] 和 [x]:Tensor[Float32] 的报错
        k = k.astype(mstype.float32)
        v = v.astype(mstype.float32)
        q = q.astype(mstype.float32)

        # 多头注意力参数
        h = self.num_heads
        dh = self.d_model // h
        b_size = q.shape[0]
        
        # 重塑维度: (B, L, H, Dh)
        k_h = self.reshape(k, (b_size, -1, h, dh))
        v_h = self.reshape(v, (b_size, -1, h, dh))
        q_h = self.reshape(q, (b_size, -1, h, dh))

        # 【核心修改 2】：使用 Transpose + MatMul 替代 Einsum
        # 1. 计算注意力分数
        q_h_t = ops.transpose(q_h, (0, 2, 1, 3))  # (B, H, M, Dh)
        k_h_t = ops.transpose(k_h, (0, 2, 3, 1))  # (B, H, Dh, N)
        
        # MatMul: (B, H, M, Dh) @ (B, H, Dh, N) -> (B, H, M, N)
        attn_logits = ops.matmul(q_h_t, k_h_t) / math.sqrt(dh)
        attn = self.softmax(attn_logits)
        
        # 2. 加权求和
        v_h_t = ops.transpose(v_h, (0, 2, 1, 3))  # (B, H, N, Dh)
        # MatMul: (B, H, M, N) @ (B, H, N, Dh) -> (B, H, M, Dh)
        z_h_t = ops.matmul(attn, v_h_t)          
        
        # 还原形状
        z_h = ops.transpose(z_h_t, (0, 2, 1, 3))  # (B, M, H, Dh)
        z = self.reshape(z_h, (b_size, -1, self.d_model))

        # 特征聚合 (此时 z, q, v 已经是 float32)
        z_mean = ops.reduce_mean(z, 1)
        q_mean = ops.reduce_mean(q, 1)
        gavg = ops.reduce_mean(v, 1)

        # 拼接与输出
        # 再次确保拼接到 post 层前类型正确
        g = ops.concat([z_mean, gavg, q_mean], axis=-1)
        g = self.post(g)
        
        return g, k, v