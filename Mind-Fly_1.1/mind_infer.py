import sys
import os

# 自动修复路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIND_PPO_DIR = os.path.join(BASE_DIR, "mind_ppo")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, MIND_PPO_DIR)

import numpy as np
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Tensor

# 导入你真实的模型
from mind_models import PPOPolicy
from mind_encoder import RayEncoder

# ============================
# 【完全按照你源码的参数】
# ============================
VEC_DIM    = 135        # 雷达128 + 位姿7 = 135（必须这个数！）
ACTION_DIM = 2          # 输出 vx, omega
HIDDEN     = 64
D_MODEL    = 128
NUM_QUERIES = 4
NUM_HEADS  = 4
CKPT_FILE  = os.path.join(BASE_DIR, "runs", "latest_policy.ckpt")

# 动作限制（训练时的限制）
LIMITS = Tensor([1.0, 1.0], dtype=ms.float32)  # [vx, omega]

# ============================
# ✅ 完全复刻训练代码初始化
# ============================
encoder = RayEncoder(
    vec_dim=VEC_DIM,
    hidden=HIDDEN,
    d_model=D_MODEL,
    num_queries=NUM_QUERIES,
    num_heads=NUM_HEADS,
    learnable_queries=True
)

policy = PPOPolicy(
    vec_dim=VEC_DIM,
    action_dim=ACTION_DIM,
    hidden=HIDDEN,
    d_model=D_MODEL,
    num_queries=NUM_QUERIES,
    num_heads=NUM_HEADS,
    learnable_queries=True
)

# 加载权重
param_dict = load_checkpoint(CKPT_FILE)
load_param_into_net(policy, param_dict)
policy.set_train(False)

# ============================
# 推理函数（最终版）
# ============================
def ppo_predict(laser_data):
    """
    输入：激光雷达 128 维数据
    输出：vx, omega
    """
    # 补 7 位姿数据（推理用默认值）
    pose_data = np.zeros(7, dtype=np.float32)
    obs = np.concatenate([laser_data, pose_data], axis=0).astype(np.float32)
    
    # 构造模型输入
    obs_tensor = Tensor(obs[None, :], dtype=ms.float32)
    
    # 推理
    output = policy.act(obs_tensor, limits=LIMITS)
    
    # 解析动作
    action = output.action.asnumpy()[0]
    vx = action[0]
    omega = action[1]
    return vx, omega

# ============================
# 测试
# ============================
if __name__ == "__main__":
    print("   PPO 模型加载成功！")
    
    # 模拟 128 维激光雷达数据
    test_lidar = np.random.rand(128).astype(np.float32)
    
    # 推理
    vx, omega = ppo_predict(test_lidar)
    
    print(f"   推理结果：")
    print(f"   前进速度 vx    = {vx:.3f}")
    print(f"   旋转速度 omega = {omega:.3f}")