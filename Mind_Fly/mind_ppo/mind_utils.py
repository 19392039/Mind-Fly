from __future__ import annotations

import os
import sys
from typing import Optional
import numpy as np

def infer_vec_dim(env_cfg_path: str, mission_cfg_path: Optional[str]) -> int:
    """
    自动推断环境向量观测维度
    功能：通过启动MissionPlanner创建环境实例，重置环境获取观测，计算展平后的向量观测维度
    核心逻辑：
    1. 处理MissionPlanner模块的导入路径，确保模块可访问
    2. 初始化MissionPlanner并配置观测模式为向量模式
    3. 创建环境实例，重置环境获取初始观测
    4. 将观测展平为一维数组，返回其长度作为向量维度
    输入：
    - env_cfg_path: 环境配置文件路径
    - mission_cfg_path: 任务配置文件路径（可选）
    输出：
    - vec_dim: 展平后的向量观测维度（整数）
    """
    try:
        from mission import MissionPlanner
    except Exception:
        _MIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agent interface'))
        if os.path.isdir(_MIS_DIR) and _MIS_DIR not in sys.path:
            sys.path.append(_MIS_DIR)
        from mission import MissionPlanner

    mpn = MissionPlanner(env_cfg_path, mission_cfg_path, generator_module=None, generator_kwargs=None)
    try:
        obs_cfg = mpn.env_cfg.get("obs", {}) or {}
        obs_cfg["mode"] = "vector"
        if "rays_align_yaw" not in obs_cfg:
            obs_cfg["rays_align_yaw"] = True
        mpn.env_cfg["obs"] = obs_cfg
    except Exception:
        pass
    env = mpn.respawn_env(current_env=None, rotate_map=True, seed=None)
    obs = env.reset()
    vec_dim = int(np.asarray(obs).reshape(-1).shape[0])
    return vec_dim