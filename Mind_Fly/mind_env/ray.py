from __future__ import annotations

from typing import Tuple, Sequence, Dict, Any
import numpy as np


def raycast(occ: np.ndarray, origin: Tuple[float, float], direction: Tuple[float, float], max_dist: float, step: float = 0.5) -> float:
    """
    二维栅格地图射线投射函数
    功能：在二值占据栅格地图上执行简单的射线投射，检测射线与障碍物的交点距离
    核心逻辑：
    1. 从原点出发，沿指定方向以固定步长采样栅格点
    2. 检查采样点是否超出地图边界或命中障碍物
    3. 返回命中距离（像素单位），未命中则返回最大距离
    输入：
    - occ: H×W的布尔型栅格地图，True表示障碍物
    - origin: 射线起点的像素坐标 (x, y)
    - direction: 归一化的射线方向向量 (dx, dy)
    - max_dist: 射线最大投射距离（像素）
    - step: 沿射线的采样步长（像素），默认0.5
    输出：射线命中距离（像素），未命中返回max_dist
    """
    ox, oy = origin
    dx, dy = direction
    if dx == 0 and dy == 0:
        return 0.0
    H, W = occ.shape
    dist = 0.0
    while dist < max_dist:
        x = int(round(ox + dx * dist))
        y = int(round(oy + dy * dist))
        if y < 0 or y >= H or x < 0 or x >= W:
            return dist
        if occ[y, x]:
            return dist
        dist += step
    return max_dist


def radial_scan(occ: np.ndarray,
                origin: Tuple[float, float],
                n_rays: int = 36,
                max_dist: float = 50.0,
                step: float = 0.25,
                fov_deg: float = 360.0,
                start_angle_deg: float = 0.0) -> np.ndarray:
    """
    径向射线扫描函数
    功能：从原点向指定视野范围内均匀发射多条射线，返回每条射线的障碍物命中距离
    核心逻辑：
    1. 计算视野范围内均匀分布的射线角度
    2. 将角度转换为方向向量
    3. 对每条射线调用raycast函数获取命中距离
    4. 返回所有射线的距离数组
    输入：
    - occ: H×W的布尔型占据栅格地图，True表示障碍物
    - origin: 扫描中心点的像素坐标 (x, y)
    - n_rays: 射线数量，默认36
    - max_dist: 射线最大投射距离（像素），默认50.0
    - step: 沿射线的采样步长（像素），默认0.25
    - fov_deg: 视野范围（角度），默认360°（全向）
    - start_angle_deg: 起始角度（角度），0°朝右，逆时针为正方向
    输出：n_rays长度的浮点数组，存储每条射线的命中距离
    """
    if n_rays <= 0:
        return np.zeros((0,), dtype=np.float32)
    angles = np.deg2rad(start_angle_deg + np.linspace(0.0, max(0.0, fov_deg), num=n_rays, endpoint=False))
    dx = np.cos(angles)
    dy = np.sin(angles)
    dists = np.empty((n_rays,), dtype=np.float32)
    for i in range(n_rays):
        dists[i] = float(raycast(occ, origin, (dx[i], dy[i]), max_dist=max_dist, step=step))
    return dists


def compute_ray_defaults(obs_cfg: Dict[str, Any], patch_meters: float) -> Tuple[int, float, float]:
    """
    射线参数自动计算函数
    功能：根据观测配置和视野距离（米）推导射线扫描的默认参数
    核心逻辑：
    1. 从配置中提取射线步长（米）和最大间隙（米）
    2. 根据视野半径和最大间隙计算所需的射线数量（保证射线覆盖无间隙）
    3. 返回标准化的射线数量、步长和视野半径参数
    输入：
    - obs_cfg: 观测配置字典，包含：
      - ray_max_gap: 射线间最大间隙（米），<=0时返回0条射线
      - ray_step_m: 沿射线的采样步长（米），默认0.025
    - patch_meters: 视野距离/雷达半径（米）
    输出：三元组 (n_rays, ray_step_m, view_radius_m)
      - n_rays: 计算得到的射线数量
      - ray_step_m: 射线采样步长（米）
      - view_radius_m: 视野半径（米）
    """
    step_m = float(obs_cfg.get("ray_step_m", 0.025))
    radius_m = float(patch_meters)
    max_gap = float(obs_cfg.get("ray_max_gap", 0.0))
    if max_gap > 0.0 and radius_m > 0.0:
        n_rays = int(np.ceil((2.0 * np.pi * radius_m) / max(max_gap, 1e-9)))
    else:
        n_rays = 0
    return n_rays, step_m, radius_m