from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import math
import mindspore as ms
from mindspore import ops, Tensor

from .ray import compute_ray_defaults

def _wrap_angle_pi(yaw: Tensor) -> Tensor:
    """
    角度归一化函数
    功能：将任意角度值归一化到 [-π, π] 区间，保证角度计算的连续性
    核心逻辑：通过取模运算将角度映射到 [0, 2π]，再偏移到 [-π, π]
    输入：yaw - 待归一化的角度张量（弧度）
    输出：归一化后的角度张量，范围 [-π, π]
    """
    pi_tensor = ms.Tensor(math.pi, ms.float32)
    two_pi_tensor = ms.Tensor(2.0 * math.pi, ms.float32)
    return ((yaw + pi_tensor) % two_pi_tensor) - pi_tensor


@dataclass
class SimGPUEnvConfig:
    """
    随机射线环境配置类
    功能：封装SimRandomBatchEnv环境的所有配置参数，包含物理参数、奖励系数、射线参数等
    核心参数说明：
    - dt: 仿真时间步长（秒）
    - n_envs: 并行环境数量
    - patch_meters: 视野半径（米）
    - ray相关参数：控制射线数量、步长、间隙等
    - 安全参数：safe_distance_m 安全距离，vx_max/omega_max 速度/角速度上限
    - 奖励系数：w_collision/w_progress等，控制不同奖励项的权重
    - 任务点参数：控制任务点生成、成功判定的距离阈值
    """
    dt: float = 0.1
    n_envs: int = 256
    patch_meters: float = 10.0
    ray_step_m: float = 0.025
    n_rays: int = 0
    ray_max_gap: float = 0.25
    safe_distance_m: float = 0.75
    vx_max: float = 1.5
    omega_max: float = 2.0
    w_collision: float = 1.0
    w_progress: float = 0.01
    w_limits: float = 0.1
    orientation_verify: bool = False
    w_jerk: float = 0.0
    w_jerk_omega: float = 0.0
    reward_time: float = 0.0
    collision_done: bool = False
    blank_ratio_base: float = 40.0
    blank_ratio_randmax: float = 40.0
    blank_ratio_std_ratio: float = 0.33
    narrow_passage_gaussian: bool = False
    narrow_passage_std_ratio: float = 0.3
    device: Optional[str] = None
    task_point_max_dist_m: float = 8.0
    task_point_success_radius_m: float = 0.25
    task_point_random_interval_max: int = 0


class SimRandomBatchEnv:
    """
    批量随机射线环境（适配MindSpore 2.x）
    功能：为PPO算法提供无全局地图/目标的随机射线环境，支持批量并行仿真
    核心逻辑：
    1. 状态管理：跟踪偏航角、线速度、位置、指令历史等状态
    2. 射线生成：随机生成符合空/障比例的射线距离，模拟环境感知
    3. 动作执行：将动作映射到机器人运动，计算碰撞、进度等奖励
    4. 任务点管理：随机生成任务点，判定任务完成并重新采样
    5. 观测构建：融合射线数据、参考特征、指令历史等生成观测向量
    """

    def __init__(self, cfg: SimGPUEnvConfig) -> None:
        self.cfg = cfg
        self.float_type = ms.float32
        self.int_type = ms.int32
        self.bool_type = ms.bool_

        n_rays = int(cfg.n_rays)
        if n_rays <= 0:
            n_rays, _, _ = compute_ray_defaults(
                {"ray_max_gap": float(cfg.ray_max_gap)},
                float(cfg.patch_meters),
            )
        self.n_rays = int(max(0, n_rays))
        self.view_radius_m = float(cfg.patch_meters)
        
        if self.view_radius_m <= 0.0:
            raise ValueError(f"patch_meters must be positive, got {self.view_radius_m}")
        if cfg.vx_max <= 0.0 or cfg.omega_max <= 0.0:
            raise ValueError("velocity limits must be positive")
            
        self.B = int(cfg.n_envs)
        
        self.t = ops.zeros((self.B,), self.int_type)
        self.yaw = ops.zeros((self.B,), self.float_type)
        self.vel_xy = ops.zeros((self.B, 2), self.float_type)
        self.pos_xy = ops.zeros((self.B, 2), self.float_type)
        self.prev_cmd = ops.zeros((self.B, 3), self.float_type)
        self.prev_prev_cmd = ops.zeros((self.B, 3), self.float_type)
        
        if self.n_rays > 0:
            self._ray_ang = ops.arange(self.n_rays, dtype=self.float_type) * (2.0 * math.pi / float(self.n_rays))
        else:
            self._ray_ang = ops.zeros((0,), dtype=self.float_type)
            
        self._rays_m = ops.zeros((self.B, self.n_rays), dtype=self.float_type)
        self._ref_vec = ops.zeros((self.B, 2), dtype=self.float_type)
        self._ref_feat = ops.zeros((self.B, 2), dtype=self.float_type)
        self._global_task_xy = ops.zeros((self.B, 2), dtype=self.float_type)
        self._local_task_xy = ops.zeros((self.B, 2), dtype=self.float_type)
        
        self.interval_max = int(getattr(self.cfg, "task_point_random_interval_max", 0))
        self._task_redraw_counter = ops.zeros((self.B,), dtype=self.int_type)
        
        if self.interval_max > 0:
            self._task_redraw_target = ops.randint(1, self.interval_max + 1, (self.B,), dtype=self.int_type)
        else:
            self._task_redraw_target = ops.zeros((self.B,), dtype=self.int_type)
            
        self._resample_fov_and_ref()
        self._sample_new_global_task_points(mask=ops.ones((self.B,), dtype=self.bool_type))

    def get_limits(self) -> Tensor:
        """
        获取动作限制函数
        功能：返回线速度和角速度的最大值，用于动作缩放和奖励计算
        输出：张量 [vx_max, omega_max]，形状 [2]
        """
        return Tensor([self.cfg.vx_max, self.cfg.omega_max], dtype=self.float_type)

    def reset(self) -> Tensor:
        """
        环境重置函数
        功能：重置所有环境状态（时间、位置、速度、指令历史等），重新采样FOV和任务点
        输出：重置后的初始观测张量，形状 [B, obs_dim]
        """
        self.t = ops.zeros((self.B,), self.int_type)
        self.yaw = ops.zeros((self.B,), self.float_type)
        self.vel_xy = ops.zeros((self.B, 2), self.float_type)
        self.pos_xy = ops.zeros((self.B, 2), self.float_type)
        self.prev_cmd = ops.zeros((self.B, 3), self.float_type)
        self.prev_prev_cmd = ops.zeros((self.B, 3), self.float_type)
        
        self._resample_fov_and_ref()
        self._sample_new_global_task_points(mask=ops.ones((self.B,), dtype=self.bool_type))
        
        if self.interval_max > 0:
            self._task_redraw_counter = ops.zeros((self.B,), dtype=self.int_type)
            self._task_redraw_target = ops.randint(1, self.interval_max + 1, (self.B,), dtype=self.int_type)
        
        return self.observe()

    def observe(self) -> Tensor:
        """
        观测生成函数
        功能：调用_build_obs构建当前状态的观测向量
        输出：观测张量，形状 [B, obs_dim]
        """
        return self._build_obs(self._rays_m, self._ref_feat)

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        """
        环境步进函数
        功能：执行动作，更新环境状态，计算奖励、终止状态和附加信息
        核心逻辑：
        1. 动作裁剪：将输入动作限制在速度/角速度范围内
        2. 运动学更新：根据动作更新机器人位置、速度、偏航角
        3. 碰撞检测：基于射线距离判断是否发生碰撞
        4. 奖励计算：融合进度、碰撞、抖动、速度限制等多维度奖励
        5. 任务点更新：判定任务完成并重新采样任务点
        6. 状态重置：碰撞时重置对应环境状态（若开启collision_done）
        输入：action - 动作张量，形状 [B, 2] 或 [B, 3]
        输出：四元组 (obs_next, rew, term, info)
            - obs_next: 下一步观测 [B, obs_dim]
            - rew: 奖励张量 [B,]
            - term: 终止标志 [B,]
            - info: 附加信息字典，包含碰撞、成功、最小射线距离等
        """
        vx_max = float(self.cfg.vx_max)
        om_max = float(self.cfg.omega_max)
        dt = float(self.cfg.dt)

        vx_cmd = ops.clamp(action[:, 0], -vx_max, vx_max)
        if action.shape[1] == 2:
            vy_cmd = ops.zeros_like(vx_cmd)
            om_cmd = ops.clamp(action[:, 1], -om_max, om_max)
        else:
            om_cmd = ops.clamp(action[:, 2], -om_max, om_max)
            vy_cmd = ops.zeros_like(vx_cmd)

        pos_prev = self.pos_xy.copy()
        dx_local = vx_cmd * dt
        dy_local = vy_cmd * dt
        
        yaw_end = _wrap_angle_pi(self.yaw + om_cmd * dt)
        c1 = ops.cos(yaw_end)
        s1 = ops.sin(yaw_end)
        
        vx_w_end = c1 * vx_cmd - s1 * vy_cmd
        vy_w_end = s1 * vx_cmd + c1 * vy_cmd
        
        self.vel_xy = ops.stack([vx_w_end, vy_w_end], axis=-1)
        self.yaw = yaw_end
        self.t = self.t + 1
        
        if self.interval_max > 0:
            self._task_redraw_counter = self._task_redraw_counter + 1
            
        pos_x_new = self.pos_xy[:, 0] + vx_w_end * dt
        pos_y_new = self.pos_xy[:, 1] + vy_w_end * dt
        self.pos_xy = ops.stack([pos_x_new, pos_y_new], axis=-1)
        
        travel = ops.sqrt(ops.square(dx_local) + ops.square(dy_local))
        ang = ops.atan2(dy_local, dx_local)
        ray_d_along = self._interp_ray_distance(self._rays_m, ang)
        
        pen = travel - ray_d_along
        collided = ops.logical_and((pen > 1e-6), (ray_d_along > 0.0))
        
        d_prev = ops.sqrt(ops.square(self._global_task_xy[:, 0] - pos_prev[:, 0]) + ops.square(self._global_task_xy[:, 1] - pos_prev[:, 1]))
        d_next = ops.sqrt(ops.square(self._global_task_xy[:, 0] - self.pos_xy[:, 0]) + ops.square(self._global_task_xy[:, 1] - self.pos_xy[:, 1]))
        delta_d = d_next - d_prev
        
        denom_progress = vx_max * dt
        jerk_x = (vx_cmd - 2.0 * self.prev_cmd[:, 0] + self.prev_prev_cmd[:, 0]) / vx_max
        jerk_omega = (om_cmd - 2.0 * self.prev_cmd[:, 2] + self.prev_prev_cmd[:, 2]) / om_max
        limit_hit = ops.logical_or((ops.abs(vx_cmd) >= vx_max - 1e-9), (ops.abs(om_cmd) >= om_max - 1e-9))

        min_ray_dist_m = self._rays_m.min(axis=-1)
        task_resampled = ops.zeros((self.B,), dtype=self.bool_type)

        jerk_norm = ops.clamp((jerk_x * jerk_x) / 16.0, 0.0, 1.0)
        jerk_omega_norm = ops.clamp((jerk_omega * jerk_omega) / 16.0, 0.0, 1.0)
        v_lin = ops.sqrt(ops.square(vx_w_end) + ops.square(vy_w_end))
        v_ratio = ops.clamp(v_lin / vx_max, 0.0, 1.0)
        base_progress = (-delta_d) / denom_progress

        if bool(self.cfg.orientation_verify):
            dot_hv = c1 * vx_w_end + s1 * vy_w_end
            cos_heading_vel = ops.where(
                v_lin > 1e-9,
                ops.clamp(dot_hv / v_lin, -1.0, 1.0),
                ops.ones_like(v_lin)
            )
        else:
            cos_heading_vel = ops.ones_like(base_progress)

        progress = ops.where(delta_d > 0.0, -ops.abs(base_progress), base_progress)
        if bool(self.cfg.orientation_verify):
            allow_pos = ops.logical_and(delta_d < 0.0, cos_heading_vel > 0.0)
            progress = ops.where(allow_pos, progress, -ops.abs(progress))

        rew_progress = self.cfg.w_progress * progress
        rew_progress = ops.where(delta_d > 0.0, -ops.abs(rew_progress), rew_progress)

        rew = (
            rew_progress
            - self.cfg.w_collision * (v_ratio * collided.to(self.float_type))
            - self.cfg.w_jerk * jerk_norm
            - self.cfg.w_jerk_omega * jerk_omega_norm
            - self.cfg.w_limits * limit_hit.to(self.float_type)
            - float(getattr(self.cfg, "reward_time", 0.0))
        ).to(self.float_type)

        self.prev_prev_cmd = self.prev_cmd
        zero_hist = ops.zeros_like(vx_cmd)
        self.prev_cmd = ops.stack([vx_cmd, zero_hist, om_cmd], axis=-1)

        if bool(getattr(self.cfg, "collision_done", False)):
            term = collided.to(self.bool_type)
            if bool(term.any()):
                self.t = ops.where(term, ops.zeros_like(self.t), self.t)
                self.yaw = ops.where(term, ops.zeros_like(self.yaw), self.yaw)
                term_2d = term.unsqueeze(-1).broadcast_to(self.vel_xy.shape)
                term_3d = term.unsqueeze(-1).broadcast_to(self.prev_cmd.shape)
                self.vel_xy = ops.where(term_2d, ops.zeros_like(self.vel_xy), self.vel_xy)
                self.pos_xy = ops.where(term_2d, ops.zeros_like(self.pos_xy), self.pos_xy)
                self.prev_cmd = ops.where(term_3d, ops.zeros_like(self.prev_cmd), self.prev_cmd)
                self.prev_prev_cmd = ops.where(term_3d, ops.zeros_like(self.prev_prev_cmd), self.prev_prev_cmd)
                
                self._resample_fov_and_ref()
                self._sample_new_global_task_points(mask=term)
                task_resampled = ops.logical_or(task_resampled, term)
        else:
            term = ops.zeros((self.B,), dtype=self.bool_type)

        u = self.pos_xy - pos_prev
        uu = u[:, 0] * u[:, 0] + u[:, 1] * u[:, 1]
        w0 = self._global_task_xy - pos_prev
        
        move_mask = uu > 0.0
        safe_uu = ops.where(move_mask, uu, ops.ones_like(uu))
        t_proj_raw = (w0[:, 0] * u[:, 0] + w0[:, 1] * u[:, 1]) / safe_uu
        t_proj = ops.where(move_mask, ops.clamp(t_proj_raw, 0.0, 1.0), ops.zeros_like(uu))
        
        nearest_x = pos_prev[:, 0] + t_proj * u[:, 0]
        nearest_y = pos_prev[:, 1] + t_proj * u[:, 1]
        dist2_near = ops.square(nearest_x - self._global_task_xy[:, 0]) + ops.square(nearest_y - self._global_task_xy[:, 1])
        
        r_s = float(self.cfg.task_point_success_radius_m)
        success = dist2_near <= (r_s * r_s)
        
        if bool(success.any()):
            self._sample_new_global_task_points(mask=success)
            task_resampled = ops.logical_or(task_resampled, success)

        self._resample_fov_and_ref()
        obs_next = self._build_obs(self._rays_m, self._ref_feat)
        
        info: Dict[str, Any] = {
            "limits": self.get_limits(),
            "success": success,
            "timeout": ops.zeros((self.B,), dtype=self.bool_type),
            "min_ray_dist_m": min_ray_dist_m,
            "collided": collided,
        }
        return obs_next, rew, term, info

    def _resample_fov_and_ref(self) -> None:
        """
        FOV和参考向量重采样函数
        功能：随机生成符合空/障比例的射线距离，更新局部任务点和参考特征
        核心逻辑：
        1. 基于配置的空白比例生成随机空障掩码
        2. 根据掩码生成射线距离（空白区域为最大距离，障碍区域随机距离）
        3. 支持高斯分布的窄通道模拟
        4. 更新局部任务点和参考特征向量
        """
        if self.n_rays <= 0:
            self._rays_m = ops.zeros_like(self._rays_m)
            self._update_local_task_points()
            self._update_ref_from_local()
            return

        base = float(self.cfg.blank_ratio_base)
        jitter = float(self.cfg.blank_ratio_randmax)
        std_ratio = float(getattr(self.cfg, "blank_ratio_std_ratio", 0.33))
        sigma = max(jitter * std_ratio, 1e-6)
        
        p_empty_raw = ops.standard_normal((self.B,)) * sigma + base
        p_empty = (ops.clamp(p_empty_raw, base, base + jitter) / 100.0).to(self.float_type)
        
        mask_empty = ops.uniform((self.B, self.n_rays), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32)) < p_empty.view(-1, 1)
        
        use_gaussian = bool(getattr(self.cfg, "narrow_passage_gaussian", False))
        if use_gaussian:
            std_ratio = float(getattr(self.cfg, "narrow_passage_std_ratio", 0.3))
            sigma = max(self.view_radius_m * std_ratio, 1e-6)
            dist = ops.abs(ops.standard_normal((self.B, self.n_rays))) * sigma
            dist = ops.clamp(dist, 0.0, self.view_radius_m)
            rays_m = ops.where(mask_empty, ops.full_like(dist, self.view_radius_m), dist)
        else:
            prop = ops.uniform((self.B, self.n_rays), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32))
            rays_m = prop * self.view_radius_m
            rays_m = ops.where(mask_empty, ops.full_like(rays_m, self.view_radius_m), rays_m)
            
        self._rays_m = rays_m.to(self.float_type)
        self._update_local_task_points()
        self._update_ref_from_local()

    def _update_local_task_points(self, mask: Optional[Tensor] = None) -> None:
        """
        局部任务点更新函数
        功能：将全局任务点投影到局部坐标系，考虑射线距离限制
        核心逻辑：
        1. 计算全局任务点相对于机器人的方向和距离
        2. 插值射线距离得到任务点方向的可通行距离
        3. 将任务点投影到可通行范围内，更新局部任务点
        输入：mask - 可选掩码，仅更新掩码为True的环境
        """
        dx = self._global_task_xy[:, 0] - self.pos_xy[:, 0]
        dy = self._global_task_xy[:, 1] - self.pos_xy[:, 1]
        dist_global = ops.sqrt(ops.square(dx) + ops.square(dy))

        nz = dist_global > 0.0
        safe_dist = ops.where(nz, dist_global, ops.ones_like(dist_global))
        dir_x = ops.where(nz, dx / safe_dist, ops.zeros_like(dx))
        dir_y = ops.where(nz, dy / safe_dist, ops.zeros_like(dy))

        ang_world = ops.atan2(dy, dx)
        ang_body = _wrap_angle_pi(ang_world - self.yaw)
        los_dist = ops.clamp(self._interp_ray_distance(self._rays_m, ang_body), min=0.0)
        travel = ops.minimum(dist_global, los_dist)

        new_x = self.pos_xy[:, 0] + travel * dir_x
        new_y = self.pos_xy[:, 1] + travel * dir_y

        if mask is None:
            self._local_task_xy = ops.stack([new_x, new_y], axis=-1)
        else:
            mask_2d = mask.unsqueeze(-1).broadcast_to(self._local_task_xy.shape)
            new_xy = ops.stack([new_x, new_y], axis=-1)
            self._local_task_xy = ops.where(mask_2d, new_xy, self._local_task_xy)

    def _update_ref_from_local(self, mask: Optional[Tensor] = None) -> None:
        """
        参考特征更新函数
        功能：基于局部任务点计算参考特征向量（相对角度的正弦/余弦）
        核心逻辑：
        1. 计算局部任务点相对于机器人的方向向量
        2. 计算方向向量与机器人朝向的夹角
        3. 生成包含夹角正弦/余弦的参考特征
        输入：mask - 可选掩码，仅更新掩码为True的环境
        """
        dx = self._local_task_xy[:, 0] - self.pos_xy[:, 0]
        dy = self._local_task_xy[:, 1] - self.pos_xy[:, 1]
        n = ops.sqrt(ops.square(dx) + ops.square(dy))
        
        hx = ops.cos(self.yaw)
        hy = ops.sin(self.yaw)
        tx = ops.where(n > 1e-9, dx / ops.where(n > 1e-9, n, ops.ones_like(n)), hx)
        ty = ops.where(n > 1e-9, dy / ops.where(n > 1e-9, n, ops.ones_like(n)), hy)

        cos_th = ops.clamp(tx * hx + ty * hy, -1.0, 1.0)
        sin_th = ops.clamp(ty * hx - tx * hy, -1.0, 1.0)
        ref_feat = ops.stack([sin_th, cos_th], axis=-1)
        ref_vec = ops.stack([tx, ty], axis=-1)

        if mask is None:
            self._ref_vec = ref_vec
            self._ref_feat = ref_feat
        else:
            mask_2d = mask.unsqueeze(-1).broadcast_to(self._ref_vec.shape)
            self._ref_vec = ops.where(mask_2d, ref_vec, self._ref_vec)
            self._ref_feat = ops.where(mask_2d, ref_feat, self._ref_feat)

    def _sample_new_global_task_points(self, mask: Tensor) -> None:
        """
        全局任务点采样函数
        功能：在安全区域内随机采样新的全局任务点
        核心逻辑：
        1. 确定任务点的距离范围（成功半径到最大距离）
        2. 选择满足安全距离的射线方向，无安全方向则选最远方向
        3. 在选定方向上随机生成任务点位置
        输入：mask - 掩码，仅为True的环境采样新任务点
        """
        if not bool(mask.any()):
            return
            
        r_min = max(float(self.cfg.task_point_success_radius_m), 0.0)
        r_max = float(min(float(self.cfg.task_point_max_dist_m), float(self.view_radius_m)))
        r_max = max(r_max, r_min)
        
        u = ops.uniform((self.B,), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32))
        dist = r_min + (r_max - r_min) * u
        safe = self._rays_m >= float(self.cfg.safe_distance_m)
        has_any = safe.any(axis=-1)
        
        argmax_idx = self._rays_m.argmax(axis=-1)
        rand_scores = ops.uniform((self.B, self.n_rays), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32))
        scores = ops.where(safe, rand_scores, ops.full_like(rand_scores, float("-inf")))
        pick_any = ops.argmax(scores, dim=-1)
        
        idx = ops.where(has_any, pick_any, argmax_idx)
        if self.n_rays > 0:
            idx = ops.clamp(idx, 0, self.n_rays - 1)
            
        dth = (2.0 * math.pi) / float(max(self.n_rays, 1))
        jitter_local = (ops.uniform((self.B,), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32)) - 0.5) * float(dth)
        
        ang = self._ray_ang[idx]
        th = _wrap_angle_pi(self.yaw + ang + jitter_local)
        
        new_x = self.pos_xy[:, 0] + dist * ops.cos(th)
        new_y = self.pos_xy[:, 1] + dist * ops.sin(th)
        new_xy = ops.stack([new_x, new_y], axis=-1)
        
        mask_2d = mask.unsqueeze(-1).broadcast_to(self._global_task_xy.shape)
        self._global_task_xy = ops.where(mask_2d, new_xy, self._global_task_xy)

        self._update_local_task_points(mask=mask)
        self._update_ref_from_local(mask=mask)

    def _build_obs(self, rays_m: Tensor, ref_feat: Tensor) -> Tensor:
        """
        观测构建核心函数
        功能：融合射线数据、参考特征、指令历史等生成最终观测向量
        核心逻辑：
        1. 归一化射线距离、指令历史、距离等特征
        2. 拼接所有特征：射线数据 + 参考特征 + 前一指令 + 指令变化 + 归一化距离
        输入：
        - rays_m: 射线距离张量 [B, n_rays]
        - ref_feat: 参考特征张量 [B, 2]
        输出：观测张量 [B, n_rays + 7]（无射线时为7维）
        """
        vx_lim = float(self.cfg.vx_max)
        om_lim = float(self.cfg.omega_max)
        prev_vx_n = self.prev_cmd[:, 0] / vx_lim
        prev_om_n = self.prev_cmd[:, 2] / om_lim
        prev_cmd_n = ops.stack([prev_vx_n, prev_om_n], axis=-1)
        
        dvx_n = (self.prev_cmd[:, 0] - self.prev_prev_cmd[:, 0]) / (2.0 * vx_lim)
        dom_n = (self.prev_cmd[:, 2] - self.prev_prev_cmd[:, 2]) / (2.0 * om_lim)
        dprev_all_n = ops.stack([dvx_n, dom_n], axis=-1)
        
        dist = ops.sqrt(ops.square(self._local_task_xy[:, 0] - self.pos_xy[:, 0]) + ops.square(self._local_task_xy[:, 1] - self.pos_xy[:, 1]))
        dist_n = ops.clamp(dist / self.view_radius_m, 0.0, 1.0).unsqueeze(-1)

        if self.n_rays <= 0:
            return ops.concat([ref_feat, prev_cmd_n, dprev_all_n, dist_n], axis=-1).to(self.float_type)

        rays_n = ops.clamp(rays_m / self.view_radius_m, 0.0, 1.0)
        parts = [rays_n, ref_feat, prev_cmd_n, dprev_all_n, dist_n]
        return ops.concat(parts, axis=-1).to(self.float_type)

    def _interp_ray_distance(self, rays_m: Tensor, angle: Tensor) -> Tensor:
        """
        射线距离插值函数
        功能：根据给定角度插值得到对应方向的射线距离
        核心逻辑：
        1. 将角度映射到射线索引范围
        2. 线性插值相邻射线的距离，得到指定角度的距离
        输入：
        - rays_m: 射线距离张量 [B, n_rays]
        - angle: 目标角度张量 [B,]
        输出：插值后的距离张量 [B,]
        """
        if self.n_rays <= 0:
            return ops.full((self.B,), float("inf"), dtype=self.float_type)
            
        R = float(self.n_rays)
        dth = (2.0 * math.pi) / R

        a = angle % (2.0 * math.pi)
        f = a / dth
        i0 = ops.floor(f).to(self.int_type)
        t = f - i0.to(self.float_type)
        
        if self.n_rays > 0:
            i0 = ops.clamp(i0, 0, self.n_rays - 1)
            i1 = (i0 + 1) % int(self.n_rays)
        else:
            i1 = i0
            
        ar = ops.arange(self.B, dtype=self.int_type)
        d0 = rays_m[ar, i0]
        d1 = rays_m[ar, i1]
        return d0 * (1.0 - t) + d1 * t

def infer_obs_dim(cfg) -> int:
    """
    观测维度推断函数
    功能：根据环境配置自动计算观测向量的维度
    核心逻辑：无射线时固定7维，有射线时为射线数量+7维
    输入：cfg - 环境配置对象
    输出：观测维度（整数）
    """
    if int(cfg.n_rays) <= 0:
        return 7
    return int(cfg.n_rays) + 7
    
if __name__ == "__main__":
    cfg = SimGPUEnvConfig(n_envs=4)
    env = SimRandomBatchEnv(cfg)
    
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    test_action = ms.ops.standard_normal((4, 3))
    obs_next, reward, terminated, info = env.step(test_action)
    
    print(f"Reward sample: {reward}")
    print(f"Collided mask: {info['collided']}")
    print("Successfully ran one step!")