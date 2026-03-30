from __future__ import annotations

"""
Key points:
- No global map or goal; each step samples per-ray FOV distances according to the empty/obstacle ratio.
- Only yaw, world-frame linear velocity, and two-step command history are tracked for rewards; position is not tracked.
- The reference direction is randomly chosen from sectors satisfying the safety distance; if none exist, use the farthest sector and uniformly sample within its width.
- Episodes never terminate (done is always False) to simplify continuous control training.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import math
import mindspore as ms
from mindspore import ops, Tensor

# 假设 compute_ray_defaults 已经迁移为返回普通 Python 数值或 MindSpore Tensors
from .ray import compute_ray_defaults

def _wrap_angle_pi(yaw: Tensor) -> Tensor:
    pi_tensor = ms.Tensor(math.pi, ms.float32)
    two_pi_tensor = ms.Tensor(2.0 * math.pi, ms.float32)
    return ((yaw + pi_tensor) % two_pi_tensor) - pi_tensor


@dataclass
class SimGPUEnvConfig:
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
    device: Optional[str] = None  # 在 MindSpore 中通常通过 ms.set_context() 全局设置
    task_point_max_dist_m: float = 8.0
    task_point_success_radius_m: float = 0.25
    task_point_random_interval_max: int = 0


class SimRandomBatchEnv:
    """Batch randomized ray environment with PPO-compatible observations and rewards.
    Adapted for MindSpore 2.x.
    """

    def __init__(self, cfg: SimGPUEnvConfig) -> None:
        self.cfg = cfg
        # MindSpore 中，计算设备通常通过全局 context 控制（如 Ascend, GPU, CPU）
        # 这里统一指定张量创建时的数据类型
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
        
        # 初始化张量状态 (在动态图模式下，我们直接使用 Tensor 并覆盖它们以更新状态)
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
        return Tensor([self.cfg.vx_max, self.cfg.omega_max], dtype=self.float_type)

    def reset(self) -> Tensor:
        # 避免就地修改 `.zero_()`，采用重新赋值
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
        return self._build_obs(self._rays_m, self._ref_feat)

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
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

        pos_prev = self.pos_xy.copy() # 代替 PyTorch 的 .clone()
        dx_local = vx_cmd * dt
        dy_local = vy_cmd * dt
        
        yaw_end = _wrap_angle_pi(self.yaw + om_cmd * dt)
        c1 = ops.cos(yaw_end)
        s1 = ops.sin(yaw_end)
        
        vx_w_end = c1 * vx_cmd - s1 * vy_cmd
        vy_w_end = s1 * vx_cmd + c1 * vy_cmd
        
        # 为了兼容 MindSpore 静态图机制，建议使用 stack 或 concat 组合状态，而非切片赋值
        self.vel_xy = ops.stack([vx_w_end, vy_w_end], axis=-1)
        self.yaw = yaw_end
        self.t = self.t + 1
        
        if self.interval_max > 0:
            self._task_redraw_counter = self._task_redraw_counter + 1
            
        pos_x_new = self.pos_xy[:, 0] + vx_w_end * dt
        pos_y_new = self.pos_xy[:, 1] + vy_w_end * dt
        self.pos_xy = ops.stack([pos_x_new, pos_y_new], axis=-1)
        
        # 兼容性处理：使用 sqrt(x^2 + y^2) 替代部分硬件可能不支持的 hypot
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

        # 碰撞终止与状态重置 (Mask 更新)
        if bool(getattr(self.cfg, "collision_done", False)):
            term = collided.to(self.bool_type)
            if bool(term.any()):
                # 在算子级别更新避免 in-place
                self.t = ops.where(term, ops.zeros_like(self.t), self.t)
                self.yaw = ops.where(term, ops.zeros_like(self.yaw), self.yaw)
                # 使用多维 where 广播
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

        # 成功抵达判定
        u = self.pos_xy - pos_prev
        uu = u[:, 0] * u[:, 0] + u[:, 1] * u[:, 1]
        w0 = self._global_task_xy - pos_prev
        
        move_mask = uu > 0.0
        safe_uu = ops.where(move_mask, uu, ops.ones_like(uu)) # 避免除 0 异常
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
        if self.n_rays <= 0:
            self._rays_m = ops.zeros_like(self._rays_m)
            self._update_local_task_points()
            self._update_ref_from_local()
            return

        base = float(self.cfg.blank_ratio_base)
        jitter = float(self.cfg.blank_ratio_randmax)
        std_ratio = float(getattr(self.cfg, "blank_ratio_std_ratio", 0.33))
        sigma = max(jitter * std_ratio, 1e-6)
        
        # 使用 standard_normal 因为部分后端 randn 支持存在细微差别
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
        pick_any = ops.argmax(scores, dim=-1) # 注意 MindSpore 中的 dim=-1 可等价于 axis=-1
        
        idx = ops.where(has_any, pick_any, argmax_idx)
        if self.n_rays > 0:
            idx = ops.clamp(idx, 0, self.n_rays - 1)
            
        dth = (2.0 * math.pi) / float(max(self.n_rays, 1))
        jitter_local = (ops.uniform((self.B,), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32)) - 0.5) * float(dth)
        
        # 兼容高级索引
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
        # 高级索引取值：注意在纯图模式（Graph Mode）下，这里如果是动态 shape 可能会有局限
        d0 = rays_m[ar, i0]
        d1 = rays_m[ar, i1]
        return d0 * (1.0 - t) + d1 * t

def infer_obs_dim(cfg) -> int:
    """独立于类之外的维度推断函数"""
    if int(cfg.n_rays) <= 0:
        return 7
    return int(cfg.n_rays) + 7
    
if __name__ == "__main__":
    # 1. 初始化配置和环境
    cfg = SimGPUEnvConfig(n_envs=4) # 先用4个环境测一下
    env = SimRandomBatchEnv(cfg)
    
    # 2. 测试 Reset
    obs = env.reset()
    print(f"Observation shape: {obs.shape}") # 预期应该是 [4, n_rays + 7]
    
    # 3. 测试 Step (模拟随机动作)
    # 动作维度 [B, 3] -> (vx, vy, omega)
    test_action = ms.ops.standard_normal((4, 3)) 
    obs_next, reward, terminated, info = env.step(test_action)
    
    print(f"Reward sample: {reward}")
    print(f"Collided mask: {info['collided']}")
    print("Successfully ran one step!")
