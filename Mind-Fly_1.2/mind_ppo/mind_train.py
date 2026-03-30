from __future__ import annotations

import os
import time
import argparse
import random
import numpy as np
from typing import Any, Dict, Optional
import sys

# 环境路径配置
current_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_script_dir)
sys.path.insert(0, root_dir)

import mindspore as ms
from mindspore import Tensor
from mindspore import nn, ops, context, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype
from tensorboardX import SummaryWriter

from mind_models import PPOPolicy
from mind_buffer import RolloutBuffer
from mind_env import load_json_config
from mind_env.sim_mind_env import SimGPUEnvConfig, SimRandomBatchEnv, infer_obs_dim as _infer_obs_dim_sim


target = "Ascend" if ms.get_context("device_target") == "Ascend" else "GPU"
ms.set_device(target)

def load_train_config(path: Optional[str]) -> Dict[str, Any]:
    """加载并初始化训练配置文件，设置各模块默认参数"""
    cfg = load_json_config(path) if path else {}
    cfg.setdefault("device", "Ascend")
    cfg.setdefault("env_config", "env_config.json")
    cfg.setdefault("mission_config", None)
    
    samp = cfg.setdefault("sampling", {})
    samp.setdefault("batch_env", 256)
    samp.setdefault("rollout_len", 128)
    samp.setdefault("reset_each_rollout", True)
    
    ppo = cfg.setdefault("ppo", {})
    ppo.setdefault("gamma", 0.99)
    ppo.setdefault("gae_lambda", 0.95)
    ppo.setdefault("clip_range", 0.2)
    ppo.setdefault("lr", 3e-4)
    ppo.setdefault("value_lr", 3e-4)
    ppo.setdefault("entropy_coef", 0.0)
    ppo.setdefault("value_coef", 0.5)
    ppo.setdefault("max_grad_norm", 0.5)
    ppo.setdefault("epochs", 4)
    ppo.setdefault("minibatch_size", 2048)
    ppo.setdefault("amp", False)
    ppo.setdefault("amp_bf16", True)
    ppo.setdefault("bootstrap", True)
    ppo.setdefault("log_std_min", -5.0)
    ppo.setdefault("log_std_max", 2.0)
    ppo.setdefault("collision_done", True)
    
    model = cfg.setdefault("model", {})
    model.setdefault("num_queries", 4)
    model.setdefault("num_heads", 4)

    run = cfg.setdefault("run", {})
    run.setdefault("total_env_steps", 2_000_000)
    run.setdefault("ckpt_dir", "runs/ppo_exp1")
    run.setdefault("log_interval", 20000)
    run.setdefault("eval_every", 100000)
    return cfg

def _extract_seed_from_train_or_env(train_cfg: Dict[str, Any], env_cfg: Dict[str, Any]) -> Optional[int]:
    """从训练配置或环境配置中提取随机种子"""
    try:
        run = train_cfg.get("run", {}) or {}
        seed_v = run.get("seed", None)
        if seed_v in (None, "", "null"):
            seed_v = None
        if seed_v is not None:
            return int(seed_v)
        sim = env_cfg.get("sim", {}) or {}
        s2 = sim.get("seed", None)
        return None if s2 in (None, "", "null") else int(s2)
    except Exception:
        return None

class PPOTrainStep(nn.Cell):
    """
    PPO 单步训练封装类
    处理前向损失计算、反向传播及梯度优化，支持静态图加速
    """
    def __init__(self, policy, optimizer, clip_eps, ent_coef, vf_coef, max_grad_norm, use_amp, weights):
        super().__init__()
        self.policy = policy
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.weights = weights
        # 定义计算梯度和损失的函数
        self.grad_fn = ops.value_and_grad(self.forward_fn, None, self.weights, has_aux=True)

    def forward_fn(self, obs, actions, old_logp, advantages, returns, limits):
        """计算 PPO 核心损失函数（策略损失、价值损失、熵损失）"""
        new_logp, ent, v_pred = self.policy.evaluate_actions(obs, actions, limits)
        
        # 策略比率计算与裁剪
        ratio = ops.exp(new_logp - old_logp)
        ratio = ops.clamp(ratio, 0.0, 10.0)
        
        surr1 = ratio * advantages
        surr2 = ops.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        
        # 三项损失
        pg_loss = -ops.reduce_mean(ops.minimum(surr1, surr2))
        v_loss = 0.5 * ops.reduce_mean(ops.pow(returns - v_pred, 2))
        ent_bonus = ops.reduce_mean(ent)
        
        total_loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent_bonus
        approx_kl = ops.reduce_mean(old_logp - new_logp)
        
        return total_loss, (pg_loss, v_loss, ent_bonus, approx_kl)

    def construct(self, obs, actions, old_logp, advantages, returns, limits):
        """训练步骤的主执行流程"""
        (total_loss, aux_infos), grads = self.grad_fn(obs, actions, old_logp, advantages, returns, limits)
        pg_loss, v_loss, ent_bonus, approx_kl = aux_infos
        
        # 全局梯度裁剪与参数更新
        grads = ops.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer(grads)
        
        return pg_loss, v_loss, ent_bonus, approx_kl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=os.path.join("config", "train_config.json"))
    args = parser.parse_args()

    # 1. 加载训练与环境配置
    cfg = load_train_config(args.train_config)
    cfg_dir = os.path.dirname(os.path.abspath(args.train_config))
    env_cfg_path = cfg.get("env_config", None)
    if env_cfg_path and not os.path.isabs(env_cfg_path):
        env_cfg_path = os.path.join(cfg_dir, env_cfg_path)

    # 2. MindSpore 上下文设置
    device_target = cfg.get("device", "Ascend")
    context.set_context(device_target=device_target)
    if device_target == "Ascend":
        context.set_context(enable_graph_kernel=False)

    env_cfg = load_json_config(env_cfg_path) if env_cfg_path else {}
    obs_cfg = env_cfg.get('obs', {}) or {}
    sim_cfg = env_cfg.get('sim', {}) or {}
    lim_cfg = env_cfg.get('limits', {}) or {}
    rew_cfg = env_cfg.get('reward', {}) or {}
    ppo_cfg = cfg.get('ppo', {}) or {}

    # 3. 设置随机种子
    B_env = int((cfg.get('sampling', {}) or {}).get('batch_env', 256))
    T_roll = int((cfg.get('sampling', {}) or {}).get('rollout_len', 128))
    reset_each_rollout = bool((cfg.get('sampling', {}) or {}).get('reset_each_rollout', True))
    run_seed = _extract_seed_from_train_or_env(cfg, env_cfg)
    if run_seed is not None:
        ms.set_seed(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)

    # 4. 初始化仿真环境
    safe_dist = float(sim_cfg.get('safe_distance', sim_cfg.get('warning_distance', 0.5)))
    sim = SimGPUEnvConfig(
        dt=float(sim_cfg.get('dt', 0.1)),
        n_envs=B_env,
        patch_meters=float(obs_cfg.get('patch_meters', 10.0)),
        ray_step_m=float(obs_cfg.get('ray_step_m', 0.025)),
        n_rays=int(obs_cfg.get('n_rays', 0)),
        ray_max_gap=float(obs_cfg.get('ray_max_gap', 0.25)),
        safe_distance_m=safe_dist,
        vx_max=float(lim_cfg.get('vx_max', 1.5)),
        omega_max=float(lim_cfg.get('omega_max', 2.0)),
        w_collision=float(rew_cfg.get('reward_collision', 1.0)),
        w_progress=float(rew_cfg.get('reward_progress', 0.01)),
        w_limits=float(rew_cfg.get('reward_limits', 0.1)),
        orientation_verify=bool(rew_cfg.get('orientation_verify', False)),
        w_jerk=float(rew_cfg.get('reward_jerk', 0.0)),
        w_jerk_omega=float(rew_cfg.get('reward_jerk_omega', 0.0)),
        reward_time=float(rew_cfg.get('reward_time', 0.0)),
        blank_ratio_base=float((obs_cfg.get('blank_ratio_base', 40.0))),
        blank_ratio_randmax=float((obs_cfg.get('blank_ratio_randmax', 40.0))),
        blank_ratio_std_ratio=float(obs_cfg.get('blank_ratio_std_ratio', 0.33)),
        narrow_passage_gaussian=bool(obs_cfg.get('narrow_passage_gaussian', False)),
        narrow_passage_std_ratio=float(obs_cfg.get('narrow_passage_std_ratio', 0.3)),
        device=device_target,
        task_point_max_dist_m=float(sim_cfg.get('task_point_max_dist_m', 8.0)),
        task_point_success_radius_m=float(sim_cfg.get('task_point_success_radius_m', 0.25)),
        task_point_random_interval_max=int(sim_cfg.get('task_point_random_interval_max', 0)),
        collision_done=bool(ppo_cfg.get('collision_done', True)),
    )
    env = SimRandomBatchEnv(sim)
    obs = env.reset()
    vec_dim = int(obs.shape[1]) if len(obs.shape) == 2 else int(_infer_obs_dim_sim(sim))
    act_dim = 2

    # 5. 初始化 PPO 策略网络
    model_cfg = cfg.get('model', {}) or {}
    policy = PPOPolicy(
        vec_dim=vec_dim,
        action_dim=act_dim,
        num_queries=int(model_cfg.get('num_queries', 4)),
        num_heads=int(model_cfg.get('num_heads', 4)),
        log_std_min=float((cfg.get('ppo', {}) or {}).get('log_std_min', -5.0)),
        log_std_max=float((cfg.get('ppo', {}) or {}).get('log_std_max', 2.0)),
    )

    # 6. 配置优化器
    ppo_cfg = cfg.get('ppo', {}) or {}
    lr_pi = float(ppo_cfg.get('lr', 3e-4))
    lr_vf = float(ppo_cfg.get('value_lr', lr_pi))
    
    pi_params = list(policy.encoder.get_parameters()) + list(policy.mu_head.get_parameters()) + [policy.log_std]
    vf_params = list(policy.value_head.get_parameters())
    all_params = pi_params + vf_params
    
    optimizer = nn.Adam(
        params=[
            {"params": pi_params, "lr": lr_pi},
            {"params": vf_params, "lr": lr_vf}
        ],
        weight_decay=1e-5
    )

    # 7. 训练设置与输出路径
    use_amp = False
    ckpt_dir = cfg['run']['ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    global_step = 0
    last_log = 0
    
    print("[PPO] 将从随机初始化开始全新训练。")

    # 8. 初始化训练步 Cell
    train_step = PPOTrainStep(
        policy=policy,
        optimizer=optimizer,
        clip_eps=float(ppo_cfg.get('clip_range', 0.2)),
        ent_coef=float(ppo_cfg.get('entropy_coef', 0.0)),
        vf_coef=float(ppo_cfg.get('value_coef', 0.5)),
        max_grad_norm=float(ppo_cfg.get('max_grad_norm', 0.5)),
        use_amp=use_amp,
        weights=all_params
    )
    train_step.set_train()

    total_env_steps = int(cfg['run']['total_env_steps'])
    if bool((cfg.get('run', {}) or {}).get('resume_as_additional', False)) and global_step > 0:
        total_env_steps = global_step + total_env_steps
        print(f"[PPO] 累计训练步数: {total_env_steps:,} (已加载 {global_step:,} + 新增 {cfg['run']['total_env_steps']:,})")

    # 9. 监控指标与日志初始化
    t0 = time.time()
    log_interval = int(cfg['run']['log_interval'])
    ep_rew_acc = ms.Tensor(np.zeros(B_env), dtype=mstype.float32)
    ep_len_acc = ms.Tensor(np.zeros(B_env), dtype=mstype.int32)
    finished_rewards = []
    finished_lengths = []

    tb_log_dir = ckpt_dir
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=ckpt_dir)

    vx_max = float(lim_cfg.get('vx_max', 1.5))
    omega_max = float(lim_cfg.get('omega_max', 2.0))
    base_limits_ms = ms.Tensor(np.array([vx_max, omega_max], dtype=np.float32), mstype.float32)
    current_limits = ops.broadcast_to(base_limits_ms.expand_dims(0), (B_env, 2))

    # 10. 主训练循环
    while global_step < total_env_steps:
        if reset_each_rollout:
            obs = env.reset()
        
        buf = RolloutBuffer(T_roll, B_env, obs_dim=vec_dim, act_dim=act_dim)
        
        policy_loss_vals = []
        value_loss_vals = []
        entropy_vals = []
        approx_kl_vals = []

        # 轨迹收集 
        for _ in range(T_roll):
            out = policy.act(obs, current_limits)
            logp = out.logp
            act = out.action
            _, _, v = policy._core(obs)
            
            next_obs, reward_t, done_t, info = env.step(act)
            d = done_t.astype(mstype.float32)
            
            buf.add(
                obs=obs,
                act=act,
                logp=logp,
                rew=reward_t,
                done=d,
                val=v,
                limits = env.get_limits().expand_dims(0).repeat((B_env, 1))
            )

            # 统计 Episode 数据
            rt = reward_t.view(-1)
            done_mask = done_t.view(-1).astype(mstype.bool_)
            ep_rew_acc = ops.where(done_mask, ep_rew_acc + rt, ms.Tensor(0.0, mstype.float32))
            ep_len_acc = ops.where(done_mask, ep_len_acc + 1, ms.Tensor(0, mstype.int32))
            
            if done_mask.any():
                finished_rewards.extend(ep_rew_acc[done_mask].asnumpy().tolist())
                finished_lengths.extend(ep_len_acc[done_mask].asnumpy().tolist())

            global_step += B_env
            obs = next_obs

        # 10.2 优势函数计算
        if bool(ppo_cfg.get('bootstrap', True)):
            last_v = policy._core(obs)[2].view(B_env, 1)
            buf.compute_gae(last_v, gamma=float(ppo_cfg.get('gamma', 0.99)), lam=float(ppo_cfg.get('gae_lambda', 0.95)))
        else:
            buf.compute_mc_returns(gamma=float(ppo_cfg.get('gamma', 0.99)))

        # 10.3 策略优化迭代 (Update Phase)
        epochs = int(ppo_cfg.get('epochs', 4))
        mb_size = int(ppo_cfg.get('minibatch_size', 2048))
        for _ in range(epochs):
            for mb in buf.minibatches(mb_size):
                pg_loss, v_loss, ent_bonus, approx_kl = train_step(
                    mb.obs, mb.actions, mb.logp, mb.advantages, mb.returns, mb.limits
                )
                
                policy_loss_vals.append(float(pg_loss.asnumpy()))
                value_loss_vals.append(float(v_loss.asnumpy()))
                entropy_vals.append(float(ent_bonus.asnumpy()))
                approx_kl_vals.append(float(approx_kl.asnumpy()))

        # 10.4 日志记录与模型保存
        if global_step - last_log >= log_interval:
            elapsed = time.time() - t0
            fps = global_step / max(1e-3, elapsed)
            
            # 计算指标 
            avg_policy_loss = float(np.mean(policy_loss_vals)) if policy_loss_vals else 0.0
            avg_value_loss = float(np.mean(value_loss_vals)) if value_loss_vals else 0.0
            avg_entropy = float(np.mean(entropy_vals)) if entropy_vals else 0.0
            avg_approx_kl = float(np.mean(approx_kl_vals)) if approx_kl_vals else 0.0
            
            print(f"[PPO] step={global_step:,} | fps={fps:.1f} | policy_loss={avg_policy_loss:.4f}")

            writer.add_scalar("loss/policy", avg_policy_loss, global_step)
            writer.add_scalar("loss/value", avg_value_loss, global_step)
            writer.add_scalar("metric/entropy", avg_entropy, global_step)
            writer.add_scalar("metric/approx_kl", avg_approx_kl, global_step)
            writer.add_scalar("speed/fps", fps, global_step)

            if finished_rewards:
                avg_ep_rew = float(np.mean(finished_rewards))
                writer.add_scalar("episode/avg_reward", avg_ep_rew, global_step)
                print(f"[PPO] avg_ep_reward = {avg_ep_rew:.2f}")

            writer.flush()
            
            # 保存模型
            ms.save_checkpoint(policy, os.path.join(ckpt_dir, 'latest_policy.ckpt'))
            
            # 重置缓存
            last_log = global_step
            finished_rewards.clear()
            finished_lengths.clear()
            policy_loss_vals.clear()
            value_loss_vals.clear()
            entropy_vals.clear()
            approx_kl_vals.clear()

    # 11. 保存最终模型
    save_checkpoint({
        'policy': policy,
        'optimizer': optimizer,
        'global_step': global_step
    }, os.path.join(ckpt_dir, 'final.ckpt'))

    write.close()

if __name__ == "__main__":
    main()