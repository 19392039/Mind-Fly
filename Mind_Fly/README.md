# GRALP-MindSpore:基于 MindSpore 2.8 的全流程开发

# GRALP-MindSpore

GRALP-MindSpore 是基于**昇思MindSpore**框架实现的轻量级PPO局部规划器，面向完全随机化、无地图的自主运动场景，通过向量化的广义射线/深度离障距离+运动学历史作为观测，输出连续的平面速度控制指令。本项目全流程基于MindSpore 2.x开发，适配Ascend/GPU/CPU硬件平台，支持批量仿真训练、混合精度加速，并提供轻量化的推理部署能力，可直接对接实际运动控制设备。

## 快速开始

### 1. 安装依赖

```Bash

# 基础依赖：numpy + onnx + onnxruntime（推理）
pip install -r requirements.txt
# 安装昇思MindSpore（适配硬件平台，建议>=2.0）
# Ascend平台：https://www.mindspore.cn/install
# GPU平台：pip install mindspore-gpu==2.3.0
# CPU平台：pip install mindspore==2.3.0
# 可选：matplotlib + scipy（用于射线分布分析）
pip install matplotlib scipy
```

- 训练依赖：MindSpore核心库（需匹配硬件CUDA/Ascend驱动）；

- 推理依赖：onnx + onnxruntime（CPU/GPU），无需MindSpore依赖；

- 硬件适配：优先支持Ascend NPU，兼顾GPU/CPU训练与推理。

### 2. 配置文件

核心配置文件位于`config/`目录，分为环境配置和训练配置，参数含义清晰，无需额外适配原生框架，直接面向MindSpore调优。

#### `config/env_config.json`（环境&仿真配置）

```JSON

{
  "limits": {"vx_max": 1.5, "omega_max": 2.0},  // 线速度/角速度物理限制
  "sim": {"dt": 0.1, "safe_distance": 0.75, "task_point_max_dist_m": 8.0},  // 仿真步长&任务点参数
  "obs": {"patch_meters": 10.0, "ray_max_gap": 0.25, "blank_ratio_base": 40.0},  // 射线观测参数
  "reward": {"w_collision": 1.0, "w_progress": 0.01, "orientation_verify": false}  // 奖励权重
}
```

- `limits`：被控设备的速度物理上限，直接匹配实际硬件；

- `sim`：仿真核心参数，控制任务点生成、安全距离、仿真时间步；

- `obs`：射线观测配置，自动推导射线数量，控制空/障比例随机化；

- `reward`：多维度奖励权重，平衡碰撞惩罚、任务进度奖励等。

#### `config/train_config.json`（训练&模型配置）

```JSON

{
  "device": "Ascend",  // 训练设备：Ascend/GPU/CPU
  "sampling": {"batch_env": 256, "rollout_len": 128},  // 批量环境&轨迹长度
  "ppo": {"gamma": 0.99, "lr": 3e-4, "epochs": 4, "clip_range": 0.2},  // PPO超参数
  "model": {"num_queries": 4, "num_heads": 4},  // 注意力网络参数
  "run": {"total_env_steps": 2000000, "ckpt_dir": "runs/ppo_mind", "log_interval": 20000}  // 训练运行参数
}
```

- `device`：指定训练硬件，MindSpore自动完成算子适配；

- `sampling`：批量仿真参数，根据硬件算力调整`batch_env`；

- `ppo`：PPO算法核心超参数，包含折扣系数、学习率、裁剪范围等；

- `model`：Ray-Attention注意力网络参数，控制网络规模与推理效率；

- `run`：训练流程参数，指定总训练步数、检查点保存路径、日志打印间隔。

### 3. 开始训练

直接执行训练入口脚本，MindSpore自动完成静态图编译、硬件适配、混合精度加速，无需额外配置：

```Bash

python -m rl_ppo.ppo_train --train_config config/train_config.json
```

- 训练启动后，自动创建`ckpt_dir`指定的检查点目录，实时保存最新模型；

- 训练过程中打印关键指标：训练步数、FPS、策略损失、价值损失、平均奖励、碰撞率等；

- 支持断点续训：停止训练后，重新执行脚本将自动从`ckpt_dir`下的最新检查点恢复训练。

## 核心特性

### 1. 全MindSpore原生实现

- 基于MindSpore静态图模式开发，兼顾训练效率与推理部署；

- 适配MindSpore原生算子、混合精度、梯度裁剪、分布式训练（可扩展）；

- 所有模块（环境、网络、缓冲区、训练器）均基于MindSpore API实现，无第三方框架依赖。

### 2. 随机化无地图仿真环境

- **批量并行仿真**：基于MindSpore张量操作实现批量环境交互，大幅提升训练效率；

- **每步射线重采样**：随机生成空/障分布的射线观测，模拟真实环境的不确定性，无需预先生成地图；

- **无全局地图任务点**：任务点在视野范围内随机生成，按射线视距裁剪，贴合实际无地图局部规划场景；

- **多维度奖励机制**：包含任务进度、碰撞惩罚、速度饱和惩罚、加加速度惩罚，引导模型学习安全高效的运动策略。

### 3. 轻量级Ray-Attention网络

网络主干基于**射线卷积+多查询多头注意力**实现，兼顾特征提取能力与轻量化部署需求：

1. **光线路径**：1D深度可分离卷积+SE通道注意力，高效编码射线离障距离特征；

2. **注意力融合**：将射线特征与运动学历史融合，捕捉空间关联信息；

3. **策略/价值头**：共享编码器主干，分别输出动作分布（高斯分布+Tanh压缩）与状态价值；

4. **动作约束**：输出速度指令自动匹配物理上限，直接对接实际设备控制接口。

### 4. 高效PPO训练框架

- **GAE优势估计**：基于广义优势估计计算优势函数，平衡方差与偏差，提升训练稳定性；

- **批量缓冲区**：实现高效的轨迹数据存储与小批量采样，适配MindSpore张量操作；

- **混合精度训练**：自动开启FP16混合精度，大幅提升训练速度，同时保证训练精度；

- **梯度裁剪**：全局梯度裁剪，避免训练过程中梯度爆炸，提升训练鲁棒性。

## 目录结构

本项目目录结构清晰，模块解耦，便于二次开发与功能扩展，所有模块均基于MindSpore实现：

```Plain Text

├── config/                  # 配置文件目录
│   ├── env_config.json      # 环境/仿真/奖励配置
│   └── train_config.json    # PPO/模型/训练运行配置
├── rl_ppo/                  # PPO核心训练模块
│   ├── ppo_train.py         # 训练入口脚本
│   ├── mind_models.py       # MindSpore版策略/价值网络（Ray-Attention）
│   ├── mind_buffer.py       # MindSpore版GAE轨迹缓冲区
│   └── mind_encoder.py      # Ray-Attention编码器主干
├── mind_env/                # MindSpore版仿真环境
│   ├── sim_mind_env.py      # 批量随机射线仿真环境（核心）
│   ├── ray.py               # 射线生成与距离计算工具
│   └── mind_env_utils.py    # 环境工具函数（配置加载、数据转换）
├── tools/                   # 辅助工具
│   ├── export_onnx.py       # 模型导出ONNX脚本（推理部署）
│   └── analyze_ray.py       # 射线分布与训练指标分析工具
├── runs/                    # 检查点与日志默认目录
│   └── ppo_mind/            # 训练检查点、TensorBoard日志
├── requirements.txt         # 项目依赖清单
└── README.md                # 项目说明文档
```

## 模型推理部署

### 1. 导出ONNX模型

训练完成后，执行导出脚本将MindSpore模型转换为ONNX格式，脱离训练框架依赖，支持轻量化推理：

```Bash

python tools/export_onnx.py --ckpt_path runs/ppo_mind/latest.ckpt --onnx_path runs/ppo_mind/gralp_mind.onnx
```

- 导出的ONNX模型包含完整的推理逻辑，输入为标准化观测向量，输出为速度控制指令；

- 自动完成模型输入输出标准化，无需额外处理，直接对接推理框架。

### 2. 轻量化推理

基于onnxruntime实现推理，支持CPU/GPU，无需安装MindSpore，推理代码示例：

```Python

import onnxruntime as ort
import numpy as np

# 加载ONNX模型
ort_sess = ort.InferenceSession("runs/ppo_mind/gralp_mind.onnx", providers=["CPUExecutionProvider"])
# 构造观测向量（维度：射线数+7，与训练配置一致）
obs = np.random.randn(1, 43).astype(np.float32)  # 示例：40条射线+7维运动学历史
# 推理得到速度指令 (vx, omega)
action = ort_sess.run(None, {"obs": obs})[0]
print(f"推理速度指令：vx={action[0][0]:.3f}, omega={action[0][1]:.3f}")
```

- 推理输入：标准化的观测向量（射线距离归一化+运动学历史归一化）；

- 推理输出：连续速度指令（vx：线速度，omega：角速度），直接用于设备控制；

- 硬件适配：更换`providers`为`["CUDAExecutionProvider"]`即可实现GPU加速推理。

## 观测与动作格式

### 观测向量（维度：R+7，R为射线数量）

观测由**归一化射线距离**+**运动学历史特征**组成，所有特征均归一化到[0,1]或[-1,1]，保证模型训练稳定性：

`[rays_norm(R), sin_ref, cos_ref, prev_vx/lim, prev_omega/lim, Δvx/(2·lim), Δomega/(2·lim), dist_to_task/patch_meters]`

- `rays_norm(R)`：R条射线的离障距离，归一化到[0,1]；

- `sin_ref/cos_ref`：任务点相对朝向的正余弦，归一化到[-1,1]；

- `prev_vx/lim/prev_omega/lim`：上一步速度指令，归一化到[-1,1]；

- `Δvx/Δomega`：速度指令变化量，归一化到[-1,1]；

- `dist_to_task/patch_meters`：到任务点的距离，归一化到[0,1]。

### 动作向量（维度：2）

输出连续的平面速度控制指令，直接匹配被控设备的物理限制，无需额外转换：

`[vx, omega]`

- `vx`：线速度指令，范围[-vx_max, vx_max]，对应`env_config.json`中的`vx_max`；

- `omega`：角速度指令，范围[-omega_max, omega_max]，对应`env_config.json`中的`omega_max`。

## 训练监控

训练过程中自动生成TensorBoard日志，可通过以下命令启动监控面板，实时查看训练指标：

```Bash

tensorboard --logdir runs/ppo_mind/
```

可监控的核心指标包括：

- 损失指标：策略损失、价值损失、熵奖励、KL散度；

- 奖励指标：平均episode奖励、任务进度奖励、碰撞惩罚；

- 训练效率：FPS、训练步数、批量处理时间；

- 模型参数：动作分布对数标准差、注意力权重分布；

- 环境指标：碰撞率、任务完成率、平均episode长度。

## 硬件适配建议

### 训练硬件

- **Ascend NPU**：推荐Ascend 910/310P，适配MindSpore原生算子，训练效率最高，支持大批次`batch_env=2048`；

- **GPU**：推荐NVIDIA RTX 3090/A100，支持混合精度训练，建议`batch_env=512-1024`；

- **CPU**：仅用于算法调试与小批量测试，建议`batch_env=32-64`。

### 推理硬件

- **嵌入式CPU**：如RK3588、Jetson Nano，基于onnxruntime CPU推理，满足实时控制需求；

- **GPU**：如Jetson TX2/Xavier，基于onnxruntime GPU推理，提升推理速度；

- **Ascend NPU**：如Ascend 310，基于MindSpore Lite或onnxruntime Ascend推理，兼顾效率与功耗。

## 二次开发指南

本项目所有模块解耦，基于MindSpore原生API实现，便于二次开发与功能扩展：

1. **环境扩展**：修改`mind_env/sim_mind_env.py`，可添加自定义观测特征、奖励机制、任务点生成逻辑；

2. **网络扩展**：修改`rl_ppo/mind_encoder.py`，可调整注意力网络参数、添加新的特征提取模块；

3. **算法扩展**：基于`rl_ppo/mind_buffer.py`和`ppo_train.py`，可扩展DDPG/TD3/SAC等其他强化学习算法；

4. **硬件对接**：修改推理脚本，将ONNX模型推理输出的速度指令对接实际设备的控制接口（如串口、CAN总线）。

## 常见问题

1. **训练时FPS过低**：根据硬件算力降低`train_config.json`中的`batch_env`参数，或开启混合精度训练；

2. **模型碰撞率过高**：提高`env_config.json`中的`w_collision`奖励权重，增加碰撞惩罚；

3. **模型不收敛**：调整PPO学习率`lr`、裁剪范围`clip_range`，或增加`total_env_steps`总训练步数；

4. **ONNX推理报错**：确保训练时的观测维度与推理时一致，检查输入张量的数据类型为`float32`；

5. **Ascend平台算子报错**：确保MindSpore版本与Ascend驱动版本匹配，参考MindSpore官方安装文档。
> （注：文档部分内容可能由 AI 生成）
