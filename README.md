# 基于 MindSpore 的无人机/机器人 LiDAR 避障强化学习项目

![MindSpore](https://img.shields.io/badge/MindSpore-2.8.0-blue.svg)
![Hardware](https://img.shields.io/badge/Hardware-Ascend_NPU-orange.svg)
![Python](https://img.shields.io/badge/Python-3.10-green.svg)

## 项目简介
本项目是一个基于 **MindSpore** 框架实现的强化学习自动巡航与避障系统。通过集成 **Transformer (RayEncoder)** 提取多维 LiDAR（激光雷达）点云特征，并利用 **PPO (Proximal Policy Optimization)** 算法进行端到端的策略训练。

本项目已在华为 **昇腾 (Ascend) NPU** 平台上完成适配与性能优化，特别针对深度强化学习在国产算力芯片上的数值稳定性问题进行了深度调优，成功解决了长周期训练中的 NaN 爆炸问题。

---

## 技术亮点
* **全栈国产化**：基于 MindSpore 2.8+ 与华为 ModelArts 平台开发，充分发挥昇腾芯片算力。
* **Transformer 特征提取**：采用自注意力机制处理非结构化 LiDAR 射线数据，增强模型对复杂障碍物环境的感知深度。
* **高性能训练**：利用 MindSpore 静态图（Graph Mode）与混合精度（float16）技术，训练吞吐量稳定在 **400+ FPS**。
* **数值稳定性防御 (Anti-NaN)**：内置梯度裁剪、Log_std 范围硬约束以及 NaN 自动排毒逻辑，保障模型在 40 万步以上的长周期训练稳定性。

---

## 核心架构
* **算法**: PPO (Clipped Objective)
* **感知网络 (Encoder)**: Transformer-based RayEncoder
* **决策网络 (Actor-Critic)**: 多层感知机 (MLP)



---

## 快速开始

### 1. 环境准备
推荐在华为云 ModelArts（预装镜像MindSpore2.6.0rc1以上版本）中运行

### 2. 模型训练
请在终端执行以下命令开始训练：

`python mind_train.py`

### 3.训练输出说明

训练过程会实时打印：
- FPS：训练速度
- avg_ep_reward：平均回合奖励（越高越好）
- log_std：动作探索标准差（收敛时约 -1.0 ~ -1.5）
- loss：策略与价值损失

所有输出保存在：runs/ppo_exp1/

该目录下包含：
- latest_policy.ckpt：最新训练权重
- final.ckpt：最终训练权重
- TensorBoard 事件日志文件
- 
### 4.TensorBoard 可视化
启动命令：

`tensorboard --logdir runs/ppo_exp1/`

可观测曲线：
Reward 奖励曲线（持续上升）
log_std 标准差曲线（持续下降并稳定）
Loss 曲线
回合长度变化
收敛判断：
Reward 稳定不再增长
log_std 稳定在 -1.0 ~ -1.5
智能体避障流畅、不碰撞

### 5.模型推理
训练完成后运行：
`python mind_infer.py`
功能：
模拟 LiDAR 数据，加载训练好的 ckpt 权重

