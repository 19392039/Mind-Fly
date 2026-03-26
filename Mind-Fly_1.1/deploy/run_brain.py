import sys
import os
import time
import numpy as np
from rplidar import RPLidar

# ================= ⚙️ 飞行参数设置 =================
PORT_NAME = 'COM5'
BAUD_RATE = 115200

# 距离阈值 (单位: mm)
SAFE_DIST = 800.0      # 安全距离 (0.8米)，大于这个距离全速前进
STOP_DIST = 400.0      # 刹车距离 (0.4米)，小于这个距离开始倒车
WALL_DIST = 500.0      # 侧面避障距离 (0.5米)，小于这个距离开始推离

# 速度限制
MAX_VX = 0.5           # 最大前进速度 (m/s)
MAX_VY = 0.3           # 最大横移速度 (m/s)
MAX_OMEGA = 1.0        # 最大旋转速度 (rad/s)

# PID 系数 (调整反应灵敏度)
KP_VX = 0.002          # 前后灵敏度
KP_VY = 0.003          # 左右灵敏度
KP_OMEGA = 0.005       # 转向灵敏度
# ===================================================

def get_sector_min_dist(scan, target_angle, window=30.0):
    """
    获取指定角度扇区内的最小距离
    scan: 雷达数据列表
    target_angle: 中心角度 (如 180)
    window: 扇区宽度 (如 ±30度)
    """
    valid_dists = []
    for pt in scan:
        dist = pt[2]
        angle = pt[1]
        
        # 1. 过滤无效点 (0) 和过远点 (超过3米忽略，减少干扰)
        if dist > 10 and dist < 3000:
            # 2. 角度匹配 (处理 0/360 交界处的逻辑)
            # 简单起见，这里假设不跨越 0度 (180, 90, 270 都没问题)
            if (target_angle - window) <= angle <= (target_angle + window):
                valid_dists.append(dist)
    
    if len(valid_dists) > 0:
        return np.min(valid_dists)
    else:
        return 5000.0 # 没看到障碍物，返回无限远

def main():
    lidar = RPLidar(PORT_NAME, baudrate=BAUD_RATE)
    
    print(">>> 全向感知飞行控制器已启动 <<<")
    print("策略: 180°为前, 90°/270°为左右")
    print(f"{'前距':<6} {'左距':<6} {'右距':<6} || {'VX':<6} {'VY':<6} {'OMEGA':<6} || {'状态'}")
    print("-" * 80)

    try:
        lidar.start_motor()
        for scan in lidar.iter_scans(max_buf_meas=500):
            t_now = time.time()
            
            # --- 1. 感知层 (Perception) ---
            # 根据你之前的测试：180是前。
            # 假设顺时针旋转：90是右，270是左 (如果反了，下面逻辑里的 Left/Right 对调即可)
            
            d_front = get_sector_min_dist(scan, 180.0, window=30)
            d_right = get_sector_min_dist(scan, 90.0,  window=30)
            d_left  = get_sector_min_dist(scan, 270.0, window=30) # 有些雷达左是270

            # --- 2. 决策层 (Decision) ---
            
            # A. 计算 VX (前后控制)
            # 逻辑：距离越近，速度越小；小于 Stop 距离则为负
            # 误差 = (当前前方距离 - 期望刹车距离)
            vx = (d_front - STOP_DIST) * KP_VX
            # 限幅
            vx = max(min(vx, MAX_VX), -MAX_VX)

            # B. 计算 VY (左右横移控制 - 仅适用于能横移的无人机)
            # 逻辑：保持左右居中。左边近就往右飞(Vy>0)，右边近就往左飞(Vy<0)
            # 误差 = (左边距离 - 右边距离)
            # 如果左边是 300，右边是 1000 -> 误差 -700 -> 往右飞
            # 注意：坐标系定义可能不同，通常右为正Y，或者右为正Y
            # 假设：Body Frame下，右是正 Y
            vy = 0.0
            if d_left < WALL_DIST or d_right < WALL_DIST:
                # 只有当某一侧离墙太近时才触发侧向避障
                vy = (d_left - d_right) * KP_VY
            
            vy = max(min(vy, MAX_VY), -MAX_VY)

            # C. 计算 Omega (转向控制 - 适用于所有无人机)
            # 逻辑：前方有障碍时，向宽敞的一侧转弯
            omega = 0.0
            if d_front < SAFE_DIST:
                # 前方有障碍，开始转向
                if d_left > d_right:
                    # 左边宽敞 -> 向左转 (通常逆时针为正，即 +Omega)
                    # 也有可能顺时针为正，看飞控定义。这里假设 左转 = +Omega
                    omega = 0.5 
                else:
                    # 右边宽敞 -> 向右转 (-Omega)
                    omega = -0.5
            else:
                # 前方无障碍，微调方向保持走直线 (可选)
                # 这里简单处理：让它保持直行
                omega = 0.0

            # --- 3. 输出层 (Action) ---
            
            status = "🟢 巡航"
            if vx < 0: status = "🔴 倒车/刹车"
            elif abs(omega) > 0.1: status = "🔄 避障转向"
            elif abs(vy) > 0.1: status = "↔️ 侧向修正"

            # 打印最终指令
            print(f"\r{d_front:4.0f} | {d_left:4.0f} | {d_right:4.0f} || {vx:+4.2f} | {vy:+4.2f} | {omega:+4.2f} || {status:<10}", end='')

    except Exception as e:
        print(f"\nError: {e}")
    except KeyboardInterrupt:
        print("\n停止。")
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()

if __name__ == "__main__":
    main()