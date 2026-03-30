import numpy as np
from rplidar import RPLidar, RPLidarException
import threading
import time

class LidarDriver:
    def __init__(self, port='COM5', n_rays=128, max_dist=10.0, baudrate=115200):
        """
        专门针对 RPLidar A1 + Windows (COM5) 的驱动
        """
        self.port = port
        self.baudrate = baudrate
        self.n_rays = n_rays
        self.max_dist = max_dist
        
        self.lidar = None
        # 初始化数据为最大距离 (假设周围全是空旷的)
        self.scan_data = np.ones(self.n_rays) * self.max_dist 
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """启动雷达"""
        try:
            # 连接雷达
            self.lidar = RPLidar(self.port, baudrate=self.baudrate)
            # 强制启动电机
            self.lidar.start_motor()
            print(f"✅ [Lidar] 成功连接到 {self.port}，波特率 {self.baudrate}")
            
            self.running = True
            # 开启后台线程一直在那读数据
            self.thread = threading.Thread(target=self._scan_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"❌ [Lidar Error] 连接失败: {e}")
            print("提示: 请检查 COM5 是否被其他软件占用了 (比如刚才那个黑框)")

    def stop(self):
        self.running = False
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()
            except:
                pass
        print("[Lidar] 已停止。")

    def _scan_loop(self):
        """后台循环：读取原始数据并“清洗”成 128 个点"""
        while self.running:
            try:
                # iter_scans 会返回一整圈的数据
                for scan in self.lidar.iter_scans(max_buf_meas=500):
                    if not self.running: break
                    self._process_scan(scan)
            except RPLidarException as e:
                # 雷达经常会报一些小错，忽略并重试即可
                # print(f"[Lidar Info] 信号抖动: {e}") 
                if self.lidar:
                    try:
                        self.lidar.clean_input()
                    except:
                        pass
            except Exception as e:
                print(f"[Lidar Error] {e}")
                break

    def _process_scan(self, scan):
        """
        关键算法：把不定长的 360 个点 -> 变成固定的 128 个点
        """
        # 1. 准备一个全是 10米的空数组
        temp_rays = np.ones(self.n_rays) * self.max_dist 

        # 2. 遍历这一圈扫描到的每一个点
        for (_, angle, dist_mm) in scan:
            if dist_mm <= 10: continue # 过滤噪点
            
            # 3. 算出这个点属于 0~127 中的哪一个格子
            # 角度(0-360) / 360 * 128
            idx = int((angle / 360.0) * self.n_rays)
            
            # 防止下标越界
            idx = min(idx, self.n_rays - 1)
            
            dist_m = dist_mm / 1000.0
            
            # 4. 取最小值 (Min Pooling)：如果这个格子里已经有更近的点，保留更近的
            if dist_m < temp_rays[idx]:
                temp_rays[idx] = dist_m

        # 5. 更新给主程序看
        with self.lock:
            self.scan_data = temp_rays

    def get_observation(self):
        """给神经网络喂数据用的接口"""
        with self.lock:
            return self.scan_data.copy()

# --- 这里是测试代码，只在这个文件运行时才跑 ---
if __name__ == "__main__":
    # 创建驱动实例
    driver = LidarDriver(port='COM5', n_rays=128)
    driver.start()
    
    try:
        while True:
            # 每隔 0.5秒 看一次数据
            obs = driver.get_observation()
            
            # 打印 4 个方向的距离 (前、左、后、右)
            n = 128
            # 索引大概是: 前=0, 左=32, 后=64, 右=96
            front = obs[0]
            left  = obs[n // 4]
            back  = obs[n // 2]
            right = obs[n * 3 // 4]
            
            print(f"前: {front:.2f}m | 左: {left:.2f}m | 后: {back:.2f}m | 右: {right:.2f}m")
            print("-" * 30)
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        driver.stop()