import numpy as np
import math
from typing import Callable, Optional, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.special import gamma

@dataclass
class VSSParam:
    """
    Heston模型参数数据类
    """
    kappa: float  # Mean reversion speed
    nu: float     # Volatility of volatility
    rho: float    # Correlation between Brownian motions
    theta: float  # Long-term variance
    X_0: float    # Initial variance
    H: float      # Hurst index for Rough Stein-Stein model

    def __post_init__(self):
        if not (0 < self.H < 1):
            raise ValueError("Hurst index H must be in the interval (0, 1).")

class VolterraSteinSteinSimulator:
    """
    Volterra-Stein-Stein 模型模拟器类
    实现广义Volterra-Stein-Stein模型的路径模拟，支持任意κ值
    """
    
    def __init__(self, params: VSSParam, random_seed: Optional[int] = None):
        """
        初始化模拟器
        
        参数:
        random_seed: 随机种子，用于重现结果
        """
        self.params = params
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def set_params(self, param: VSSParam):
        self.params = param
    
    @staticmethod
    def K_brownian(t: float, s: float) -> float:
        """
        标准布朗运动核函数
        
        参数:
        t: 当前时间
        s: 历史时间
        
        返回:
        核函数值
        """
        return 1.0 if s < t else 0.0
    
    @staticmethod
    def K_fractional(t: float, s: float, H: float = 0.1) -> float:
        """
        分数布朗运动核函数（Riemann-Liouville型）
        
        参数:
        t: 当前时间
        s: 历史时间
        H: Hurst指数，默认0.25
        
        返回:
        核函数值
        """
        if s >= t:
            return 0.0
        return (t - s)**(H - 0.5) / math.gamma(H + 0.5)
    
    def simulate_vss_path(self, 
                         T: float, 
                         n_points: int, 
                         S0: float,
                         N_paths: int = 1,
                         random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟广义Volterra-Stein-Stein模型（κ可为任意值）的路径，使用欧拉-丸山离散化
        
        参数:
        T: 时间终点
        n_points: 时间网格点数
        kappa: 均值回归速率参数
        nu: 波动率参数
        g0_func: 均值函数g0(t)，可调用函数
        K_func: 核函数K(t,s)，可调用函数
        N_paths: 模拟路径数
        random_seed: 随机种子，用于重现结果
        
        返回:
        t_grid: 时间网格，形状 (n_points,)
        X_paths: Xt的模拟路径，形状 (N_paths, n_points)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 创建时间网格 [0, T]
        t_grid = np.linspace(0, T, n_points)
        dt = t_grid[1] - t_grid[0]
        
        # 初始化路径数组
        X_paths = np.zeros((N_paths, n_points))
        S_paths = np.zeros((N_paths, n_points))  # 如果需要存储S_t，可以启用

        # 计算核矩阵 K(t_i, t_j) 对于所有 i, j，这将大幅提升递归计算速度
        # 因为核函数是Volterra型（s >= t 时为0），所以矩阵是下三角矩阵
        K_matrix = np.zeros((n_points, n_points))
        for i in range(n_points):
            t_i = t_grid[i]
            for j in range(i+1):  # 只计算下三角部分，因为当 j > i 时 K(t_i, t_j)=0
                t_j = t_grid[j]
                K_matrix[i, j] = self.K_fractional(t_i, t_j, self.params.H)
        
        # 计算g0向量
        alpha = self.params.H + 0.5
        g0_vector = self.params.X_0 + self.params.theta * t_grid ** alpha / gamma(1 + alpha)
        
        # 生成布朗运动增量 dW ~ N(0, sqrt(dt))
        # 形状: (N_paths, n_points)
        dW = np.random.normal(0, math.sqrt(dt), size=(N_paths, n_points))
        
        # 对每条路径进行循环
        for path in range(N_paths):
            # 初始化当前路径
            X_current = np.zeros(n_points)
            
            # 递归计算每个时间点 t_i 的 X_{t_i}
            for i in range(n_points):
                # 递推公式:
                # X_{t_i} = g0(t_i) + κ * Δt * Σ_{j=0}^{i-1} [K(t_i, t_j) * X_{t_j}] + ν * Σ_{j=0}^{i-1} [K(t_i, t_j) * ΔW_j]
                
                # 1. 计算求和项: Σ_{j=0}^{i-1} [K(t_i, t_j) * X_{t_j}]
                sum_mean = 0.0
                if i > 0:
                    # 使用向量化操作计算求和，避免内部循环
                    sum_mean = np.dot(K_matrix[i, :i], X_current[:i])
                
                # 2. 计算求和项: Σ_{j=0}^{i-1} [K(t_i, t_j) * ΔW_j]
                sum_vol = 0.0
                if i > 0:
                    sum_vol = np.dot(K_matrix[i, :i], dW[path, :i])
                
                # 3. 组合所有项
                X_current[i] = g0_vector[i] + self.params.kappa * dt * sum_mean + self.params.nu * sum_vol
            
            # 存储当前路径
            X_paths[path, :] = X_current
            dB = self.params.rho * dW[path] + np.sqrt(1 - self.params.rho**2) * np.random.normal(0, math.sqrt(dt), size=n_points)
 
            # dS/S = X_t dB
            S_paths[path, 0] = S0
            XdB = np.zeros(n_points)  # XdB[0] = X_0 * dB_0 = 0
            XdB[1:] = X_current[:-1] * dB[:-1]
            S_paths[path, 1:] = S0 * np.cumprod(1 + XdB[1:])  # S_t = S_0 * Π(1 + X_s dB_s)
        
        return t_grid, X_paths, S_paths
    


# 示例使用
if __name__ == "__main__":
    plt.rcParams['font.family'] = ['simhei']

    # 创建模拟器实例
    param = VSSParam(
        kappa=-8.9e-5,      # Mean reversion speed
        nu=0.176,         # Volatility of volatility
        rho=-0.704,       # Correlation between Brownian motions
        theta= -0.044,     # Long-term variance
        X_0=0.113,        # Initial volatility (sqrt(variance))
        H=0.279           # Hurst index for Rough Stein-Stein model
    )
    # simulator = VolterraSteinSteinSimulator(param, random_seed=42)
    simulator = VolterraSteinSteinSimulator(param)
    
    # 定义参数
    T = 1.0
    n_points = 252
    S0 = 100.
    g0_func = lambda t: 0.2  # 常数均值
    
    print("=== Volterra-Stein-Stein 模型模拟测试 ===")
    
    # 测试1: 使用指数核函数模拟
    print("1. 使用指数核函数模拟...")
    t_grid, X_paths, S_paths = simulator.simulate_vss_path(T, n_points, S0, N_paths=10)
    print(f"   模拟完成: 时间网格形状 {t_grid.shape}, 路径形状 {X_paths.shape}")
    
    # 测试2: 绘制路径
    print("2. 绘制模拟路径...")
    plt.figure(figsize=(12, 6))
    for i in range(X_paths.shape[0]):
        plt.plot(t_grid, S_paths[i], lw=1, alpha=0.7)
    plt.title("Volterra-Stein-Stein 模型模拟路径 (指数核函数)")
    plt.xlabel("时间")
    plt.ylabel("资产价格 S_t")
    plt.grid(True)
    plt.show()
    
    # # 测试3: 绘制扇形图
    # print("3. 绘制扇形图...")
    # simulator.plot_fan_chart(t_grid, X_paths, title="指数核函数扇形图")
    
    # # 测试4: 比较不同核函数
    # print("4. 比较不同核函数...")
    # simulator.compare_kernels(T=0.5, n_points=100, N_paths=3)
    
    print("=== 测试完成 ===")