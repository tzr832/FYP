
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class HestonParam:
    """
    Heston模型参数数据类
    
    参数:
        v0 (float): 初始波动率, 默认0.04
        theta (float): 长期波动率均值, 默认0.04  
        kappa (float): 波动率回归速度, 默认4.0
        rho (float): 资产价格与波动率的相关系数, 默认-0.7
        sigmav (float): 波动率的波动率, 默认0.4
        mu (float): 资产收益率, 默认0.05
    """
    v0: float = 0.04
    theta: float = 0.04
    kappa: float = 4.0
    rho: float = -0.7
    sigmav: float = 0.4
    mu: float = 0.05


class HestonSimulator:
    """
    Heston随机波动率模型模拟器
    
    支持多个路径同步模拟和绘制fan chart功能
    """
    
    def __init__(self, param: HestonParam):
        """
        初始化Heston模拟器
        
        参数:
            param (HestonParam): Heston模型参数
        """
        self.set_params(param)
    
    def set_params(self, param: HestonParam):
        """
        更新模型参数
        
        参数:
            param (HestonParam): 新的Heston模型参数
        """
        self.v0 = param.v0
        self.theta = param.theta
        self.kappa = param.kappa
        self.rho = param.rho
        self.sigmav = param.sigmav
        self.mu = param.mu
    
    def get_param(self, want_class=True):
        """
        获取当前模型参数
        
        参数:
            want_class (bool): 是否返回HestonParam类实例
            
        返回:
            HestonParam或tuple: 模型参数
        """
        if want_class:
            return HestonParam(v0=self.v0,
                            theta=self.theta,
                            kappa=self.kappa,
                            sigmav=self.sigmav,
                            rho=self.rho,
                            mu=self.mu)
        else:
            return (self.v0, self.theta, self.kappa, self.sigmav, self.rho, self.mu)
    
    def simulate_euler(self, 
                      S0: float, 
                      T: float, 
                      n_steps: int, 
                      n_paths: int = 1000,
                      antithetic: bool = True,
                      fix_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用Euler-Maruyama方法模拟Heston过程
        
        参数:
            S0 (float): 初始资产价格
            T (float): 总模拟时间
            n_steps (int): 时间步数
            n_paths (int): 模拟路径数量
            antithetic (bool): 是否使用对偶变量法
            fix_seed (int, optional): 随机数种子
            
        返回:
            Tuple[np.ndarray, np.ndarray]: (价格路径, 方差路径)
                形状: (n_paths, n_steps+1)
        """
        if fix_seed is not None:
            np.random.seed(fix_seed)
        
        # 计算时间步长
        dt = T / n_steps
        
        # 初始化路径数组
        S_paths = np.zeros((n_paths, n_steps + 1))
        v_paths = np.zeros((n_paths, n_steps + 1))
        
        # 设置初始值
        S_paths[:, 0] = S0
        v_paths[:, 0] = self.v0
        
        # 生成随机数
        if antithetic:
            # 使用对偶变量法
            n_actual_paths = n_paths // 2
            Z1 = np.random.randn(n_actual_paths, n_steps)
            Z2 = np.random.randn(n_actual_paths, n_steps)
            
            # 生成相关的随机数
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            # 创建对偶路径
            W1_full = np.vstack([W1, -W1])
            W2_full = np.vstack([W2, -W2])
        else:
            # 不使用对偶变量法
            Z1 = np.random.randn(n_paths, n_steps)
            Z2 = np.random.randn(n_paths, n_steps)
            W1_full = Z1
            W2_full = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        
        # 模拟过程
        for t in range(n_steps):
            # 当前时刻的方差
            v_current = v_paths[:, t]
            
            # 确保方差非负（反射边界条件）
            v_current = np.maximum(v_current, 0)
            
            # 计算方差增量
            dv = self.kappa * (self.theta - v_current) * dt + \
                 self.sigmav * np.sqrt(v_current) * W2_full[:, t] * np.sqrt(dt)
            
            # 更新方差路径
            v_paths[:, t+1] = v_current + dv
            
            # 确保方差非负
            v_paths[:, t+1] = np.maximum(v_paths[:, t+1], 0)
            
            # 计算价格增量
            dS = self.mu * S_paths[:, t] * dt + \
                 np.sqrt(v_paths[:, t]) * S_paths[:, t] * W1_full[:, t] * np.sqrt(dt)
            
            # 更新价格路径
            S_paths[:, t+1] = S_paths[:, t] + dS
        
        # 保存最后模拟的路径用于绘图
        self._last_S_paths = S_paths
        self._last_v_paths = v_paths
        
        return S_paths, v_paths
    
    def simulate_milstein(self, 
                         S0: float, 
                         T: float, 
                         n_steps: int, 
                         n_paths: int = 1000,
                         antithetic: bool = True,
                         fix_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用Milstein方法模拟Heston过程（更精确的离散化）
        
        参数:
            S0 (float): 初始资产价格
            T (float): 总模拟时间
            n_steps (int): 时间步数
            n_paths (int): 模拟路径数量
            antithetic (bool): 是否使用对偶变量法
            fix_seed (int, optional): 随机数种子
            
        返回:
            Tuple[np.ndarray, np.ndarray]: (价格路径, 方差路径)
                形状: (n_paths, n_steps+1)
        """
        if fix_seed is not None:
            np.random.seed(fix_seed)
        
        # 计算时间步长
        dt = T / n_steps
        
        # 初始化路径数组
        S_paths = np.zeros((n_paths, n_steps + 1))
        v_paths = np.zeros((n_paths, n_steps + 1))
        
        # 设置初始值
        S_paths[:, 0] = S0
        v_paths[:, 0] = self.v0
        
        # 生成随机数
        if antithetic:
            n_actual_paths = n_paths // 2
            Z1 = np.random.randn(n_actual_paths, n_steps)
            Z2 = np.random.randn(n_actual_paths, n_steps)
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            W1_full = np.vstack([W1, -W1])
            W2_full = np.vstack([W2, -W2])
        else:
            Z1 = np.random.randn(n_paths, n_steps)
            Z2 = np.random.randn(n_paths, n_steps)
            W1_full = Z1
            W2_full = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        
        # 模拟过程
        for t in range(n_steps):
            # 当前时刻的方差
            v_current = v_paths[:, t]
            v_current = np.maximum(v_current, 0)
            
            # Milstein方法：添加二阶项
            # 对于方差过程：dv = kappa*(theta - v)*dt + sigmav*sqrt(v)*dW2 + 0.25*sigmav^2*(dW2^2 - dt)
            dv_euler = self.kappa * (self.theta - v_current) * dt + \
                      self.sigmav * np.sqrt(v_current) * W2_full[:, t] * np.sqrt(dt)
            
            # Milstein修正项
            milstein_correction = 0.25 * self.sigmav**2 * (W2_full[:, t]**2 * dt - dt)
            
            # 更新方差路径
            v_paths[:, t+1] = v_current + dv_euler + milstein_correction
            v_paths[:, t+1] = np.maximum(v_paths[:, t+1], 0)
            
            # 对于价格过程：使用Milstein方法
            # dS = mu*S*dt + sqrt(v)*S*dW1 + 0.5*v*S*(dW1^2 - dt)
            dS_euler = self.mu * S_paths[:, t] * dt + \
                      np.sqrt(v_current) * S_paths[:, t] * W1_full[:, t] * np.sqrt(dt)
            
            # Milstein修正项
            milstein_correction_S = 0.5 * v_current * S_paths[:, t] * (W1_full[:, t]**2 * dt - dt)
            
            # 更新价格路径
            S_paths[:, t+1] = S_paths[:, t] + dS_euler + milstein_correction_S
        
        # 保存最后模拟的路径用于绘图
        self._last_S_paths = S_paths
        self._last_v_paths = v_paths
        
        return S_paths, v_paths
    
    def simulate_full_truncation(self, 
                               S0: float, 
                               T: float, 
                               n_steps: int, 
                               n_paths: int = 1000,
                               antithetic: bool = True,
                               fix_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用完全截断方法模拟Heston过程（处理负方差的稳健方法）
        
        参数:
            S0 (float): 初始资产价格
            T (float): 总模拟时间
            n_steps (int): 时间步数
            n_paths (int): 模拟路径数量
            antithetic (bool): 是否使用对偶变量法
            fix_seed (int, optional): 随机数种子
            
        返回:
            Tuple[np.ndarray, np.ndarray]: (价格路径, 方差路径)
                形状: (n_paths, n_steps+1)
        """
        if fix_seed is not None:
            np.random.seed(fix_seed)
        
        dt = T / n_steps
        
        S_paths = np.zeros((n_paths, n_steps + 1))
        v_paths = np.zeros((n_paths, n_steps + 1))
        
        S_paths[:, 0] = S0
        v_paths[:, 0] = self.v0
        
        # 生成随机数
        if antithetic:
            n_actual_paths = n_paths // 2
            Z1 = np.random.randn(n_actual_paths, n_steps)
            Z2 = np.random.randn(n_actual_paths, n_steps)
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            W1_full = np.vstack([W1, -W1])
            W2_full = np.vstack([W2, -W2])
        else:
            Z1 = np.random.randn(n_paths, n_steps)
            Z2 = np.random.randn(n_paths, n_steps)
            W1_full = Z1
            W2_full = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        
        # 完全截断方法
        for t in range(n_steps):
            v_current = v_paths[:, t]
            
            # 完全截断：在计算扩散项时截断方差
            v_positive = np.maximum(v_current, 0)
            
            # 更新方差路径
            dv = self.kappa * (self.theta - v_current) * dt + \
                 self.sigmav * np.sqrt(v_positive) * W2_full[:, t] * np.sqrt(dt)
            v_paths[:, t+1] = v_current + dv
            
            # 更新价格路径
            dS = self.mu * S_paths[:, t] * dt + \
                 np.sqrt(v_positive) * S_paths[:, t] * W1_full[:, t] * np.sqrt(dt)
            S_paths[:, t+1] = S_paths[:, t] + dS
        
        # 保存最后模拟的路径用于绘图
        self._last_S_paths = S_paths
        self._last_v_paths = v_paths
        
        return S_paths, v_paths
    
    def plot_fan_chart(self, 
                      S_paths: np.ndarray, 
                      time_grid: np.ndarray,
                      percentiles: list = [5, 25, 50, 75, 95],
                      title: str = "Heston模型价格路径Fan Chart",
                      figsize: Tuple[int, int] = (12, 8),
                      show_paths: bool = True,
                      n_sample_paths: int = 20) -> plt.Figure:
        """
        绘制价格路径的fan chart
        
        参数:
            S_paths (np.ndarray): 价格路径数组，形状 (n_paths, n_times)
            time_grid (np.ndarray): 时间网格
            percentiles (list): 要显示的百分位数
            title (str): 图表标题
            figsize (Tuple): 图表大小
            show_paths (bool): 是否显示样本路径
            n_sample_paths (int): 显示的样本路径数量
            
        返回:
            plt.Figure: matplotlib图形对象
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # 计算百分位数
        percentiles_sorted = sorted(percentiles)
        percentile_values = np.percentile(S_paths, percentiles_sorted, axis=0)
        
        # 绘制fan chart
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(percentiles_sorted)//2 + 1))
        
        # 填充不同百分位区域
        for i in range(len(percentiles_sorted)//2):
            lower_idx = i
            upper_idx = len(percentiles_sorted) - 1 - i
            ax1.fill_between(time_grid, 
                           percentile_values[lower_idx, :], 
                           percentile_values[upper_idx, :],
                           alpha=0.3, color=colors[i],
                           label=f'{percentiles_sorted[lower_idx]}%-{percentiles_sorted[upper_idx]}%')
        
        # 绘制中位数
        median_idx = len(percentiles_sorted) // 2
        ax1.plot(time_grid, percentile_values[median_idx, :], 
                'r-', linewidth=2, label=f'中位数 ({percentiles_sorted[median_idx]}%)')
        
        # 绘制样本路径（如果要求）
        if show_paths and n_sample_paths > 0:
            n_paths_total = S_paths.shape[0]
            sample_indices = np.random.choice(n_paths_total, 
                                            min(n_sample_paths, n_paths_total), 
                                            replace=False)
            for idx in sample_indices:
                ax1.plot(time_grid, S_paths[idx, :], 'gray', alpha=0.1)
        
        ax1.set_xlabel('时间')
        ax1.set_ylabel('资产价格')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制波动率路径的统计信息
        if hasattr(self, '_last_v_paths'):
            v_paths = self._last_v_paths
            v_percentiles = np.percentile(v_paths, percentiles_sorted, axis=0)
            
            # 绘制波动率fan chart
            for i in range(len(percentiles_sorted)//2):
                lower_idx = i
                upper_idx = len(percentiles_sorted) - 1 - i
                ax2.fill_between(time_grid, 
                               v_percentiles[lower_idx, :], 
                               v_percentiles[upper_idx, :],
                               alpha=0.3, color=colors[i],
                               label=f'{percentiles_sorted[lower_idx]}%-{percentiles_sorted[upper_idx]}%')
            
            # 绘制中位数
            ax2.plot(time_grid, v_percentiles[median_idx, :], 
                    'r-', linewidth=2, label=f'中位数 ({percentiles_sorted[median_idx]}%)')
            
            ax2.set_xlabel('时间')
            ax2.set_ylabel('波动率')
            ax2.set_title('Heston模型波动率路径Fan Chart')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_paths(self,
                  S_paths: np.ndarray,
                  time_grid: np.ndarray,
                  n_paths: int = 10,
                  title: str = "Heston模型价格路径",
                  figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        绘制多个价格路径
        
        参数:
            S_paths (np.ndarray): 价格路径数组
            time_grid (np.ndarray): 时间网格
            n_paths (int): 要显示的路径数量
            title (str): 图表标题
            figsize (Tuple): 图表大小
            
        返回:
            plt.Figure: matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        n_paths_total = S_paths.shape[0]
        paths_to_show = min(n_paths, n_paths_total)
        
        # 随机选择要显示的路径
        indices = np.random.choice(n_paths_total, paths_to_show, replace=False)
        
        # 绘制路径
        for idx in indices:
            ax.plot(time_grid, S_paths[idx, :], alpha=0.7, linewidth=1)
        
        # 绘制均值路径
        mean_path = np.mean(S_paths, axis=0)
        ax.plot(time_grid, mean_path, 'k-', linewidth=2, label='均值路径')
        
        ax.set_xlabel('时间')
        ax.set_ylabel('资产价格')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_statistics(self,
                       S_paths: np.ndarray,
                       time_grid: np.ndarray,
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制价格路径的统计信息
        
        参数:
            S_paths (np.ndarray): 价格路径数组
            time_grid (np.ndarray): 时间网格
            figsize (Tuple): 图表大小
            
        返回:
            plt.Figure: matplotlib图形对象
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 计算统计量
        mean_path = np.mean(S_paths, axis=0)
        std_path = np.std(S_paths, axis=0)
        min_path = np.min(S_paths, axis=0)
        max_path = np.max(S_paths, axis=0)
        
        # 绘制均值路径
        ax1.plot(time_grid, mean_path, 'b-', linewidth=2)
        ax1.set_xlabel('时间')
        ax1.set_ylabel('资产价格')
        ax1.set_title('价格路径均值')
        ax1.grid(True, alpha=0.3)
        
        # 绘制标准差
        ax2.plot(time_grid, std_path, 'r-', linewidth=2)
        ax2.set_xlabel('时间')
        ax2.set_ylabel('标准差')
        ax2.set_title('价格路径标准差')
        ax2.grid(True, alpha=0.3)
        
        # 绘制最小最大值范围
        ax3.fill_between(time_grid, min_path, max_path, alpha=0.3, color='green')
        ax3.plot(time_grid, mean_path, 'b-', linewidth=1)
        ax3.set_xlabel('时间')
        ax3.set_ylabel('资产价格')
        ax3.set_title('价格路径范围 (最小-最大)')
        ax3.grid(True, alpha=0.3)
        
        # 绘制最终价格分布
        final_prices = S_paths[:, -1]
        ax4.hist(final_prices, bins=50, alpha=0.7, density=True)
        ax4.set_xlabel('最终价格')
        ax4.set_ylabel('密度')
        ax4.set_title('最终价格分布')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    plt.rcParams['font.family'] = ['simhei']

    # 测试Heston模拟器
    print("=== Heston模型模拟器测试 ===")
    
    # 设置参数
    param = HestonParam(v0=0.04, theta=0.04, kappa=2.0, rho=-0.7, sigmav=0.1, mu=0.05)
    simulator = HestonSimulator(param)
    
    # 模拟参数
    S0 = 100.0
    T = 1.0
    n_steps = 252
    n_paths = 1000
    
    print(f"模拟参数: S0={S0}, T={T}, n_steps={n_steps}, n_paths={n_paths}")
    print(f"Heston参数: v0={param.v0}, theta={param.theta}, kappa={param.kappa}, rho={param.rho}, sigmav={param.sigmav}, mu={param.mu}")
    
    # 使用Euler方法模拟
    print("\n使用Euler方法模拟...")
    S_paths_euler, v_paths_euler = simulator.simulate_euler(S0, T, n_steps, n_paths, fix_seed=42)
    
    # 使用Milstein方法模拟
    print("使用Milstein方法模拟...")
    S_paths_milstein, v_paths_milstein = simulator.simulate_milstein(S0, T, n_steps, n_paths, fix_seed=42)
    
    # 使用完全截断方法模拟
    print("使用完全截断方法模拟...")
    S_paths_ft, v_paths_ft = simulator.simulate_full_truncation(S0, T, n_steps, n_paths, fix_seed=42)
    
    # 创建时间网格
    time_grid = np.linspace(0, T, n_steps + 1)
    
    # 计算最终价格的统计信息
    final_prices_euler = S_paths_euler[:, -1]
    final_prices_milstein = S_paths_milstein[:, -1]
    final_prices_ft = S_paths_ft[:, -1]
    
    print(f"\n最终价格统计:")
    print(f"Euler方法: 均值={np.mean(final_prices_euler):.4f}, 标准差={np.std(final_prices_euler):.4f}")
    print(f"Milstein方法: 均值={np.mean(final_prices_milstein):.4f}, 标准差={np.std(final_prices_milstein):.4f}")
    print(f"完全截断方法: 均值={np.mean(final_prices_ft):.4f}, 标准差={np.std(final_prices_ft):.4f}")
    
    # 绘制fan chart
    print("\n绘制fan chart...")
    fig_fan = simulator.plot_fan_chart(S_paths_euler, time_grid,
                                     title="Heston模型价格路径Fan Chart (Euler方法)")
    plt.show()
    
    # 绘制样本路径
    print("绘制样本路径...")
    fig_paths = simulator.plot_paths(S_paths_euler, time_grid, n_paths=20,
                                   title="Heston模型价格路径 (Euler方法)")
    plt.show()
    
    # 绘制统计信息
    print("绘制统计信息...")
    fig_stats = simulator.plot_statistics(S_paths_euler, time_grid)
    plt.show()
    
    print(f"\n=== Heston模型模拟器测试完成 ===")