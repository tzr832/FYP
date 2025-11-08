import numpy as np
import math
from typing import Callable, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.special import gamma

from VSS_COS import VSSParam

class VolterraSteinSteinSimulator:
    """
    Volterra-Stein-Stein model simulator class
    Implements path simulation for generalized Volterra-Stein-Stein model, supports arbitrary κ values
    """
    
    def __init__(self, params: VSSParam, random_seed: Optional[int] = None):
        """
        Initialize simulator
        
        Parameters:
        random_seed: random seed for reproducible results
        """
        self.params = params
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def set_params(self, param: VSSParam):
        self.params = param
    
    @staticmethod
    def K_brownian(t: float, s: float) -> float:
        """
        Standard Brownian motion kernel function
        
        Parameters:
        t: current time
        s: historical time
        
        Returns:
        kernel function value
        """
        return 1.0 if s < t else 0.0
    
    @staticmethod
    def K_fractional(t: float, s: float, H: float = 0.1) -> float:
        """
        Fractional Brownian motion kernel function (Riemann-Liouville type)
        
        Parameters:
        t: current time
        s: historical time
        H: Hurst index, default 0.25
        
        Returns:
        kernel function value
        """
        if s >= t:
            return 0.0
        return (t - s)**(H - 0.5) / math.gamma(H + 0.5)
    
    def simulate_vss_path(self, 
                         T: float, 
                         S0: float,
                         n_points: int, 
                         N_paths: int = 1,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate paths for generalized Volterra-Stein-Stein model (κ can be arbitrary) using Euler-Maruyama discretization
        
        Parameters:
        - T: time horizon
        - S0: initial asset price
        - n_points: number of time grid points
        - N_paths: number of simulation paths
        
        Returns:
        t_grid: time grid, shape (n_points,)
        X_paths: simulated paths of Xt, shape (N_paths, n_points)
        """
        
        # Create time grid [0, T]
        t_grid = np.linspace(0, T, n_points)
        dt = t_grid[1] - t_grid[0]
        
        # Initialize path arrays
        X_paths = np.zeros((N_paths, n_points))
        S_paths = np.zeros((N_paths, n_points))  # Enable if S_t storage is needed

        # Compute kernel matrix K(t_i, t_j) for all i, j, this significantly improves recursive computation speed
        # Since kernel function is Volterra type (0 when s >= t), the matrix is lower triangular
        K_matrix = np.zeros((n_points, n_points))
        for i in range(n_points):
            t_i = t_grid[i]
            for j in range(i+1):  # Only compute lower triangular part, since K(t_i, t_j)=0 when j > i
                t_j = t_grid[j]
                K_matrix[i, j] = self.K_fractional(t_i, t_j, self.params.H)
        
        # Compute g0 vector
        alpha = self.params.H + 0.5
        g0_vector = self.params.X_0 + self.params.theta * t_grid ** alpha / gamma(1 + alpha)
        
        # Generate Brownian motion increments dW ~ N(0, sqrt(dt))
        # Shape: (N_paths, n_points)
        dW = np.random.normal(0, math.sqrt(dt), size=(N_paths, n_points))
        
        # Loop over each path
        for path in range(N_paths):
            # Initialize current path
            X_current = np.zeros(n_points)
            
            # Recursively compute X_{t_i} for each time point t_i
            for i in range(n_points):
                # Recurrence formula:
                # X_{t_i} = g0(t_i) + κ * Δt * Σ_{j=0}^{i-1} [K(t_i, t_j) * X_{t_j}] + ν * Σ_{j=0}^{i-1} [K(t_i, t_j) * ΔW_j]
                
                # 1. Compute summation term: Σ_{j=0}^{i-1} [K(t_i, t_j) * X_{t_j}]
                sum_mean = 0.0
                if i > 0:
                    # Use vectorized operation to compute summation, avoiding inner loops
                    sum_mean = np.dot(K_matrix[i, :i], X_current[:i])
                
                # 2. Compute summation term: Σ_{j=0}^{i-1} [K(t_i, t_j) * ΔW_j]
                sum_vol = 0.0
                if i > 0:
                    sum_vol = np.dot(K_matrix[i, :i], dW[path, :i])
                
                # 3. Combine all terms
                X_current[i] = g0_vector[i] + self.params.kappa * dt * sum_mean + self.params.nu * sum_vol
            
            # Store current path
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
    plt.rcParams['font.family'] = ['SimHei']

    # Create simulator instance
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
    
    # Define parameters
    T = 1.0
    n_points = 252
    S0 = 100.
    g0_func = lambda t: 0.2  # Constant mean
    
    print("=== Volterra-Stein-Stein Model Simulation Test ===")
    
    # Test 1: Simulate with exponential kernel function
    print("1. Simulating with exponential kernel function...")
    t_grid, X_paths, S_paths = simulator.simulate_vss_path(T, S0, n_points, N_paths=10)
    print(f"   Simulation completed: time grid shape {t_grid.shape}, path shape {X_paths.shape}")
    
    # Test 2: Plot paths
    print("2. Plotting simulated paths...")
    plt.figure(figsize=(12, 6))
    for i in range(X_paths.shape[0]):
        plt.plot(t_grid, S_paths[i], lw=1, alpha=0.7)
    plt.title("Volterra-Stein-Stein Model Simulated Paths")
    plt.xlabel("Time")
    plt.ylabel("Asset Price S_t")
    plt.grid(True)
    plt.show()
    
    # # 测试3: 绘制扇形图
    # print("3. 绘制扇形图...")
    # simulator.plot_fan_chart(t_grid, X_paths, title="指数核函数扇形图")
    
    # # 测试4: 比较不同核函数
    # print("4. 比较不同核函数...")
    # simulator.compare_kernels(T=0.5, n_points=100, N_paths=3)
    
    print("=== Test Completed ===")