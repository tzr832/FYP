import numpy as np
from dataclasses import dataclass
from typing import Union
from scipy.special import gamma, hyp2f1
import scipy.linalg as la
from cmath import polar

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

class VSSPricerCOS:
    """
    Volterra Stein-Stein model option pricer using the COS method.
    """
    def __init__(self, params: VSSParam):
        self.params = params
        self.KK = None
        self.SIG = None
        self.g = None
    
    def set_params(self, param: VSSParam):
        self.params = param
    
    def get_params(self, return_class: bool = True) -> VSSParam|tuple:
        if not return_class:
            return (self.params.kappa, self.params.nu, self.params.rho,
                    self.params.theta, self.params.X_0, self.params.H)
        else:
            return self.params
    
    def _rss_g_K_SIG_det(self, n, T):
        """
        Computes the kernel matrix, covariance matrix, and adjusted input vector for the
        generalized Volterra Stein-Stein (Rough Stein-Stein) model.

        Parameters:
        - n: int
            Number of discretization steps.
        - T: float
            Maturity time.
        - nu: float
            Volatility of volatility.
        - theta: float
            Long-term variance parameter for the deterministic input curve g_0(t).
        - X_0: float
            Initial value of the volatility process.
        - H: float
            Hurst index.

        Returns:
        - g: numpy array of shape (n, 1)
            Vector of adjusted values based on the input curve for each time step.
        - self.KK: numpy array of shape (n, n)
            2D matrix representing the kernel's approximation matrix.
        - self.SIG: numpy array of shape (n, n)
            2D covariance matrix Σ_0.
        """
        alpha = self.params.H + 0.5

        # Time discretization from 0 to T
        t = np.linspace(0, T, n + 1)

        # Define indices for 2D matrices
        tj_1 = np.tile(t[:-1], n).reshape(n, n)  # Times tj excluding the final point
        ti_1 = tj_1.T  # Transpose to create a grid of ti values
        tj = np.tile(t[1:], n).reshape(n, n)  # Times tj excluding the initial point

        # Initialize kernel matrix self.KK
        self.KK = np.zeros((n, n))
        # K^n_{ij}= \bm 1_{j\leq i-1}\int_{t_{j-1}}^{t_j} K(t_{i-1},s)ds  , \quad 1 \leq  i,j\leq n, 
        self.KK[tj <= ti_1] = ((ti_1 - tj_1) ** alpha - (ti_1 - tj) ** alpha)[tj <= ti_1] / gamma(1 + alpha)
        self.KK_sum = self.KK + self.KK.T
        self.KK_mul = self.KK @ self.KK.T

        # Compute covariance matrix self.SIG
        # \SIGma^n_{ij}=\nu^2 \int_0^T K(t_{i-1},s)K(t_{j-1},s)ds, \quad 1\leq i,j\leq n.
        self.SIG = hyp2f1(1, 1 - alpha, 1 + alpha, np.minimum(ti_1, tj_1) / np.maximum(ti_1, tj_1)) \
            * np.minimum(ti_1, tj_1) ** alpha / (np.maximum(ti_1, tj_1) ** (1 - alpha)) \
            * self.params.nu ** 2 / (gamma(1 + alpha) * gamma(alpha))

        # Handle numerical issues
        self.SIG[0, 0] = 0  # Setting the first element to zero
        self.SIG[self.SIG == np.inf] = 0  # Handle any infinite values due to division by zero

        # Compute adjusted vector g based on initial conditions
        self.g = (self.params.X_0 + self.params.theta * t[:-1] ** alpha / gamma(1 + alpha)).reshape(-1, 1)
    
    def _rss_discontinuous_eval(self, u, w, r, n, T, moneyness) -> tuple[complex, complex]:
        """
        Evaluates the Fourier-Laplace transform and Fredholm determinant
        for a single pair (u, w) using the discontinuous formula.
        """
        
        a = w + 0.5 * (u**2 - u)
        b = self.params.kappa + self.params.rho * self.params.nu * u

        # Compute tilde(self.SIG) = inv(I - b * K) * self.SIG * inv(I - b * K).T
        X = np.eye(n) - b * self.KK
        self.SIG_tilde = la.solve(X, la.solve(X, self.SIG.T).T)

        # Compute the determinant involved in the characteristic function
        D = np.eye(n) - 2 * a * T / n * self.SIG_tilde
        det_val = la.det(D)

        # Compute matrix Psi
        denom = np.eye(n) - b * self.KK_sum + b**2*(self.KK_mul) - 2 * a * T / n * self.SIG
        Psi = a * la.solve(denom, np.eye(n))

        # Quadratic form in g: g.T @ Psi @ g
        quad_form = (self.g.T @ Psi @ self.g)[0, 0] * T / n

        # Final Fourier-Laplace value
        value = np.exp(u * (np.log(moneyness) + r * T) + quad_form) / np.sqrt(det_val)

        return value, det_val

    def _discontinuous_cf_det(self, n, u_array, w_array, r, T, moneyness) -> tuple[np.ndarray[complex], np.ndarray[complex]]:
        """
        Computes the discontinuous Fourier-Laplace transform for arrays of complex parameters
        u and w using the exponential decay kernel.
        """
        values = []
        dets = []
        for u, w in zip(u_array, w_array):
            val, det_val = self._rss_discontinuous_eval(u, w, r, n, T, moneyness)
            values.append(val)
            dets.append(det_val)
        return np.array(values, dtype=complex), np.array(dets, dtype=complex)
    
    def _rss_cf(self, n, u_array, w_array, r, T, moneyness) -> tuple[np.ndarray[complex], np.ndarray[complex]]:
        """
        Computes the Fourier-Laplace transform of the log price and integrated variance and the Fredholm determinant
        of tilde(Σ_0) based on the Volterra Stein-Stein model with exponential kernel for arrays of complex parameters 
        u and w, using the determinant formula combined with the Lipschitz-based algorithm.
        """
        # Compute the discontinuous characteristic function
        if self.g is None or self.KK is None or self.SIG is None:
            self._rss_g_K_SIG_det(n, T)
        fourier_laplace_values, det_values = self._discontinuous_cf_det(n, u_array, w_array, r, T, moneyness)

        # Apply the rotation count algorithm in its naive version
        pc=np.array([polar(i) for i in det_values])
        arg = pc[:,1]
        bad_ind = np.where(np.abs(arg[1:]-arg[:-1])>5)[0]
        for i in bad_ind.tolist():
            fourier_laplace_values[i+1:] = - fourier_laplace_values[i+1:]
        
        # Ensure the phi(0,0) has a positive real part
        max_abs_idx = np.argmax(np.abs(fourier_laplace_values))
        if np.abs(fourier_laplace_values[max_abs_idx]) < 0:
            fourier_laplace_values = -fourier_laplace_values

        return fourier_laplace_values
    
    def _chi(self, c, d, a, b, k):
        """
        优化版本的chi_k函数，计算函数 g(y) = e^y 在区间 [c,d] 上的余弦系数
        
        参数:
        c, d: 积分区间端点
        a, b: 整体积分区间端点
        k: 余弦项的阶数
        
        返回:
        χ_k(c,d) 的解析解
        """
        k = np.asarray(k)

        # 标量情况
        # 处理 k=0 的特殊情况
        mask_k0 = (k == 0)
        mask_k_nonzero = ~mask_k0

        result = np.zeros_like(k, dtype=float)

        if np.any(mask_k_nonzero):
            # k ≠ 0 的情况
            omega = k[mask_k_nonzero] * np.pi / (b - a)
            omega_sq = omega**2

            cos_term_d = np.cos(omega * (d - a))
            cos_term_c = np.cos(omega * (c - a))
            sin_term_d = np.sin(omega * (d - a))
            sin_term_c = np.sin(omega * (c - a))

            denominator = 1 + omega_sq

            result[mask_k_nonzero] = (cos_term_d * np.exp(d) - cos_term_c * np.exp(c) +
                                        omega * sin_term_d * np.exp(d) -
                                        omega * sin_term_c * np.exp(c)) / denominator

        if np.any(mask_k0):
            # k = 0 的情况
            result[mask_k0] = np.exp(d) - np.exp(c)

        return result

    def _psi(self, c, d, a, b, k):
        """
        优化版本的psi_k函数，计算函数 g(y) = 1 在区间 [c,d] 上的余弦系数
        
        参数:
        c, d: 积分区间端点
        a, b: 整体积分区间端点
        k: 余弦项的阶数
        
        返回:
        ψ_k(c,d) 的解析解
        """
        k = np.asarray(k)

        # 标量情况
        # 处理 k=0 的特殊情况
        mask_k0 = (k == 0)
        mask_k_nonzero = ~mask_k0
        
        result = np.zeros_like(k, dtype=float)
        
        if np.any(mask_k_nonzero):
            # k ≠ 0 的情况
            omega = k[mask_k_nonzero] * np.pi / (b - a)
            sin_term_d = np.sin(omega * (d - a))
            sin_term_c = np.sin(omega * (c - a))
            
            result[mask_k_nonzero] = (b - a) / (k[mask_k_nonzero] * np.pi) * (sin_term_d - sin_term_c)
        
        if np.any(mask_k0):
            # k = 0 的情况
            result[mask_k0] = d - c
        
        return result
    
    def _cal_integral_bounds(self, L: float, r:float , tau: float, h: float = 1e-4) -> tuple[float, float]:
        """
        计算积分区间 [a, b] 的上下限
        
        参数:
        L: 积分区间倍数
        tau: 到期时间
        h: 数值微分的步长（默认值调整为1e-4以提高稳定性）
        
        返回:
        a, b: 积分区间端点
        """
        # 计算均值和方差
        u = np.array([-2*h, -h, 0, h, 2*h]) * 1j
        w = np.zeros_like(u)
        cf_values = self._rss_cf(252, u, w, r, tau, 1.0)

        # 使用更稳定的数值微分方法计算矩
        c1 = np.real((cf_values[3] - cf_values[1]) / (4j*h))
        
        # 二阶矩：使用中心差分公式
        c2 = -np.real((cf_values[1] - 2*cf_values[2] + cf_values[3]) / (h**2))
        
        # # 四阶矩：使用更稳定的计算方法
        # d2CF_h = -np.real((cf_values[4] - 2*cf_values[3] + cf_values[2]) / (h**2))
        # d2CF_negh = -np.real((cf_values[2] - 2*cf_values[1] + cf_values[0]) / (h**2))

        # d3CF_0 = (d2CF_h - c2) / h
        # d3CF_negh = (c2 - d2CF_negh) / h

        # c4 = (d3CF_0 - d3CF_negh) / h
        
        a = c1 - L * np.sqrt(c2)
        b = c1 + L * np.sqrt(c2)
        
        return a, b

    def call(self, S: float, K: Union[float, np.ndarray], r: float, tau: float,
                   N: int = 256, L: int = 20) -> np.ndarray:
        """
        使用优化COS方法计算Volterra Stein-Stein模型下的欧式看涨期权价格（多个行权价）
        
        参数:
            S: 标的资产价格（标量）
            K: 执行价格（标量或数组，形状为(M,)）
            r: 无风险利率
            tau: 到期时间
            N: 余弦展开项数（默认256）
            L: 积分区间倍数（默认10）
            
        返回:
            看涨期权价格（与K同形状的数组）
        """
        # 确保K是numpy数组
        K = np.asarray(K)
        original_shape = K.shape
        K_flat = K.flatten()
        
        # 计算积分区间 [a, b]
        a,b = self._cal_integral_bounds(L, r, tau)
        b_minus_a = b - a
        
        # 计算phi_{levy}
        k = np.arange(N)
        omega = k * np.pi / b_minus_a
        
        # 使用Volterra Stein-Stein特征函数计算phi_levy
        # 这里使用moneyness=1.0来计算特征函数
        moneyness = 1.0
        n = 252  # 离散化步数，一年252个交易日
        u_array = omega * 1j
        w_array = np.zeros_like(omega, dtype=complex)  # 对于对数价格特征函数，w=0
        
        phi_levy = self._rss_cf(n, u_array, w_array, r, tau, moneyness)
        phi_levy[0] *= 0.5  # 零频率项需要特殊缩放
        
        # 看涨期权的Uk系数
        Uk_call = (self._chi(0, b, a, b, k) - self._psi(0, b, a, b, k)) * 2 / b_minus_a
        
        Ck = phi_levy * Uk_call  # (N, )
        
        # 计算Strike-wise项
        x_vec = np.log(S / K_flat)
        strike_bias = np.exp(1j * omega[np.newaxis, :] * (x_vec[:, np.newaxis] - a))
        
        call = K_flat * np.exp(-r * tau) * np.real(strike_bias @ Ck)
        
        return call.reshape(original_shape)

    def put(self, S: float, K: Union[float, np.ndarray], r: float, tau: float,
                  N: int = 256, L: int = 20) -> np.ndarray:
        """
        使用优化COS方法计算Volterra Stein-Stein模型下的欧式看跌期权价格（多个行权价）
        
        参数:
            S: 标的资产价格（标量）
            K: 执行价格（标量或数组，形状为(M,)）
            r: 无风险利率
            tau: 到期时间
            N: 余弦展开项数（默认256）
            L: 积分区间倍数（默认10）
            
        返回:
            看跌期权价格（与K同形状的数组）
        """
        # 确保K是numpy数组
        K = np.asarray(K)
        original_shape = K.shape
        K_flat = K.flatten()
        
        a,b = self._cal_integral_bounds(L, r, tau)
        b_minus_a = b - a
        
        # 计算phi_{levy}
        k = np.arange(N)
        omega = k * np.pi / b_minus_a
        
        # 使用Volterra Stein-Stein特征函数计算phi_levy
        # 这里使用moneyness=1.0来计算特征函数
        moneyness = 1.0
        n = 252  # 离散化步数，一年252个交易日
        u_array = omega * 1j
        w_array = np.zeros_like(omega, dtype=complex)  # 对于对数价格特征函数，w=0
        
        phi_levy = self._rss_cf(n, u_array, w_array, r, tau, moneyness)
        phi_levy[0] *= 0.5  # 零频率项需要特殊缩放
        
        # 看跌期权的Uk系数（与看涨期权不同）
        Uk_put = (self._psi(a, 0, a, b, k) - self._chi(a, 0, a, b, k)) * 2 / b_minus_a
        
        Ck = phi_levy * Uk_put  # (N, )
        
        # 计算Strike-wise项
        x_vec = np.log(S / K_flat)
        strike_bias = np.exp(1j * omega[np.newaxis, :] * (x_vec[:, np.newaxis] - a))
        
        put = K_flat * np.exp(-r * tau) * np.real(strike_bias @ Ck)
        
        return put.reshape(original_shape)

    def price(self,
          S0: float,
          strike: dict[str, np.ndarray | list],
          r: float,
          q: float,
          tau: float,
          N: int = 256,
          L: int = 20) -> dict[str, np.ndarray | list]:
        """
        计算指定行权价看涨和看跌期权价格
        优化版本：计算call和put行权价并集的cf，再分别算出call和put的Uk和strike_bias
        
        参数:
            S0 (float): 初始资产价格
            strike (dict): 执行价字典 {'call': array, 'put': array}
            r (float): 无风险利率
            q (float): 股息率
            tau (float): 到期时间
            N (int): 余弦展开项数（默认256）
            L (int): 积分区间倍数（默认10）
            
        返回:
            dict: {'call': 看涨期权价格数组, 'put': 看跌期权价格数组}
        """
        assert list(strike.keys()) == ['call', 'put']
        
        # 合并所有行权价并去重
        all_strikes = np.unique(np.concatenate([strike['call'], strike['put']]))
        all_strikes_flat = all_strikes.flatten()
        
        # 计算积分区间 [a, b]
        a,b = self._cal_integral_bounds(L, r, tau)
        b_minus_a = b - a
        
        # 计算phi_{levy} - 一次性计算特征函数
        k = np.arange(N)
        omega = k * np.pi / b_minus_a
        
        # 使用Volterra Stein-Stein特征函数计算phi_levy
        moneyness = 1.0
        n = 252  # 离散化步数，一年252个交易日
        u_array = omega * 1j
        w_array = np.zeros_like(omega, dtype=complex)  # 对于对数价格特征函数，w=0
        
        phi_levy = self._rss_cf(n, u_array, w_array, r, tau, moneyness)
        phi_levy[0] *= 0.5  # 零频率项需要特殊缩放
        
        # 计算call和put的Uk系数
        Uk_call = (self._chi(0, b, a, b, k) - self._psi(0, b, a, b, k)) * 2 / b_minus_a
        Uk_put = (self._psi(a, 0, a, b, k) - self._chi(a, 0, a, b, k)) * 2 / b_minus_a
        
        # 计算call和put的Ck系数
        Ck_call = phi_levy * Uk_call  # (N, )
        Ck_put = phi_levy * Uk_put   # (N, )
        
        # 计算Strike-wise项
        x_vec = np.log(S0 / all_strikes_flat)
        strike_bias = np.exp(1j * omega[np.newaxis, :] * (x_vec[:, np.newaxis] - a))
        
        # 计算所有行权价的call和put价格
        discount_factor = np.exp(-r * tau)
        
        call_prices_all = all_strikes_flat * discount_factor * np.real(strike_bias @ Ck_call)
        put_prices_all = all_strikes_flat * discount_factor * np.real(strike_bias @ Ck_put)
        
        # 创建映射字典，将行权价映射到对应的价格
        price_map_call = dict(zip(all_strikes_flat, call_prices_all))
        price_map_put = dict(zip(all_strikes_flat, put_prices_all))
        
        # 获取指定行权价的价格
        call_prices = np.array([price_map_call[k] for k in strike['call']])
        put_prices = np.array([price_map_put[k] for k in strike['put']])
        
        return {'call': call_prices, 'put': put_prices}


if __name__ == "__main__":
    # 参数设置
    param = VSSParam(
        kappa=8.9e-5,      # Mean reversion speed
        nu=0.176,         # Volatility of volatility
        rho=-0.704,       # Correlation between Brownian motions
        theta= -0.044,     # Long-term variance
        X_0=0.113,        # Initial volatility (sqrt(variance))
        H=0.279           # Hurst index for Rough Stein-Stein model
    )
    pricer = VSSPricerCOS(param)
    
    S0 = 100.0
    r = 0.03
    q = 0.0
    tau = 1.0

    # 测试多个行权价
    K_array = np.linspace(80, 120, 9)  # 行权价从80到120，共9个点
    
    print(f"\nVolterra Stein-Stein模型COS定价测试:")
    print(f"参数: kappa={param.kappa}, nu={param.nu}, rho={param.rho}, theta={param.theta}, X_0={param.X_0}, H={param.H}")
    print(f"市场参数: S0={S0}, r={r}, tau={tau}")
    print(f"行权价: {K_array}")
    
    # 测试price方法
    strike_dict = {'call': K_array, 'put': K_array}
    prices = pricer.price(S0, strike_dict, r, q, tau)
    put_by_parity = prices['call'] + K_array * np.exp(-r * tau) - S0  # 通过看涨-看跌平价计算看跌价格
    call_by_parity = prices['put'] + S0 - K_array * np.exp(-r * tau)  # 通过看跌-看涨平价计算看涨价格
    parity = prices['call'] + K_array * np.exp(-r * tau) - prices['put'] - S0
    # parity = prices['call'] + K_array - prices['put'] - S0


    print("行权价\t看涨期权价格\t看跌期权价格\t看涨平价\t看跌平价\t平价误差")
    for i, k in enumerate(K_array):
        print(f"{k}\t{prices['call'][i]:.6f}\t{prices['put'][i]:.6f}\t{call_by_parity[i]:.6f}\t{put_by_parity[i]:.6f}\t{parity[i]:.6e}")
    
    # # # 验证看涨-看跌平价
    # parity_check = prices['call'] + K_array * np.exp(-r * tau) - prices['put'] - S0
    # print(f"\n看涨-看跌平价验证: {np.abs(parity_check)}")
    
    print(f"\n=== Volterra Stein-Stein COS定价器测试完成 ===")
