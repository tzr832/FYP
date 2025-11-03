import numpy as np
from dataclasses import dataclass
from typing import Union
from scipy.special import gamma, hyp2f1
import scipy.linalg as la
import numdifftools as nd
import warnings
warnings.filterwarnings('ignore')

DEBUG = True
if DEBUG:
    import time
    import matplotlib.pyplot as plt

@dataclass
class VSSParam:
    """
    Heston模型参数数据类
    """
    kappa: float = -8.9e-5  # Mean reversion speed
    nu: float = 0.176     # Volatility of volatility
    rho: float = -0.704  # Correlation between Brownian motions
    theta: float = -0.044  # Long-term variance
    X_0: float = 0.113    # Initial variance
    H: float = 0.279      # Hurst index for Rough Stein-Stein model

    def __post_init__(self):
        if not (0 < self.H < 1):
            raise ValueError("Hurst index H must be in the interval (0, 1).")
        if not (-1 < self.rho < 1):
            raise ValueError("rho must be in the interval (-1, 1).")

class VSSPricerCOS:
    """
    Volterra Stein-Stein model option pricer using the COS method.
    """
    def __init__(self, params: VSSParam, n: int = 252):
        self.params = params
        self.KK = None
        self.SIG = None
        self.g = None
        self.n = n
    
    def set_params(self, param: VSSParam):
        self.params = param
    
    def get_params(self, return_class: bool = True) -> VSSParam|tuple:
        if not return_class:
            return (self.params.kappa, self.params.nu, self.params.rho,
                    self.params.theta, self.params.X_0, self.params.H)
        else:
            return self.params
    
    def _rss_g_K_SIG_det(self, T):
        """
        Computes the kernel matrix, covariance matrix, and adjusted input vector for the
        generalized Volterra Stein-Stein (Rough Stein-Stein) model.

        Parameters:
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

        Update values as follow:
        - g: numpy array of shape (n, 1)
            Vector of adjusted values based on the input curve for each time step.
        - self.KK: numpy array of shape (n, n)
            2D matrix representing the kernel's approximation matrix.
        - self.SIG: numpy array of shape (n, n)
            2D covariance matrix Σ_0.
        """
        alpha = self.params.H + 0.5

        # Time discretization from 0 to T
        t = np.linspace(0, T, self.n + 1)

        # Define indices for 2D matrices
        tj_1 = np.tile(t[:-1], self.n).reshape(self.n, self.n)  # Times tj excluding the final point
        ti_1 = tj_1.T  # Transpose to create a grid of ti values
        tj = np.tile(t[1:], self.n).reshape(self.n, self.n)  # Times tj excluding the initial point

        # Initialize kernel matrix self.KK
        self.KK = np.zeros((self.n, self.n))
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
        self.g = (self.params.X_0 + self.params.theta * t[:-1] ** alpha / gamma(1 + alpha))
    
    def _rss_cf(self, u, w, r, T, moneyness) -> np.ndarray:
        """
        Computes the discontinuous Fourier-Laplace transform for arrays of complex parameters
        u and w using the fBM kernal.
        """
        if self.g is None or self.KK is None or self.SIG is None:
            self._rss_g_K_SIG_det(T)
        elif self.g.size != self.n:
            self._rss_g_K_SIG_det(T)

        if u.ndim > 1:
            u = u.flatten()
        
        a = w + 0.5 * (u**2 - u)
        b = self.params.kappa + self.params.rho * self.params.nu * u

        # Compute tilde(SIG) = inv(I - b * K) * SIG * inv(I - b * K).T
        X = np.eye(self.n) - b[:,np.newaxis,np.newaxis] * self.KK[np.newaxis,:,:]
        Sigma_Xinv = np.swapaxes(la.solve(X, self.SIG.T), 1 ,2)
        SIG_tilde = la.solve(X, Sigma_Xinv)

        # Compute the determinant involved in the characteristic function
        D = np.eye(self.n) - 2 * a[:, np.newaxis, np.newaxis] * T / self.n * SIG_tilde
        det_val = la.det(D)

        # Compute matrix Psi
        denom = np.eye(self.n) - b[:,np.newaxis,np.newaxis] * self.KK_sum[np.newaxis,:,:] \
              + b[:,np.newaxis,np.newaxis]**2 * (self.KK_mul[np.newaxis,:,:]) \
              - 2 * a[:,np.newaxis,np.newaxis] * T / self.n * self.SIG[np.newaxis,:,:]
        Psi = a[:, np.newaxis, np.newaxis] * la.solve(denom, np.eye(self.n))

        # Quadratic form in g: g.T @ Psi @ g
        quad_form = np.einsum('j,ijk,k -> i', self.g, Psi, self.g) * T / self.n

        values = np.exp(u * (np.log(moneyness) + r * T) + quad_form) / np.sqrt(det_val)

        # Apply the rotation count algorithm in its naive version
        arg=np.angle(det_val)
        bad_ind = np.where(np.abs(arg[1:]-arg[:-1])>5)[0]
        for i in bad_ind.tolist():
            values[i+1:] = - values[i+1:]

        # Ensure the phi(0,0) has a positive real part
        max_abs_idx = 0 if u[0] == 0 else np.argmax(np.abs(values))
        if np.abs(values[max_abs_idx]) < 0:
            values = -values

        return values
    
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
    
    def _cal_integral_bounds(self, L: float, r:float , tau: float, h: float = 1e-6) -> tuple[float, float]:
        """
        计算积分区间 [a, b] 的上下限
        
        参数:
        L: 积分区间倍数
        tau: 到期时间
        h: 数值微分的步长（默认值调整为1e-4以提高稳定性）
        
        返回:
        a, b: 积分区间端点
        """

        # 使用更稳定的数值微分方法计算矩
        func_df1 = lambda x: -self._rss_cf(np.array([x])*1j, np.zeros(1), r, tau, 1.0).imag
        func_df2 = lambda x: -self._rss_cf(np.array([x])*1j, np.zeros(1), r, tau, 1.0).real
        func_df3 = lambda x:  self._rss_cf(np.array([x])*1j, np.zeros(1), r, tau, 1.0).imag
        func_df4 = lambda x:  self._rss_cf(np.array([x])*1j, np.zeros(1), r, tau, 1.0).real


        mu1 = nd.Derivative(func_df1, n=1)(0)
        mu2 = nd.Derivative(func_df2, n=2)(0)
        mu3 = nd.Derivative(func_df3, n=3)(0)
        mu4 = nd.Derivative(func_df4, n=4)(0)

        c1 = mu1
        c2 = mu2 - mu1**2
        c4 = mu4 - 4*mu3*mu1 - 3*mu2**2 + 12*mu2*mu1**2 - 6*mu1**4
        
        a = c1 - L * np.sqrt(c2 + np.sqrt(c4))
        b = c1 + L * np.sqrt(c2 + np.sqrt(c4))    
        
        return a, b

    def call(self, S: float, K: Union[float, np.ndarray], r: float, tau: float,
                   N: int = 256, L: int = 10) -> np.ndarray:
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
        u_array = omega * 1j
        w_array = np.zeros_like(omega, dtype=complex)  # 对于对数价格特征函数，w=0

        phi_levy = self._rss_cf(u_array, w_array, r, tau, moneyness)
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
                  N: int = 256, L: int = 10) -> np.ndarray:
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
        u_array = omega * 1j
        w_array = np.zeros_like(omega, dtype=complex)  # 对于对数价格特征函数，w=0

        phi_levy = self._rss_cf(u_array, w_array, r, tau, moneyness)
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
          L: int = 10) -> dict[str, np.ndarray | list]:
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
        u_array = omega * 1j
        w_array = np.zeros_like(omega, dtype=complex)  # 对于对数价格特征函数，w=0

        phi_levy = self._rss_cf(u_array, w_array, r, tau, moneyness)
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
        
        mask_call = np.isin(all_strikes_flat, strike['call'])
        indices = np.where(mask_call)[0]
        call_prices = call_prices_all[indices]

        mask_put = np.isin(all_strikes_flat, strike['put'])
        indices = np.where(mask_put)[0]
        put_prices = put_prices_all[indices]
        
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
    
    S0 = 25617.42
    r = 0.03
    q = 0.0
    tau = 5.0
    # 测试多个行权价
    K_array = np.array([
        13600, 13700, 13800, 13900, 14000, 14100, 14200, 14300, 14400, 14500,
        14600, 14700, 14800, 14900, 15000, 15100, 15200, 15300, 15400, 15500,
        15600, 15700, 15800, 15900, 16000, 16100, 16200, 16300, 16400, 16500,
        16600, 16700, 16800, 16900, 17000, 17100, 17200, 17300, 17400, 17500,
        17600, 17700, 17800, 17900, 18000, 18100, 18200, 18300, 18400, 18500,
        18600, 18700, 18800, 18900, 19000, 19100, 19200, 19300, 19400, 19500,
        19600, 19700, 19800, 19900, 20000, 20200, 20400, 20600, 20800, 21000,
        21200, 21400, 21600, 21800, 22000, 22200, 22400, 22600, 22800, 23000,
        23200, 23400, 23600, 23800, 24000, 24200, 24400, 24600, 24800, 25000,
        25200, 25400, 25600, 25800, 26000, 26200, 26400, 26600, 26800, 27000,
        27200, 27400, 27600, 27800, 28000, 28200, 28400, 28600, 28800, 29000,
        29200, 29400, 29600, 29800, 30000, 30200, 30400, 30600, 30800, 31000
      ])  



    
    print(f"\nVolterra Stein-Stein模型COS定价测试:")
    print(f"参数: kappa={param.kappa}, nu={param.nu}, rho={param.rho}, theta={param.theta}, X_0={param.X_0}, H={param.H}")
    print(f"市场参数: S0={S0}, r={r}, tau={tau}")
    print(f"行权价: {K_array}")
    
    # 测试price方法
    strike_dict = {'call': K_array, 'put': K_array}

    pricer.n = 32
    if DEBUG:
        start = time.time()
        prices_less = pricer.price(S0, strike_dict, r, q, tau)
        print(f"定价用时(n = {pricer.n}): {time.time() - start}")  
    else:
        prices_less = pricer.price(S0, strike_dict, r, q, tau)
    parity = prices_less['call'] + K_array * np.exp(-r * tau) - prices_less['put'] - S0
    print(f"tau = 1")
    print("行权价\t看涨期权价格\t看跌期权价格\t平价误差")
    for i, k in enumerate(K_array[::5]):
        print(f"{k}\t{prices_less['call'][i]:.6f}\t{prices_less['put'][i]:.6f}\t{parity[i]:.6e}")

    pricer.n *= 5
    # tau *= 2
    if DEBUG:
        start = time.time()
        prices_more = pricer.price(S0, strike_dict, r, q, tau)
        print(f"定价用时(n = {pricer.n}): {time.time() - start}")  
    else:
        prices_more = pricer.price(S0, strike_dict, r, q, tau)
    parity = prices_more['call'] + K_array * np.exp(-r * tau) - prices_more['put'] - S0
    # print(f"tau = 2")
    print("行权价\t看涨期权价格\t看跌期权价格\t平价误差")
    for i, k in enumerate(K_array[::5]):
        print(f"{k}\t{prices_more['call'][i]:.6f}\t{prices_more['put'][i]:.6f}\t{parity[i]:.6e}")
    
    if DEBUG:
        print("\n比较不同n值下的定价结果差异:")
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        print(f"relative error: {np.max(np.abs((prices_less['call'] - prices_more['call']) / prices_more['call'])):.6e} (call),\
               {np.max(np.abs((prices_less['put'] - prices_more['put']) / prices_more['put'])):.6e} (put)")

        axs[0].plot(K_array, prices_less['call'], label='Call n less')
        axs[0].plot(K_array, prices_more['call'], '--', label='Call n more')
        axs[0].plot(K_array, prices_less['put'], label='Put n less')
        axs[0].plot(K_array, prices_more['put'], '--', label='Put n more')
        axs[0].legend()

        axs[1].plot(K_array, np.abs((prices_less['call'] - prices_more['call'])), label='Call abs error')
        axs[1].plot(K_array, np.abs((prices_less['put'] - prices_more['put'])), label='Put abs error')
        axs[1].legend()
        plt.show()
    print(f"\n=== Volterra Stein-Stein COS定价器测试完成 ===")
