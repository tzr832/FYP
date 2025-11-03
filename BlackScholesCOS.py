
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union, Optional
from scipy.stats import norm
from dataclasses import dataclass

plt.rcParams['font.family'] = 'SimHei'

DEBUG = False

@dataclass
class BSParam:
    """Black-Scholes 模型参数类"""
    sigma: float  # 波动率


class BSPricerCOS:
    """
    Black-Scholes 期权定价器，支持COS方法和传统公式方法
    """
    
    def __init__(self, param: Optional[BSParam] = None):
        """
        初始化定价器
        
        参数:
        param: Black-Scholes 模型参数，如果为None则需要在调用方法时单独指定
        """
        self.param = param
    
    def set_params(self, param: BSParam):
        """设置模型参数"""
        self.param = param
    
    def get_param(self, want_class=True):
        """获取模型参数"""
        if want_class:
            return self.param
        else:
            return self.param.sigma
    
    @staticmethod
    def gbm_cf(u, x, r, sigma, tau):
        """
        计算几何布朗运动的特征函数（对数价格）- 完全向量化版本
        
        参数:
        u: 特征函数的参数（复数或实数，形状为(N,)）
        x: log-moneyness (float or array), which is equal to log(S/K)
        r: 无风险利率
        sigma: 波动率
        tau: 时间
        
        返回:
        特征函数值 phi(u) = E[exp(iu ln(S_t/K))]，形状与u相同
        """
        # 确保输入为numpy数组
        u = np.asarray(u)
        
        # 特征函数公式: φ(u) = exp(iu ln(S0/K) + iu(mu - sigma²/2)tau - (u² sigma² tau)/2)
        mu = r - 0.5 * sigma**2
        log_term = x + mu  
        
        # 广播计算特征函数矩阵
        result = np.exp((1j * u * log_term - 0.5 * (u**2) * (sigma**2)) * tau)

        if DEBUG:
            print(f"GBM CF computed for u with shape {u.shape}, result shape: {result.shape}")
            plt.plot(u, result.real, label='Real Part')
            plt.plot(u, result.imag, label='Imaginary Part')
            plt.legend()
            plt.grid()
            plt.show()
        return result  
    
    @staticmethod
    def chi(c, d, a, b, k):
        """
        计算函数 g(y) = e^y 在区间 [c,d] 上的余弦系数

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
    
    @staticmethod
    def psi(c, d, a, b, k):
        """
        计算函数 g(y) = 1 在区间 [c,d] 上的余弦系数
        
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
    
    def call(self, S: float, K: Union[float, np.ndarray], r: float, tau: float,
             N: int = 256, L: int = 10):
        """
        使用COS方法计算看涨期权价格
        
        参数:
        S: 标的资产价格
        K: 行权价（单个或多个）
        r: 无风险利率
        tau: 到期时间
        N: COS展开项数
        L: 积分范围乘数
        
        返回:
        看涨期权价格
        """
        # 获取波动率参数
        if self.param is not None:
            sigma = self.param.sigma
        else:
            raise ValueError("必须设置波动率参数")
        
        K = np.asarray(K)

        # 确定积分区间
        mu = r - 0.5*sigma**2 + np.log(S)
        integ_radius = np.sqrt(sigma**2 * tau)
        a = mu * tau - L * integ_radius - np.log(K.max())
        b = mu * tau + L * integ_radius - np.log(K.min())
        b_minus_a = b - a

        # 计算phi_{levy}
        k = np.arange(N)
        omega = k * np.pi / b_minus_a
        phi_levy = self.gbm_cf(omega, 0, r, sigma, tau) # (N, )
        phi_levy[0] *= 0.5 

        # 看涨期权的Uk系数
        Uk_call = (self.chi(0, b, a, b, k) - self.psi(0, b, a, b, k)) * 2 / b_minus_a # (N, )

        Ck = phi_levy * Uk_call #  (N, )
        # 计算Strike-wise项
        x_vec = np.log(S / K)
        strike_bias = np.exp(1j * omega.reshape((1, -1)) * (x_vec.reshape((-1, 1)) - a)) # (M, N)

        call = K * np.exp(-r*tau) * np.real(strike_bias @ Ck)   
        return call
    
    def put(self, S: float, K: Union[float, np.ndarray], r: float, tau: float,
            N: int = 256, L: int = 10):
        """
        使用COS方法计算看跌期权价格
        
        参数:
        S: 标的资产价格
        K: 行权价（单个或多个）
        r: 无风险利率
        tau: 到期时间
        N: COS展开项数
        L: 积分范围乘数
        
        返回:
        看跌期权价格
        """
        # 获取波动率参数
        if self.param is not None:
            sigma = self.param.sigma
        else:
            raise ValueError("必须设置波动率参数")
        
        K = np.asarray(K)

        # 确定积分区间（与call相同）
        mu = r - 0.5*sigma**2 + np.log(S)
        integ_radius = np.sqrt(sigma**2 * tau)
        a = mu * tau - L * integ_radius - np.log(K.max())
        b = mu * tau + L * integ_radius - np.log(K.min())
        b_minus_a = b - a

        # 计算phi_{levy}
        k = np.arange(N)
        omega = k * np.pi / b_minus_a
        phi_levy = self.gbm_cf(omega, 0, r, sigma, tau) # (N, )
        phi_levy[0] *= 0.5 

        # 看跌期权的Uk系数（与看涨期权不同）
        Uk_put = (self.psi(a, 0, a, b, k) - self.chi(a, 0, a, b, k)) * 2 / b_minus_a # (N, )

        Ck = phi_levy * Uk_put #  (N, )
        # 计算Strike-wise项
        x_vec = np.log(S / K)
        strike_bias = np.exp(1j * omega.reshape((1, -1)) * (x_vec.reshape((-1, 1)) - a)) # (M, N)

        put = K * np.exp(-r*tau) * np.real(strike_bias @ Ck)   
        return put
    
    def call_formula(self, S: float, K: Union[float, np.ndarray], r: float, tau: float):
        """
        使用传统Black-Scholes公式计算看涨期权价格
        
        参数:
        S: 标的资产价格
        K: 行权价（单个或多个）
        r: 无风险利率
        tau: 到期时间
        
        返回:
        看涨期权价格
        """
        # 获取波动率参数
        if self.param is not None:
            sigma = self.param.sigma
        else:
            raise ValueError("必须设置波动率参数")
        
        K = np.asarray(K)
        
        K = np.asarray(K)
        
        # Black-Scholes公式
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        
        call = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        return call
    
    def put_formula(self, S: float, K: Union[float, np.ndarray], r: float, tau: float):
        """
        使用传统Black-Scholes公式计算看跌期权价格
        
        参数:
        S: 标的资产价格
        K: 行权价（单个或多个）
        r: 无风险利率
        tau: 到期时间
        
        返回:
        看跌期权价格
        """
        # 获取波动率参数
        if self.param is not None:
            sigma = self.param.sigma
        else:
            raise ValueError("必须设置波动率参数")
        
        K = np.asarray(K)
        
        K = np.asarray(K)
        
        # Black-Scholes公式
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        
        put = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put
    
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
        优化版本：使用call-put平价关系，只计算call option价格，然后推导put option价格
        
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
        
        # 使用优化COS方法计算所有行权价的call价格
        call_prices_all = self.call(S0, all_strikes_flat, r, tau, N, L)
        
        # 使用call-put平价关系计算put option价格
        # call-put平价: C + K*e^(-rT) = P + S*e^(-qT)
        # 所以 P = C + K*e^(-rT) - S*e^(-qT)
        discount_factor = np.exp(-r * tau)
        spot_adjusted = S0 * np.exp(-q * tau) if q != 0 else S0
        
        put_prices_all = call_prices_all + all_strikes_flat * discount_factor - spot_adjusted
        
        # 直接返回指定行权价的价格（无需插值）
        # 创建映射字典，将行权价映射到对应的价格
        price_map_call = dict(zip(all_strikes, call_prices_all))
        price_map_put = dict(zip(all_strikes, put_prices_all))
        
        # 获取指定行权价的价格
        call_prices = np.array([price_map_call[k] for k in strike['call']])
        put_prices = np.array([price_map_put[k] for k in strike['put']])
        
        return {'call': call_prices, 'put': put_prices}

    def price_without_parity(self,
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
        
        # 确定积分区间（与call相同）
        mu = r - 0.5*sigma**2 + np.log(S0)
        integ_radius = np.sqrt(sigma**2 * tau)
        a = mu * tau - L * integ_radius - np.log(all_strikes_flat.max())
        b = mu * tau + L * integ_radius - np.log(all_strikes_flat.min())
        b_minus_a = b - a

        # 计算phi_{levy}
        k = np.arange(N)
        omega = k * np.pi / b_minus_a
        phi_levy = self.gbm_cf(omega, 0, r, sigma, tau) # (N, )
        phi_levy[0] *= 0.5
        
        phi_levy = self.gbm_cf(omega, 0, r, sigma, tau)
        phi_levy[0] *= 0.5  # 零频率项需要特殊缩放
        
        # 计算call和put的Uk系数
        Uk_call = (self.chi(0, b, a, b, k) - self.psi(0, b, a, b, k)) * 2 / b_minus_a
        Uk_put = (self.psi(a, 0, a, b, k) - self.chi(a, 0, a, b, k)) * 2 / b_minus_a
        
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


# 保留原有的函数接口以保持向后兼容性
def gbm_cf(u, x, r, sigma, tau):
    """计算几何布朗运动的特征函数"""
    return BSPricerCOS.gbm_cf(u, x, r, sigma, tau)

def BS_call_multi(S: float, K: Union[float, np.ndarray], r: float, sigma: float, tau: float,
                  cf: Callable = gbm_cf, N: int = 256, L: int = 10):
    """使用COS方法计算多个看涨期权价格"""
    param = BSParam(sigma=sigma)
    pricer = BSPricerCOS(param)
    return pricer.call(S, K, r, tau, N, L)

def BS_put_multi(S: float, K: Union[float, np.ndarray], r: float, sigma: float, tau: float,
                 cf: Callable = gbm_cf, N: int = 256, L: int = 10):
    """使用COS方法计算多个看跌期权价格"""
    param = BSParam(sigma=sigma)
    pricer = BSPricerCOS(param)
    return pricer.put(S, K, r, tau, N, L)

def chi(c, d, a, b, k):
    """计算函数 g(y) = e^y 在区间 [c,d] 上的余弦系数"""
    return BSPricerCOS.chi(c, d, a, b, k)

def psi(c, d, a, b, k):
    """计算函数 g(y) = 1 在区间 [c,d] 上的余弦系数"""
    return BSPricerCOS.psi(c, d, a, b, k)


if __name__ == "__main__":
    # 测试参数
    S0 = 100.0
    K_array = np.array([80, 90, 100, 110, 120])
    r = 0.05
    sigma = 0.2
    tau = 1.0
    q = 0.0

    print("=== BSPricerCOS 类测试 ===")
    print(f"参数: S0={S0}, r={r}, sigma={sigma}, tau={tau}, q={q}")
    print(f"行权价: {K_array}")
    
    # 创建定价器实例
    param = BSParam(sigma=sigma)
    pricer = BSPricerCOS(param)
    
    print("\n=== COS方法定价 ===")
    # 使用COS方法计算call和put价格
    call_prices_cos = pricer.call(S0, K_array, r, tau)
    put_prices_cos = pricer.put(S0, K_array, r, tau)
    
    print("\n行权价\tCall(COS)\tPut(COS)\tCall+PV(K)\tPut+S\t平价误差")
    for i, k in enumerate(K_array):
        call_plus_pv = call_prices_cos[i] + k * np.exp(-r * tau)
        put_plus_s = put_prices_cos[i] + S0
        parity_error = abs(call_plus_pv - put_plus_s)
        print(f"{k:.1f}\t{call_prices_cos[i]:.6f}\t{put_prices_cos[i]:.6f}\t{call_plus_pv:.6f}\t{put_plus_s:.6f}\t{parity_error:.2e}")
    
    print("\n=== 传统公式定价 ===")
    # 使用传统公式计算call和put价格
    call_prices_formula = pricer.call_formula(S0, K_array, r, tau)
    put_prices_formula = pricer.put_formula(S0, K_array, r, tau)
    
    print("\n行权价\tCall(Formula)\tPut(Formula)\tCall+PV(K)\tPut+S\t平价误差")
    for i, k in enumerate(K_array):
        call_plus_pv = call_prices_formula[i] + k * np.exp(-r * tau)
        put_plus_s = put_prices_formula[i] + S0
        parity_error = abs(call_plus_pv - put_plus_s)
        print(f"{k:.1f}\t{call_prices_formula[i]:.6f}\t{put_prices_formula[i]:.6f}\t{call_plus_pv:.6f}\t{put_plus_s:.6f}\t{parity_error:.2e}")
    
    print("\n=== 方法比较 ===")
    print("行权价\tCOS-Formula差异(Call)\tCOS-Formula差异(Put)")
    for i, k in enumerate(K_array):
        call_diff = abs(call_prices_cos[i] - call_prices_formula[i])
        put_diff = abs(put_prices_cos[i] - put_prices_formula[i])
        print(f"{k:.1f}\t{call_diff:.2e}\t\t{put_diff:.2e}")
    
    print("\n=== price方法测试 ===")
    # 测试新的price方法
    strike_dict = {'call': K_array, 'put': K_array}
    prices = pricer.price_without_parity(S0, strike_dict, r, q, tau)
    
    print("行权价\tCall(Price)\tPut(Price)\tCall+PV(K)\tPut+S\t平价误差")
    for i, k in enumerate(K_array):
        call_plus_pv = prices['call'][i] + k * np.exp(-r * tau)
        put_plus_s = prices['put'][i] + S0
        parity_error = abs(call_plus_pv - put_plus_s)
        print(f"{k:.1f}\t{prices['call'][i]:.6f}\t{prices['put'][i]:.6f}\t{call_plus_pv:.6f}\t{put_plus_s:.6f}\t{parity_error:.2e}")
    
    # 性能测试
    print(f"\n=== 性能测试 ===")
    import time
    
    # 大规模测试
    large_K = np.linspace(50, 150, 1000)
    
    start_time = time.time()
    call_large_cos = pricer.call(S0, large_K, r, tau)
    call_time_cos = time.time() - start_time
    
    start_time = time.time()
    call_large_formula = pricer.call_formula(S0, large_K, r, tau)
    call_time_formula = time.time() - start_time
    
    print(f"1000个行权价定价时间:")
    print(f"  COS方法Call: {call_time_cos:.4f}秒")
    print(f"  公式方法Call: {call_time_formula:.4f}秒")
    
    # 测试向后兼容性
    print(f"\n=== 向后兼容性测试 ===")
    call_old = BS_call_multi(S0, K_array, r, sigma, tau)
    put_old = BS_put_multi(S0, K_array, r, sigma, tau)
    print(f"原有函数接口Call价格: {call_old}")
    print(f"原有函数接口Put价格: {put_old}")
    
    print(f"\n=== BSPricerCOS 类测试完成 ===")
