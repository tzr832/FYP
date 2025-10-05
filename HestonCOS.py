import numpy as np
from dataclasses import dataclass
from typing import Union


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

class HestonPricerCOS:
    """
    Heston模型期权定价器（COS方法）
    
    使用余弦展开(COS)方法计算Heston模型下的欧式期权价格
    """
    def __init__(self, param: HestonParam):
        """
        初始化定价器
        
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
                            rho=self.rho)
        else:
            return (self.v0, self.theta, self.kappa, self.sigmav, self.rho)

    def _heston_cf(self, u, tau, S, K, v0, r, params):
        """
        优化版本的Heston模型特征函数（对数价格）- 完全向量化版本
        
        参数:
        u: 特征函数的参数（复数或实数，形状为(N,)）
        tau: 时间
        S: 标的资产价格
        K: 执行价格（标量）
        v0: 初始方差
        r: 无风险利率
        params: Heston模型参数元组 (kappa, theta, sigma, rho)
        
        返回:
        特征函数值 phi(u) = E[exp(iu ln(S_t/K))]，形状与u相同
        """
        # 确保输入为numpy数组
        u = np.asarray(u)
        
        # 解包Heston参数
        kappa, theta, sigma, rho = params
        
        # 特征函数公式: φ(u) = exp(iu ln(S0/K) + iu(r - 0.5*theta)tau - A + (2*kappa*theta/sigma^2)*D)
        # helper quantities (Cui eqs. (11a,b))
        xi = kappa - sigma * rho * 1j * u
        d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
        
        # avoid overflow in hyperbolic functions
        sinh = np.sinh(d * tau / 2.0)
        cosh = np.cosh(d * tau / 2.0)
        
        # Cui eq. (15): A1, A2, B
        A1 = (u**2 + 1j * u) * sinh
        A2 = d / v0 * cosh + xi / v0 * sinh
        A = A1 / A2
        
        # Cui eq. (17b): D
        D = np.log(d / v0) + (kappa - d) * tau * 0.5 \
            - np.log((d + xi)/(2*v0) + (d - xi)/(2*v0) * np.exp(-d * tau))
        
        # Cui eq. (18): phi(u)
        log_term = np.log(S) - np.log(K) + r * tau
        phi = np.exp(1j * u * log_term
                     - (kappa * theta * rho * tau * 1j * u) / sigma
                     - A
                     + (2 * kappa * theta / sigma**2) * D)
        
        return phi

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

    def call(self, S: float, K: Union[float, np.ndarray], r: float, tau: float, 
                   N: int = 256, L: int = 10) -> np.ndarray:
        """
        使用优化COS方法计算Heston模型下的欧式看涨期权价格（多个行权价）
        
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
        
        # 确定积分区间 - 使用统一的区间（优化方法）
        # 使用对数价格的均值和方差近似
        mu = r - 0.5 * self.theta + np.log(S)
        integ_radius = np.sqrt(max((self.theta, self.v0)) * tau)  # 使用长期方差作为近似
        
        # 计算统一的积分区间[a, b]
        L *= 1.5 # 扩大积分区间以应对Heston模型的肥尾特性
        a = (mu - np.log(K_flat.max())) * tau - L * integ_radius 
        b = (mu - np.log(K_flat.min())) * tau + L * integ_radius 
        b_minus_a = b - a
        
        # 计算phi_{levy}
        k = np.arange(N)
        omega = k * np.pi / b_minus_a
        
        # 使用K=S计算phi_levy（优化方法）
        params = (self.kappa, self.theta, self.sigmav, self.rho)
        phi_levy = self._heston_cf(omega, tau, S, S, self.v0, r, params)
        phi_levy[0] *= 0.5 
        
        # 看涨期权的Uk系数
        Uk_call = (self._chi(0, b, a, b, k) - self._psi(0, b, a, b, k)) * 2 / b_minus_a
        
        Ck = phi_levy * Uk_call  # (N, )
        
        # 计算Strike-wise项
        x_vec = np.log(S / K_flat)
        # strike_bias = np.exp(1j * omega.reshape((1, -1)) * (x_vec.reshape((-1, 1)) - a))  # (M, N)
        strike_bias = np.exp(1j * omega[np.newaxis, :] * (x_vec[:, np.newaxis] - a))
        
        call = K_flat * np.exp(-r * tau) * np.real(strike_bias @ Ck)
        
        return call.reshape(original_shape)

    def put(self, S: float, K: Union[float, np.ndarray], r: float, tau: float, 
                  N: int = 256, L: int = 10) -> np.ndarray:
        """
        使用优化COS方法计算Heston模型下的欧式看跌期权价格（多个行权价）
        
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
        
        # 确定积分区间 - 使用统一的区间（优化方法）
        # 使用对数价格的均值和方差近似
        mu = r - 0.5 * self.theta + np.log(S)
        integ_radius = np.sqrt(max((self.theta, self.v0)) * tau)  # 使用长期方差作为近似
        
        # 计算统一的积分区间[a, b]
        L *= 1.5 # 扩大积分区间以应对Heston模型的肥尾特性
        a = (mu - np.log(K_flat.max())) * tau - L * integ_radius 
        b = (mu - np.log(K_flat.min())) * tau + L * integ_radius
        b_minus_a = b - a
        
        # 计算phi_{levy}
        k = np.arange(N)
        omega = k * np.pi / b_minus_a
        
        # 使用K=S计算phi_levy（优化方法）
        params = (self.kappa, self.theta, self.sigmav, self.rho)
        phi_levy = self._heston_cf(omega, tau, S, S, self.v0, r, params)
        phi_levy[0] *= 0.5 
        
        # 看跌期权的Uk系数（与看涨期权不同）
        Uk_put = (self._psi(a, 0, a, b, k) - self._chi(a, 0, a, b, k)) * 2 / b_minus_a
        
        Ck = phi_levy * Uk_put  # (N, )
        
        # 计算Strike-wise项
        x_vec = np.log(S / K_flat)
        # strike_bias = np.exp(1j * omega.reshape((1, -1)) * (x_vec.reshape((-1, 1)) - a))  # (M, N)
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
        
        # 考虑股息率调整（如果q不为0）
        if q != 0:
            call_prices_all = call_prices_all * np.exp(-q * tau)
        
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

            
if __name__ == "__main__":   
    # 参数设置
    param = HestonParam(v0=0.04, theta=0.04, kappa=2.0, rho=-0.7, sigmav=0.1)
    pricer = HestonPricerCOS(param)
    
    S0 = 100.0
    r = 0.05
    q = 0.0
    tau = 1.0

    # 测试多个行权价
    K_array = np.array([80, 90, 100, 110, 120])
    call_prices = pricer.call(S0, K_array, r, tau)
    put_prices = pricer.put(S0, K_array, r, tau)
    
    print(f"\n多个行权价测试:")
    print("行权价\t看涨期权价格\t看跌期权价格")
    for i, k in enumerate(K_array):
        print(f"{k}\t{call_prices[i]:.6f}\t\t{put_prices[i]:.6f}")
    
    # 测试price方法（与HestonPricer.price结构相仿）
    strike_dict = {'call': K_array, 'put': K_array}
    prices = pricer.price(S0, strike_dict, r, q, tau)
    
    print(f"\nprice方法测试:")
    print("行权价\t看涨期权价格\t看跌期权价格")
    for i, k in enumerate(K_array):
        print(f"{k}\t{prices['call'][i]:.6f}\t\t{prices['put'][i]:.6f}")
    
    # 验证看涨-看跌平价
    parity_check = call_prices + K_array * np.exp(-r * tau) - put_prices - S0
    print(f"\n看涨-看跌平价验证 (最大误差): {np.max(np.abs(parity_check)):.6e}")
    
    print(f"\n=== HestonModelCOS 测试完成 ===")
    
