import numpy as np
from scipy.optimize import newton, bisect
from scipy.stats import norm
from typing import Literal

def BS_price(St: float|np.ndarray, K: float, tau: float, r: float, q: float, 
             sigma: float, option: Literal['call', 'put']) -> float|np.ndarray:
    """Black-Scholes期权定价公式
    参数:
        St: 标的资产当前价格
        K: 行权价
        tau: 到期时间(年)
        r: 无风险利率
        q: 股息率
        sigma: 波动率
        option: 期权类型('call'或'put')
    返回:
        期权理论价格
    """
    # 计算d1和d2参数
    d1 = (np.log(St / K) + (r + 0.5 * sigma ** 2 - q) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    # 计算看涨期权价格
    call = St * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    
    # 根据期权类型返回相应价格
    if option == 'call':
        return call
    else:  # 看跌期权价格通过看涨期权平价公式计算
        return call - St * np.exp(-q * tau) + K * np.exp(-r * tau)

def BS_vega(sigma, S, K, r, tau, q=0):
    """
    计算Black-Scholes模型的vega（波动率敏感度）
    
    参数:
    S: 标的资产价格
    K: 执行价格
    r: 无风险利率
    sigma: 波动率
    tau: 到期时间（年）
    q: 股息率（默认0）
    
    返回:
    vega值（期权价格对波动率的敏感度）
    """
    # 计算d1参数
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    
    # 计算vega：S * sqrt(tau) * φ(d1) * e^(-q*tau)
    vega = S * np.sqrt(tau) * norm.pdf(d1) * np.exp(-q * tau)
    
    return vega

def loss_func(sigma, *args):
        """函数值：BS价格 - 市场价格"""
        St, K, tau, r, q, option_type, priceM = args 
        return BS_price(St, K, tau, r, q, sigma, option_type) - priceM

def IV_newton(priceM: float, St: float, K: float, tau: float, 
           r: float, q: float, option: Literal['call', 'put']) -> float:
    """计算隐含波动率（使用牛顿法，提供导数函数）
    
    参数:
        priceM: 期权市场价格
        St: 标的资产当前价格
        K: 行权价
        tau: 到期时间(年)
        r: 无风险利率
        q: 股息率
        option: 期权类型('call'或'put')
    
    返回:
        隐含波动率（使用牛顿法求解，提供导数函数）
    """
    
    def func(sigma):
        """函数值：BS价格 - 市场价格"""
        return BS_price(St, K, tau, r, q, sigma, option) - priceM
    
    def fprime(sigma):
        """导数值：vega值"""
        return BS_vega(sigma, St, K, r, tau, q)
    
    # 使用牛顿法求解，提供导数函数
    try:
        IV = newton(func, x0=1.0, fprime=fprime)
    except RuntimeError:
        return 0
    
    return IV

def IV_bisect(priceM: float, St: float, K: float, tau: float, 
           r: float, q: float, option: Literal['call', 'put']) -> float:
    return bisect(loss_func, -1, 5, args=(St, K, tau, r, q, option, priceM))


if __name__ == "__main__":
    S = 3684.727
    K = 3300.0
    r = 0.01801989338219163
    tau = 0.0411522633744856
    call_market = 377.0
    option_type = 'call'
    q = 0
    # 使用带导数的牛顿法计算

    import matplotlib.pyplot as plt
    sigma = np.linspace(0, 1, 100)
    plt.plot(sigma, loss_func(sigma, S, K, tau, r, q, option_type, call_market))
    plt.show()
    # print(f"{IV:.6f}")
    