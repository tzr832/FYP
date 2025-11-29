
"""
Volterra Stein-Stein COS Pricer Base Class
包含三个实现类的共同功能
"""

import torch
import torch.nn as nn
import numpy as np
import numdifftools as nd
from typing import Union, Tuple, Dict
from abc import ABC, abstractmethod


class VSSPricerCOSBase(ABC):
    """
    Volterra Stein-Stein COS Pricer 基类
    包含所有三个实现类的共同功能
    """
    
    def __init__(self, params, n: int = 252, device: str = 'cpu'):
        """
        初始化基类
        
        Args:
            params: 模型参数对象
            n: 时间离散化点数
            device: 计算设备
        """
        self.params = params
        self.device = device
        self.n = n
        
        # 预计算矩阵
        self.KK = None
        self.SIG = None
        self.g = None
        self.KK_sum = None
        self.KK_mul = None
    
    @abstractmethod
    def _compute_kernel_matrices(self, T: torch.Tensor) -> None:
        """
        计算核矩阵、协方差矩阵和调整输入向量
        子类必须实现此方法
        """
        pass
    
    def clear_kernel_matrices(self) -> None:
        """
        清除预计算的矩阵
        """
        self.KK = None
        self.SIG = None
        self.g = None
        self.KK_sum = None
        self.KK_mul = None
    
    def _rss_cf_torch(self, u: torch.Tensor, w: torch.Tensor, r: torch.Tensor, 
                     T: torch.Tensor, moneyness: torch.Tensor) -> torch.Tensor:
        """
        使用精确的 PyTorch 操作计算特征函数
        """
        if self.g is None or self.KK is None or self.SIG is None:
            self._compute_kernel_matrices(T)
        elif self.g.shape[0] != self.n:
            self._compute_kernel_matrices(T)
        
        u = u.unsqueeze(0) if u.dim() == 0 else u
        if self.device != u.device:
            u = u.to(self.device)
            w = w.to(self.device)

        a = w + 0.5 * (u**2 - u)
        b = self.params.kappa + self.params.rho * self.params.nu * u
        
        # 计算 tilde(SIG) = inv(I - b * K) * SIG * inv(I - b * K).T
        
        batch_size = u.shape[0]
        I = torch.eye(self.n, dtype=torch.complex128, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 将矩阵转换为复数类型
        KK_complex = self.KK.to(torch.complex128)
        SIG_complex = self.SIG.to(torch.complex128)
        
        X = I - b.unsqueeze(-1).unsqueeze(-1) * KK_complex.unsqueeze(0)
        
        # 使用 LU 分解求解线性系统以获得稳定性
        Sigma_Xinv = torch.linalg.solve(X, SIG_complex.unsqueeze(0).repeat(batch_size, 1, 1))
        SIG_tilde = torch.linalg.solve(X, Sigma_Xinv.transpose(-2, -1))
        
        # 计算特征函数中涉及的矩阵行列式
        D = I - 2 * a.unsqueeze(-1).unsqueeze(-1) * T / self.n * SIG_tilde
        det_val = torch.linalg.det(D)
        
        # 计算矩阵 Psi
        KK_sum_complex = self.KK_sum.to(torch.complex128)
        KK_mul_complex = self.KK_mul.to(torch.complex128)
        
        denom = (I - b.unsqueeze(-1).unsqueeze(-1) * KK_sum_complex.unsqueeze(0) +
                b.unsqueeze(-1).unsqueeze(-1)**2 * KK_mul_complex.unsqueeze(0) -
                2 * a.unsqueeze(-1).unsqueeze(-1) * T / self.n * SIG_complex.unsqueeze(0))
        
        Psi = a.unsqueeze(-1).unsqueeze(-1) * torch.linalg.solve(denom, I)
        
        # g 的二次型: g.T @ Psi @ g
        g_complex = self.g.to(torch.complex128)
        quad_form = torch.einsum('j,ijk,k->i', g_complex, Psi, g_complex) * T / self.n
        
        values = torch.exp(u * (torch.log(moneyness) + r * T) + quad_form) / torch.sqrt(det_val)
        
        # 应用旋转计数算法（简化版本）
        arg = torch.angle(det_val)
        bad_ind = torch.where(torch.abs(arg[1:] - arg[:-1]) > 5)[0]
        for i in bad_ind.tolist():
            values[i+1:] = -values[i+1:]
        
        # 确保 phi(0,0) 有正实部
        max_abs_idx = 0 if u[0] == 0 else torch.argmax(torch.abs(values))
        if torch.abs(values[max_abs_idx]) < 0:
            values = -values
            
        return values
    
    def _chi_torch(self, c: torch.Tensor, d: torch.Tensor, a: torch.Tensor,
                  b: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        chi_k 函数 - 计算 g(y) = e^y 的余弦系数
        """
        k = k.to(torch.float64)
        
        # 处理 k=0 的特殊情况
        mask_k0 = (k == 0)
        mask_k_nonzero = ~mask_k0
        
        result = torch.zeros_like(k, dtype=torch.float64, device=self.device)
        
        if torch.any(mask_k_nonzero):
            # k ≠ 0 的情况
            omega = k[mask_k_nonzero] * torch.pi / (b - a)
            omega_sq = omega**2
            
            cos_term_d = torch.cos(omega * (d - a))
            cos_term_c = torch.cos(omega * (c - a))
            sin_term_d = torch.sin(omega * (d - a))
            sin_term_c = torch.sin(omega * (c - a))
            
            denominator = 1 + omega_sq
            
            result[mask_k_nonzero] = (cos_term_d * torch.exp(d) - cos_term_c * torch.exp(c) +
                                     omega * sin_term_d * torch.exp(d) -
                                     omega * sin_term_c * torch.exp(c)) / denominator
        
        if torch.any(mask_k0):
            # k = 0 的情况
            result[mask_k0] = torch.exp(d) - torch.exp(c)
            
        return result
    
    def _psi_torch(self, c: torch.Tensor, d: torch.Tensor, a: torch.Tensor,
                  b: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        psi_k 函数 - 计算 g(y) = 1 的余弦系数
        """
        k = k.to(torch.float64)
        
        # 处理 k=0 的特殊情况
        mask_k0 = (k == 0)
        mask_k_nonzero = ~mask_k0
        
        result = torch.zeros_like(k, dtype=torch.float64, device=self.device)
        
        if torch.any(mask_k_nonzero):
            # k ≠ 0 的情况
            omega = k[mask_k_nonzero] * torch.pi / (b - a)
            sin_term_d = torch.sin(omega * (d - a))
            sin_term_c = torch.sin(omega * (c - a))
            
            result[mask_k_nonzero] = (b - a) / (k[mask_k_nonzero] * torch.pi) * (sin_term_d - sin_term_c)
        
        if torch.any(mask_k0):
            # k = 0 的情况
            result[mask_k0] = d - c
            
        return result
    
    def _cal_integral_bounds(self, L: float, r: torch.Tensor, tau: torch.Tensor) -> tuple[float, float]:
        """
        计算积分区间 [a, b] 边界
        [a,b] = [c1 - L * sqrt(c2 + sqrt(c4)), c1 + L * sqrt(c2 + sqrt(c4))]
        """
        # 定义数值微分函数
        moneyness = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        func_df1 = lambda x: -self._rss_cf_torch(torch.tensor([x])*1j, torch.zeros(1), r, tau, moneyness).cpu().detach().numpy().imag
        func_df2 = lambda x: -self._rss_cf_torch(torch.tensor(x)*1j, torch.zeros(1), r, tau, moneyness).cpu().detach().numpy().real
        func_df3 = lambda x:  self._rss_cf_torch(torch.tensor([x])*1j, torch.zeros(1), r, tau, moneyness).cpu().detach().numpy().imag
        func_df4 = lambda x:  self._rss_cf_torch(torch.tensor(x)*1j, torch.zeros(1), r, tau, moneyness).cpu().detach().numpy().real

        # 使用数值微分计算矩
        with torch.no_grad():
            mu1 = nd.Derivative(func_df1, n=1)(0)
            mu2 = nd.Derivative(func_df2, n=2)(0)
            mu3 = nd.Derivative(func_df3, n=3)(0)
            mu4 = nd.Derivative(func_df4, n=4)(0)

        # 计算累积量
        c1 = mu1
        c2 = mu2 - mu1**2
        c4 = mu4 - 4*mu3*mu1 - 3*mu2**2 + 12*mu2*mu1**2 - 6*mu1**4
        
        a = c1 - L * np.sqrt(c2 + np.sqrt(c4))
        b = c1 + L * np.sqrt(c2 + np.sqrt(c4))    
        
        return torch.tensor(a, device=self.device), torch.tensor(b, device=self.device)

    def call_price(self, S: torch.Tensor, K: torch.Tensor, r: torch.Tensor, 
                  tau: torch.Tensor, N: int = 256, L: float = 10.0) -> torch.Tensor:
        """
        计算欧式看涨期权价格
        """
        
        # 从 NumPy 获取精确边界
        a, b = self._cal_integral_bounds(L, r, tau)
        
        original_shape = K.shape
        K_flat = K.flatten()
        b_minus_a = b - a
        
        # 使用特征函数计算 phi_levy
        k = torch.arange(N, dtype=torch.float64, device=self.device)
        omega = k * torch.pi / b_minus_a
        
        # 使用特征函数计算 phi_levy
        moneyness = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        u_array = omega * 1j
        w_array = torch.zeros_like(omega, dtype=torch.complex128, device=self.device)
        
        phi_levy = self._rss_cf_torch(u_array, w_array, r, tau, moneyness)
        phi_levy[0] *= 0.5  # 零频项需要特殊缩放
        
        # 看涨期权的 Uk 系数
        Uk_call = (self._chi_torch(torch.tensor(0.0, device=self.device), b, a, b, k) - 
                  self._psi_torch(torch.tensor(0.0, device=self.device), b, a, b, k)) * 2 / b_minus_a
        
        Ck = phi_levy * Uk_call
        
        # 计算按行权价划分的项
        x_vec = torch.log(S / K_flat)
        strike_bias = torch.exp(1j * omega.unsqueeze(0) * (x_vec.unsqueeze(1) - a))
        
        call_prices = K_flat * torch.exp(-r * tau) * torch.real(strike_bias @ Ck)
        
        return call_prices.reshape(original_shape)

    def put_price(self, S: torch.Tensor, K: torch.Tensor, r: torch.Tensor,
                 tau: torch.Tensor, N: int = 256, L: float = 10.0) -> torch.Tensor:
        """
        计算欧式看跌期权价格
        """
        S = S.to(torch.float64).to(self.device)
        K = K.to(torch.float64).to(self.device)
        r = r.to(torch.float64).to(self.device)
        tau = tau.to(torch.float64).to(self.device)
        
        original_shape = K.shape
        K_flat = K.flatten()
        
        a, b = self._cal_integral_bounds(L, r, tau)
        b_minus_a = b - a
        
        # 计算 phi_levy
        k = torch.arange(N, dtype=torch.float64, device=self.device)
        omega = k * torch.pi / b_minus_a
        
        moneyness = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        u_array = omega * 1j
        w_array = torch.zeros_like(omega, dtype=torch.complex128, device=self.device)
        
        phi_levy = self._rss_cf_torch(u_array, w_array, r, tau, moneyness)
        phi_levy[0] *= 0.5
        
        # 看跌期权的 Uk 系数
        Uk_put = (self._psi_torch(a, torch.tensor(0.0, device=self.device), a, b, k) -
                 self._chi_torch(a, torch.tensor(0.0, device=self.device), a, b, k)) * 2 / b_minus_a
        
        Ck = phi_levy * Uk_put
        
        # 计算按行权价划分的项
        x_vec = torch.log(S / K_flat)
        strike_bias = torch.exp(1j * omega.unsqueeze(0) * (x_vec.unsqueeze(1) - a))
        
        put_prices = K_flat * torch.exp(-r * tau) * torch.real(strike_bias @ Ck)
        
        return put_prices.reshape(original_shape)
    
    def price(self, S0: torch.Tensor, strike: Dict[str, torch.Tensor],
              r: torch.Tensor, q: torch.Tensor, tau: torch.Tensor,
              N: int = 256, L: float = 10.0) -> Dict[str, torch.Tensor]:
        """
        计算指定行权价的看涨和看跌期权价格
        """
        if self.g is None or self.KK is None or self.SIG is None:
            self._compute_kernel_matrices(tau)
        elif self.g.shape[0] != self.n:
            self._compute_kernel_matrices(tau)
        
        # 合并所有行权价并去重
        all_strikes = torch.unique(torch.cat([strike['call'], strike['put']]))
        all_strikes_flat = all_strikes.flatten()
        
        # 计算积分区间 [a, b]
        a, b = self._cal_integral_bounds(L, r, tau)
        b_minus_a = b - a
        
        # 计算 phi_levy - 一次性计算特征函数
        k = torch.arange(N, dtype=torch.float64, device=self.device)
        omega = k * torch.pi / b_minus_a
        
        # 使用特征函数计算 phi_levy
        moneyness = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        u_array = omega * 1j
        w_array = torch.zeros_like(omega, dtype=torch.complex128, device=self.device)
        
        phi_levy = self._rss_cf_torch(u_array, w_array, r, tau, moneyness)
        phi_levy[0] *= 0.5  # 零频项需要特殊缩放
        
        # 计算看涨和看跌期权的 Uk 系数
        Uk_call = (self._chi_torch(torch.tensor(0.0, device=self.device), b, a, b, k) -
                  self._psi_torch(torch.tensor(0.0, device=self.device), b, a, b, k)) * 2 / b_minus_a
        Uk_put = (self._psi_torch(a, torch.tensor(0.0, device=self.device), a, b, k) -
                 self._chi_torch(a, torch.tensor(0.0, device=self.device), a, b, k)) * 2 / b_minus_a
        
        # 计算看涨和看跌期权的 Ck 系数
        Ck_call = phi_levy * Uk_call
        Ck_put = phi_levy * Uk_put
        
        # 计算按行权价划分的项
        x_vec = torch.log(S0 / all_strikes_flat)
        strike_bias = torch.exp(1j * omega.unsqueeze(0) * (x_vec.unsqueeze(1) - a))
        
        # 计算所有行权价的看涨和看跌价格
        discount_factor = torch.exp(-r * tau)
        
        call_prices_all = all_strikes_flat * discount_factor * torch.real(strike_bias @ Ck_call)
        put_prices_all = all_strikes_flat * discount_factor * torch.real(strike_bias @ Ck_put)
        # 过滤原始看涨和看跌行权价的结果
        mask_call = torch.isin(all_strikes_flat, strike['call'])
        indices = torch.where(mask_call)[0]
        call_prices = call_prices_all[indices]
        
        mask_put = torch.isin(all_strikes_flat, strike['put'])
        indices = torch.where(mask_put)[0]
        put_prices = put_prices_all[indices]
        
        return {'call': call_prices, 'put': put_prices}
        
