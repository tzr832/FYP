import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import numdifftools as nd
from typing import Union, Tuple, Dict

from MonotonicNetwork import MonotonicNetwork 

class NetworkParams:
    def __init__(self,kappa: float = -8.9e-5,
                 rho: float = -0.704,
                 theta: float = -0.044,
                 X_0: float = 0.113,
                 device='cpu'):
        self.device = device

        self.kappa = nn.Parameter(torch.tensor(kappa), requires_grad=True)
        self.nu = torch.tensor(1.)
        self.rho = nn.Parameter(torch.tensor(rho), requires_grad=True)
        self.theta = nn.Parameter(torch.tensor(theta), requires_grad=True)
        self.X_0 = nn.Parameter(torch.tensor(X_0), requires_grad=True)
        self.network = MonotonicNetwork(2,32,1)
        self.network.load_state_dict(torch.load('pretrain_network.pth', weights_only=False))
        self.network.eval()

class VSSPricerCOSTorch(nn.Module):
    def __init__(self, params: NetworkParams, n: int = 252, device: str = 'cpu'):
        super(VSSPricerCOSTorch, self).__init__()
        self.device = device
        self.n = n

        self.params = params
        

        # Precomputed matrices
        self.KK: torch.Tensor = None
        self.SIG: torch.Tensor = None
        self.g: torch.Tensor = None
        self.KK_sum: torch.Tensor = None
        self.KK_mul: torch.Tensor = None

    def _compute_kernel_matrices(self, T):
        t = torch.linspace(0, T.item(), self.n + 1, dtype=torch.float64, device=self.device)

        tj_1 = t[:-1].unsqueeze(0).repeat(self.n, 1)
        ti_1 = tj_1.T
        tj = t[1:].unsqueeze(0).repeat(self.n, 1)
        
        # 创建神经网络输入：ti 和 tj 的组合
        mask = tj <= ti_1
        t_input = torch.zeros_like(ti_1)
        t_input[mask] = ti_1[mask]
        t_input = t_input.reshape(-1, 1)

        s1_input = torch.zeros_like(tj)
        s1_input[mask] = tj[mask]
        s1_input = s1_input.reshape(-1, 1)

        s0_input = torch.zeros_like(tj_1)
        s0_input[mask] = tj_1[mask]
        s0_input = s0_input.reshape(-1, 1)

        inputs1 = torch.cat([t_input, s1_input], dim=1).float() 
        inputs0 = torch.cat([t_input, s0_input], dim=1).float()

        # KK_tj = self.network(inputs1).reshape(self.n, self.n)
        # KK_tj_1 = self.network(inputs0).reshape(self.n, self.n)
        KK_tj, monoloss1= self.params.network.output_and_monotonicity_loss(inputs1)
        KK_tj_1, monoloss2 = self.params.network.output_and_monotonicity_loss(inputs0)
        KK_tj = KK_tj.reshape(self.n, self.n)
        KK_tj_1 = KK_tj_1.reshape(self.n, self.n)
        
        # self.KK = KK_tj - KK_tj_1
        KK = KK_tj - KK_tj_1
        self.KK = KK * torch.tensor(0.1)
        self.KK_sum = self.KK + self.KK.T

        self.KK_mul = self.KK @ self.KK.T
        dt_inv = torch.tensor(self.n / T)
        self.SIG = self.KK_mul * dt_inv
        

        g_input = t[:-1].unsqueeze(1).repeat(1, 2).float() 

        g_output, monoloss3 = self.params.network.output_and_monotonicity_loss(g_input)
        self.g = self.params.X_0 + self.params.theta * g_output
        self.g = torch.squeeze(self.g)
        self.monoloss = monoloss1 + monoloss2 + monoloss3
    
    def _rss_cf_torch(self, u: torch.Tensor, w: torch.Tensor, r: torch.Tensor, 
                     T: torch.Tensor, moneyness: torch.Tensor) -> torch.Tensor:
        """
        Compute the characteristic function using precise PyTorch operations
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
        
        # Compute tilde(SIG) = inv(I - b * K) * SIG * inv(I - b * K).T
        
        batch_size = u.shape[0]
        I = torch.eye(self.n, dtype=torch.complex128, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Convert matrices to complex type
        KK_complex = self.KK.to(torch.complex128)
        SIG_complex = self.SIG.to(torch.complex128)
        
        X = I - b.unsqueeze(-1).unsqueeze(-1) * KK_complex.unsqueeze(0)
        
        # Solve linear systems using LU decomposition for stability
        Sigma_Xinv = torch.linalg.solve(X, SIG_complex.unsqueeze(0).repeat(batch_size, 1, 1))
        SIG_tilde = torch.linalg.solve(X, Sigma_Xinv.transpose(-2, -1))
        
        # Compute the determinant involved in the characteristic function
        D = I - 2 * a.unsqueeze(-1).unsqueeze(-1) * T / self.n * SIG_tilde
        det_val = torch.linalg.det(D)
        
        # Compute matrix Psi
        KK_sum_complex = self.KK_sum.to(torch.complex128)
        KK_mul_complex = self.KK_mul.to(torch.complex128)
        
        denom = (I - b.unsqueeze(-1).unsqueeze(-1) * KK_sum_complex.unsqueeze(0) +
                b.unsqueeze(-1).unsqueeze(-1)**2 * KK_mul_complex.unsqueeze(0) -
                2 * a.unsqueeze(-1).unsqueeze(-1) * T / self.n * SIG_complex.unsqueeze(0))
        
        Psi = a.unsqueeze(-1).unsqueeze(-1) * torch.linalg.solve(denom, I)
        
        # Quadratic form in g: g.T @ Psi @ g
        g_complex = self.g.to(torch.complex128)
        quad_form = torch.einsum('j,ijk,k->i', g_complex, Psi, g_complex) * T / self.n
        
        values = torch.exp(u * (torch.log(moneyness) + r * T) + quad_form) / torch.sqrt(det_val)
        
        # Apply rotation count algorithm (simplified version)
        arg = torch.angle(det_val)
        bad_ind = torch.where(torch.abs(arg[1:] - arg[:-1]) > 5)[0]
        for i in bad_ind.tolist():
            values[i+1:] = -values[i+1:]
        
        # Ensure phi(0,0) has a positive real part
        max_abs_idx = 0 if u[0] == 0 else torch.argmax(torch.abs(values))
        if torch.abs(values[max_abs_idx]) < 0:
            values = -values
            
        return values
    
    def _chi_torch(self, c: torch.Tensor, d: torch.Tensor, a: torch.Tensor,
                  b: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        chi_k function for PyTorch - computes cosine coefficients for g(y) = e^y
        """
        k = k.to(torch.float64)
        
        # Handle special case for k=0
        mask_k0 = (k == 0)
        mask_k_nonzero = ~mask_k0
        
        result = torch.zeros_like(k, dtype=torch.float64, device=self.device)
        
        if torch.any(mask_k_nonzero):
            # k ≠ 0 case
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
            # k = 0 case
            result[mask_k0] = torch.exp(d) - torch.exp(c)
            
        return result
    
    def _psi_torch(self, c: torch.Tensor, d: torch.Tensor, a: torch.Tensor,
                  b: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        psi_k function for PyTorch - computes cosine coefficients for g(y) = 1
        """
        k = k.to(torch.float64)
        
        # Handle special case for k=0
        mask_k0 = (k == 0)
        mask_k_nonzero = ~mask_k0
        
        result = torch.zeros_like(k, dtype=torch.float64, device=self.device)
        
        if torch.any(mask_k_nonzero):
            # k ≠ 0 case
            omega = k[mask_k_nonzero] * torch.pi / (b - a)
            sin_term_d = torch.sin(omega * (d - a))
            sin_term_c = torch.sin(omega * (c - a))
            
            result[mask_k_nonzero] = (b - a) / (k[mask_k_nonzero] * torch.pi) * (sin_term_d - sin_term_c)
        
        if torch.any(mask_k0):
            # k = 0 case
            result[mask_k0] = d - c
            
        return result
    
    def _cal_integral_bounds(self, L: float, r: torch.Tensor, tau: torch.Tensor) -> tuple[float, float]:
        """
        Calculate integration interval [a, b] bounds
        [a,b] = [c1 - L * sqrt(c2 + sqrt(c4)), c1 + L * sqrt(c2 + sqrt(c4))]
        Parameters:
        L: integration interval multiplier
        tau: time to maturity
        
        Returns:
        a, b: integration interval endpoints
        """
        # Define functions for numerical differentiation
        moneyness = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        func_df1 = lambda x: -self._rss_cf_torch(torch.tensor([x])*1j, torch.zeros(1), r, tau, moneyness).cpu().detach().numpy().imag
        func_df2 = lambda x: -self._rss_cf_torch(torch.tensor(x)*1j, torch.zeros(1), r, tau, moneyness).cpu().detach().numpy().real
        func_df3 = lambda x:  self._rss_cf_torch(torch.tensor([x])*1j, torch.zeros(1), r, tau, moneyness).cpu().detach().numpy().imag
        func_df4 = lambda x:  self._rss_cf_torch(torch.tensor(x)*1j, torch.zeros(1), r, tau, moneyness).cpu().detach().numpy().real

        # Compute moments using numerical differentiation
        mu1 = nd.Derivative(func_df1, n=1)(0)
        mu2 = nd.Derivative(func_df2, n=2)(0)
        mu3 = nd.Derivative(func_df3, n=3)(0)
        mu4 = nd.Derivative(func_df4, n=4)(0)

        # Calculate cumulants
        c1 = mu1
        c2 = mu2 - mu1**2
        c4 = mu4 - 4*mu3*mu1 - 3*mu2**2 + 12*mu2*mu1**2 - 6*mu1**4
        
        a = c1 - L * np.sqrt(c2 + np.sqrt(c4))
        b = c1 + L * np.sqrt(c2 + np.sqrt(c4))    
        
        return torch.tensor(a, device=self.device), torch.tensor(b, device=self.device)

    def call_price(self, S: torch.Tensor, K: torch.Tensor, r: torch.Tensor, 
                  tau: torch.Tensor, N: int = 256, L: float = 10.0) -> torch.Tensor:
        """
        Calculate European call option prices with precise implementation
        """
        
        # Get accurate bounds from NumPy
        a, b = self._cal_integral_bounds(L, r, tau)
        
        original_shape = K.shape
        K_flat = K.flatten()
        b_minus_a = b - a
        
        # Calculate phi_levy using characteristic function
        k = torch.arange(N, dtype=torch.float64, device=self.device)
        omega = k * torch.pi / b_minus_a
        
        # Use characteristic function to compute phi_levy
        moneyness = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        u_array = omega * 1j
        w_array = torch.zeros_like(omega, dtype=torch.complex128, device=self.device)
        
        phi_levy = self._rss_cf_torch(u_array, w_array, r, tau, moneyness)
        phi_levy[0] *= 0.5  # Zero-frequency term requires special scaling
        
        # Uk coefficients for call options
        Uk_call = (self._chi_torch(torch.tensor(0.0, device=self.device), b, a, b, k) - 
                  self._psi_torch(torch.tensor(0.0, device=self.device), b, a, b, k)) * 2 / b_minus_a
        
        Ck = phi_levy * Uk_call
        
        # Calculate strike-wise terms
        x_vec = torch.log(S / K_flat)
        strike_bias = torch.exp(1j * omega.unsqueeze(0) * (x_vec.unsqueeze(1) - a))
        
        call_prices = K_flat * torch.exp(-r * tau) * torch.real(strike_bias @ Ck)
        
        return call_prices.reshape(original_shape)

    def put_price(self, S: torch.Tensor, K: torch.Tensor, r: torch.Tensor,
                 tau: torch.Tensor, N: int = 256, L: float = 10.0) -> torch.Tensor:
        """
        Calculate European put option prices with automatic gradient support
        """
        S = S.to(torch.float64).to(self.device)
        K = K.to(torch.float64).to(self.device)
        r = r.to(torch.float64).to(self.device)
        tau = tau.to(torch.float64).to(self.device)
        
        original_shape = K.shape
        K_flat = K.flatten()
        
        a, b = self._cal_integral_bounds(L, r, tau)
        b_minus_a = b - a
        
        # Calculate phi_levy
        k = torch.arange(N, dtype=torch.float64, device=self.device)
        omega = k * torch.pi / b_minus_a
        
        moneyness = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        u_array = omega * 1j
        w_array = torch.zeros_like(omega, dtype=torch.complex128, device=self.device)
        
        phi_levy = self._rss_cf_torch(u_array, w_array, r, tau, moneyness)
        phi_levy[0] *= 0.5
        
        # Uk coefficients for put options
        Uk_put = (self._psi_torch(a, torch.tensor(0.0, device=self.device), a, b, k) -
                 self._chi_torch(a, torch.tensor(0.0, device=self.device), a, b, k)) * 2 / b_minus_a
        
        Ck = phi_levy * Uk_put
        
        # Calculate strike-wise terms
        x_vec = torch.log(S / K_flat)
        strike_bias = torch.exp(1j * omega.unsqueeze(0) * (x_vec.unsqueeze(1) - a))
        
        put_prices = K_flat * torch.exp(-r * tau) * torch.real(strike_bias @ Ck)
        
        return put_prices.reshape(original_shape)
    
    def price(self, S0: torch.Tensor, strike: Dict[str, torch.Tensor],
              r: torch.Tensor, q: torch.Tensor, tau: torch.Tensor,
              N: int = 256, L: float = 10.0) -> Dict[str, torch.Tensor]:
        """
        Calculate call and put option prices for specified strikes with gradient support
        """
        # Combine all strikes and remove duplicates
        all_strikes = torch.unique(torch.cat([strike['call'], strike['put']]))
        all_strikes_flat = all_strikes.flatten()
        
        # Calculate integration interval [a, b]
        a, b = self._cal_integral_bounds(L, r, tau)
        b_minus_a = b - a
        
        # Calculate phi_levy - compute characteristic function once
        k = torch.arange(N, dtype=torch.float64, device=self.device)
        omega = k * torch.pi / b_minus_a
        
        # Use characteristic function to compute phi_levy
        moneyness = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        u_array = omega * 1j
        w_array = torch.zeros_like(omega, dtype=torch.complex128, device=self.device)
        
        phi_levy = self._rss_cf_torch(u_array, w_array, r, tau, moneyness)
        phi_levy[0] *= 0.5  # Zero-frequency term requires special scaling
        
        # Calculate Uk coefficients for calls and puts
        Uk_call = (self._chi_torch(torch.tensor(0.0, device=self.device), b, a, b, k) -
                  self._psi_torch(torch.tensor(0.0, device=self.device), b, a, b, k)) * 2 / b_minus_a
        Uk_put = (self._psi_torch(a, torch.tensor(0.0, device=self.device), a, b, k) -
                 self._chi_torch(a, torch.tensor(0.0, device=self.device), a, b, k)) * 2 / b_minus_a
        
        # Calculate Ck coefficients for calls and puts
        Ck_call = phi_levy * Uk_call
        Ck_put = phi_levy * Uk_put
        
        # Calculate strike-wise terms
        x_vec = torch.log(S0 / all_strikes_flat)
        strike_bias = torch.exp(1j * omega.unsqueeze(0) * (x_vec.unsqueeze(1) - a))
        
        # Calculate call and put prices for all strikes
        discount_factor = torch.exp(-r * tau)
        
        call_prices_all = all_strikes_flat * discount_factor * torch.real(strike_bias @ Ck_call)
        put_prices_all = all_strikes_flat * discount_factor * torch.real(strike_bias @ Ck_put)
        
        # Filter results for original call and put strikes
        mask_call = torch.isin(all_strikes_flat, strike['call'])
        indices = torch.where(mask_call)[0]
        call_prices = call_prices_all[indices]
        
        mask_put = torch.isin(all_strikes_flat, strike['put'])
        indices = torch.where(mask_put)[0]
        put_prices = put_prices_all[indices]
        
        return {'call': call_prices, 'put': put_prices}

if __name__ == "__main__":
    import time 

    device = 'cpu'
    params = NetworkParams()
    calibrator = VSSPricerCOSTorch(params, n=252)
    
    S0 = torch.tensor(25000., dtype=torch.float64, device=device)
    r = torch.tensor(0.03, dtype=torch.float64, device=device)
    q = torch.tensor(0., dtype=torch.float64, device=device)
    tau = torch.tensor(1.0, dtype=torch.float64, device=device)
    K = {'call': torch.linspace(16000., 35000., steps=10, dtype=torch.float64, device=device),
         'put': torch.linspace(16000., 35000., steps=10, dtype=torch.float64, device=device)}
    
    print("\n=== Precise PyTorch Volterra Stein-Stein COS Pricer Test ===")
    print(f"Parameters: kappa={params.kappa.item()}, nu={params.nu.item()}, rho={params.rho.item()}, "
            f"theta={params.theta.item()}, X_0={params.X_0.item()}")
    print(f"Market parameters: S0={S0.item()}, r={r.item()}, tau={tau.item()}")
    
    # Test call price
    print("\n=== Testing call_price method ===")
    start = time.time()
    price = calibrator.price(S0, K, r, q, tau)
    print("Strike\tCall Price\tPut Price\tCall-Put parity")
    for strike, call, put in zip(K['call'].tolist(), price['call'].tolist(), price['put'].tolist()):
        parity = call - put + strike * np.exp(-r.item()*tau.item()) - S0
        print(f"{strike:.2f}\t{call:.6f}\t{put:.6f}\t{parity:.6f}")
    print(f"computed in {time.time() - start:.4f} seconds")

    
    