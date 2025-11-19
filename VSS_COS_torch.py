
"""
Precise PyTorch implementation of Volterra Stein-Stein model
This version aims to exactly replicate the NumPy implementation
"""

import torch
import torch.nn as nn
import numpy as np
import numdifftools as nd
from typing import Union, Tuple, Dict

from hyp2f1_numerical import hyp2f1

class VSSParamTorch:
    """
    Volterra Stein-Stein model parameters with PyTorch tensors
    """
    def __init__(self, 
                 kappa: float = -8.9e-5,
                 nu: float = 0.176,
                 rho: float = -0.704,
                 theta: float = -0.044,
                 X_0: float = 0.113,
                 H: float = 0.279,
                 device: str = 'cpu'):
        
        self.device = device
        
        # Convert parameters to tensors with requires_grad=True for gradient computation
        self.kappa = torch.tensor(kappa, dtype=torch.float64, device=device, requires_grad=True)
        self.nu = torch.tensor(nu, dtype=torch.float64, device=device, requires_grad=True)
        self.rho = torch.tensor(rho, dtype=torch.float64, device=device, requires_grad=True)
        self.theta = torch.tensor(theta, dtype=torch.float64, device=device, requires_grad=True)
        self.X_0 = torch.tensor(X_0, dtype=torch.float64, device=device, requires_grad=True)
        self.H = torch.tensor(H, dtype=torch.float64, device=device, requires_grad=True)
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate parameter constraints"""
        if not (0 < self.H < 1):
            raise ValueError("Hurst index H must be in the interval (0, 1).")
        if not (-1 < self.rho < 1):
            raise ValueError("rho must be in the interval (-1, 1).")
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Return parameters as dictionary"""
        return {
            'kappa': self.kappa,
            'nu': self.nu,
            'rho': self.rho,
            'theta': self.theta,
            'X_0': self.X_0,
            'H': self.H
        }
    
    def set_requires_grad(self, requires_grad: bool = True):
        """Set requires_grad for all parameters"""
        for param in [self.kappa, self.nu, self.rho, self.theta, self.X_0, self.H]:
            param.requires_grad = requires_grad


class VSSPricerCOSTorch:
    """
    Precise PyTorch implementation of Volterra Stein-Stein model option pricer
    This version aims to exactly replicate the NumPy implementation
    """
    
    def __init__(self, params: VSSParamTorch, n: int = 252, device: str = 'cpu'):
        self.params = params
        self.device = device
        self.n = n
        
        # Precomputed matrices
        self.KK = None
        self.SIG = None
        self.g = None
        self.KK_sum = None
        self.KK_mul = None
        
    def _compute_kernel_matrices(self, T: torch.Tensor) -> None:
        """
        Compute kernel matrix, covariance matrix, and adjusted input vector
        using exact PyTorch operations matching NumPy implementation
        """
        alpha = self.params.H + 0.5
        
        # Time discretization from 0 to T
        t = torch.linspace(0, T.item(), self.n + 1, dtype=torch.float64, device=self.device)
        
        # Define indices for 2D matrices - exactly matching NumPy implementation
        # tj_1 = t[:-1].repeat(self.n, 1).T  # Times tj excluding the final point
        # ti_1 = tj_1.T  # Transpose to create a grid of ti values
        # tj = t[1:].repeat(self.n, 1).T  # Times tj excluding the initial point

        tj_1 = t[:-1].unsqueeze(0).repeat(self.n, 1) # Times tj excluding the final point
        ti_1 = tj_1.T  # Transpose to create a grid of ti values
        tj = t[1:].unsqueeze(0).repeat(self.n, 1)  # Times tj excluding the initial point
        
        # Initialize kernel matrix KK
        self.KK = torch.zeros((self.n, self.n), dtype=torch.float64, device=self.device)
        
        # K^n_{ij}= \bm 1_{j\leq i-1}\int_{t_{j-1}}^{t_j} K(t_{i-1},s)ds
        mask = tj <= ti_1
        self.KK[mask] = ((ti_1 - tj_1)[mask] ** alpha - (ti_1 - tj)[mask] ** alpha) / torch.exp(torch.lgamma(1 + alpha))
        
        self.KK_sum = self.KK + self.KK.T
        self.KK_mul = self.KK @ self.KK.T
        
        # Compute covariance matrix SIG - using simplified approach for PyTorch
        # In practice, we need to implement hyp2f1 or use approximation
        min_t = torch.minimum(ti_1, tj_1)
        max_t = torch.maximum(ti_1, tj_1)

        max_t[0,0] = 1. # to deal with 0/0 error

        ratio = min_t / max_t
        hyp2f1_approx = hyp2f1(1 - alpha,
                            torch.tensor([1.0], dtype=torch.float64, device=self.device),
                            1 + alpha,
                            ratio)
        self.SIG = self.params.nu ** 2 * (hyp2f1_approx * min_t ** alpha / (max_t ** (1 - alpha)) / (torch.exp(torch.lgamma(1 + alpha)) * torch.exp(torch.lgamma(alpha))))
        
        # Compute adjusted vector g based on initial conditions
        self.g = (self.params.X_0 + self.params.theta * t[:-1] ** alpha / torch.exp(torch.lgamma(1 + alpha)))
    
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
        h: numerical differentiation step size (default adjusted to 1e-4 for stability)
        
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
    # Test the precise PyTorch implementation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f"Using device: {device}")
    

    # Create parameters
    params = VSSParamTorch(
        kappa=8.9e-5,
        nu=0.176,
        rho=-0.704,
        theta=-0.044,
        X_0=0.113,
        H=0.279,
        device=device
    )
    
    pricer = VSSPricerCOSTorch(params, n=252, device=device)
    
    # Test data
    S0 = torch.tensor(25617.42, dtype=torch.float64, device=device)
    r = torch.tensor(0.03, dtype=torch.float64, device=device)
    tau = torch.tensor(1.0, dtype=torch.float64, device=device)
    K = torch.linspace(16000.0, 35000.0, steps=100, dtype=torch.float64, device=device)
    
    print("\n=== Precise PyTorch Volterra Stein-Stein COS Pricer Test ===")
    print(f"Parameters: kappa={params.kappa.item()}, nu={params.nu.item()}, rho={params.rho.item()}, "
            f"theta={params.theta.item()}, X_0={params.X_0.item()}, H={params.H.item()}")
    print(f"Market parameters: S0={S0.item()}, r={r.item()}, tau={tau.item()}")
    
    # Test call price
    print("\n=== Testing call_price method ===")
    start = time.time()
    call_price = pricer.call_price(S0, K, r, tau)
    print("Strike\tCall Price")
    for strike, price in zip(K.tolist()[::10], call_price.tolist()[::10]):
        print(f"{strike:.2f}\t{price:.6f}")
    print(f"computed in {time.time() - start:.4f} seconds")
        

    
    # # Compare with NumPy
    # from VSS_COS import VSSParam, VSSPricerCOS
    # param_np = VSSParam(
    #     kappa=8.9e-5,
    #     nu=0.176,
    #     rho=-0.704,
    #     theta=-0.044,
    #     X_0=0.113,
    #     H=0.279
    # )

    # pricer_np = VSSPricerCOS(param_np, n=252)
    # start = time.time()
    # call_price_np = pricer_np.call(S0.cpu().numpy(), K.cpu().numpy(), r.item(), tau.item())
    # for strike, price in zip(K.tolist(), call_price_np):
    #     print(f"{strike:.2f}\t{price:.6f}")
    # print(f"computed in {time.time() - start:.4f} seconds")
    # print(f"Abs error: {abs(call_price.detach().cpu().numpy() - call_price_np)}")
    
    
    print("\n=== Test Completed Successfully ===")