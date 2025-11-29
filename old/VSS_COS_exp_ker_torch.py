
"""
Precise PyTorch implementation of Volterra Stein-Stein model
This version aims to exactly replicate the NumPy implementation
"""

import json
import torch
from torch.optim import Adam
import numpy as np
import numdifftools as nd
from typing import Union, Tuple, Dict
from math import atanh

class VSSParamLaplaceTorch:
    """
    Volterra Stein-Stein model parameters with PyTorch tensors
    """
    def __init__(self, 
                 kappa: float = -8.9e-5,
                #  nu: float = 0.176,
                 rho: float = -0.704,
                 theta: float = -0.044,
                 X_0: float = 0.113,
                 terms: int = 10,
                 device: str = 'cpu'):
        
        self.device = device
        
        # Convert parameters to tensors with requires_grad=True for gradient computation
        self.kappa = torch.tensor(kappa, dtype=torch.float64, device=device, requires_grad=True)
        self.nu = torch.tensor(1, dtype=torch.float64, device=device)
        self.rho = torch.tensor(rho, dtype=torch.float64, device=device, requires_grad=True)
        self.theta = torch.tensor(theta, dtype=torch.float64, device=device, requires_grad=True)
        self.X_0 = torch.tensor(X_0, dtype=torch.float64, device=device, requires_grad=True)
        
        self.c = torch.rand(terms, dtype=torch.float64, device=device, requires_grad=True)
        self.gamma = torch.rand(terms, dtype=torch.float64, device=device, requires_grad=True)
        with torch.no_grad():
            self.gamma.mul_(10.0) # Scale gamma to a reasonable range
            self.c.mul_(0.1)
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate parameter constraints"""
        if not (-1 < self.rho < 1):
            raise ValueError("rho must be in the interval (-1, 1).")
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Return parameters as dictionary"""
        return {
            'kappa': self.kappa.item(),
            'nu': self.nu.item(),
            'rho': self.rho.item(),
            'theta': self.theta.item(),
            'X_0': self.X_0.item(),
            'c': self.c.tolist(),
            'gamma': self.gamma.tolist()
        }
    
    def set_requires_grad(self, requires_grad: bool = True):
        """Set requires_grad for all parameters"""
        for param_name, param in [('kappa', self.kappa), ('nu', self.nu), ('rho', self.rho),
                                 ('theta', self.theta), ('X_0', self.X_0), ('c', self.c), ('gamma', self.gamma)]:
            try:
                param.requires_grad = requires_grad
            except RuntimeError as e:
                print(f"ERROR: Failed to set requires_grad for {param_name}: {e}")
                raise


class VSSPricerCOSTorch:
    """
    Precise PyTorch implementation of Volterra Stein-Stein model option pricer
    This version aims to exactly replicate the NumPy implementation
    """
    
    def __init__(self, params: VSSParamLaplaceTorch, n: int = 252, device: str = 'cpu'):
        self.params = params
        self.device = device
        self.n = n
        
        # Precomputed matrices
        self.KK = None
        self.SIG = None
        self.g = None
        self.KK_sum = None
        self.KK_mul = None
    
    def set_params(self, params: VSSParamLaplaceTorch) -> None:
        """Set model parameters"""
        self.params = params

        self.KK = None
        self.SIG = None
        self.g = None
        self.KK_sum = None
        self.KK_mul = None
    
    def _compute_kernel_matrices(self, T: torch.Tensor) -> None:
        """
        Computes the kernel matrix, covariance matrix, and adjusted input vector for the
        Volterra Stochastic Volatility model with exponential decay kernel using PyTorch.

        Parameters:
        - T: torch.Tensor
            Maturity time.

        Sets:
        - self.g: torch.Tensor of shape (n,)
            Vector of adjusted values based on the input curve for each time step.
        - self.KK: torch.Tensor of shape (n, n)
            2D matrix representing the kernel's approximation matrix.
        - self.SIG: torch.Tensor of shape (n, n)
            2D covariance matrix Σ_0.
        - self.KK_sum: torch.Tensor of shape (n, n)
            Sum of kernel matrices for characteristic function computation.
        - self.KK_mul: torch.Tensor of shape (n, n)
            Product of kernel matrices for characteristic function computation.
        """
        n = self.n
        nu = self.params.nu
        theta = self.params.theta
        X_0 = self.params.X_0
        c = self.params.c
        gamma = self.params.gamma
        
        assert len(c) == len(gamma), "Length of c and gamma must be the same."
        terms = len(c)

        # Time discretization from 0 to T
        t = torch.linspace(0, T.item(), n + 1, dtype=torch.float64, device=self.device)

        # Define indices for 2D matrices
        tj_1 = t[:-1].unsqueeze(0).repeat(self.n, 1) # Times tj excluding the final point
        ti_1 = tj_1.T  # Transpose to create a grid of ti values
        tj = t[1:].unsqueeze(0).repeat(self.n, 1)  # Times tj excluding the initial point

        # Initialize kernel matrix KK for exponential decay kernel
        KK = torch.zeros((terms, n, n), dtype=torch.float64, device=self.device)
        mask = tj <= ti_1
        
        for k in range(terms):
            # Compute kernel values only where mask is True
            exp_term1 = torch.exp(-gamma[k] * (ti_1 - tj))
            exp_term2 = torch.exp(-gamma[k] * (ti_1 - tj_1))
            KK[k] = torch.where(mask, (exp_term1 - exp_term2) / gamma[k], 
                                torch.tensor(0.0, dtype=torch.float64, device=self.device))
        
        # Sum over terms dimension with broadcasting
        KK = torch.sum(c.view(terms, 1, 1) * KK, dim=0)

        # Compute covariance matrix SIG
        SIG = torch.zeros((terms, n, n), dtype=torch.float64, device=self.device)
        for k in range(terms):
            min_t = torch.minimum(ti_1, tj_1)
            exp_term = torch.exp(-gamma[k] * (ti_1 + tj_1)) * (torch.exp(2 * gamma[k] * min_t) - 1) / (2 * gamma[k])
            SIG[k] = exp_term
        
        SIG = torch.sum((nu**2) * c.view(terms, 1, 1) * SIG, dim=0)

        # Compute adjusted vector g based on initial conditions
        # For exponential kernel, the deterministic part is simpler
        t_reshaped = t[:-1].unsqueeze(1)  # Shape (n, 1)
        gamma_reshaped = gamma.unsqueeze(0)  # Shape (1, terms)
        exp_term = 1 - torch.exp(-gamma_reshaped * t_reshaped)
        sum_term = torch.sum(c / gamma * exp_term, dim=1)
        g = X_0 + theta * sum_term

        # Store precomputed matrices
        self.g = g
        self.KK = KK
        self.SIG = SIG
        
        # Precompute additional matrices for characteristic function
        self.KK_sum = KK + KK.T
        self.KK_mul = KK @ KK.T
    
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
    
    def compute_gradients(self, objc) -> Dict[str, torch.Tensor]:
        """
        Compute gradients of call price with respect to all model parameters
        """
        # Ensure all parameters require gradients
        self.params.set_requires_grad(True)
        
        # Clear any existing gradients
        for param in [self.params.kappa, self.params.nu, self.params.rho,
                     self.params.theta, self.params.X_0, self.params.gamma, self.params.c]:
            if param.grad is not None:
                param.grad.zero_()
            
        # Compute gradients
        gradients = {}
        objc.backward()
        
        gradients['kappa'] = self.params.kappa.grad.clone() if self.params.kappa.grad is not None else None
        gradients['nu'] = self.params.nu.grad.clone() if self.params.nu.grad is not None else None
        gradients['rho'] = self.params.rho.grad.clone() if self.params.rho.grad is not None else None
        gradients['theta'] = self.params.theta.grad.clone() if self.params.theta.grad is not None else None
        gradients['X_0'] = self.params.X_0.grad.clone() if self.params.X_0.grad is not None else None
        gradients['gamma'] = self.params.gamma.grad.clone() if self.params.gamma.grad is not None else None
        gradients['c'] = self.params.c.grad.clone() if self.params.c.grad is not None else None
        
        return gradients
    
    def objective(self, dict_path: str='Data/250901.json') -> torch.Tensor:
        """
        Objective function for calibration: RMSE between model and market prices
        """

        with open(dict_path, 'r', encoding='utf-8') as f:
            optiondict = json.load(f)

        S0 = torch.tensor(optiondict['HSI'], dtype=torch.float64, device=self.device)
        r = torch.tensor(optiondict['rf'], dtype=torch.float64, device=self.device)
        
        error = torch.tensor([], dtype=torch.float64, device=self.device)
        
        for key, value in optiondict.items():
            if not isinstance(value, dict):
                continue
            
            strike = {'call': torch.tensor(value['strike']['call'], dtype=torch.float64, device=self.device),
                      'put': torch.tensor(value['strike']['put'], dtype=torch.float64, device=self.device)}

            q = torch.tensor(0.0, dtype=torch.float64, device=self.device)
            tau = torch.tensor(value['tau'], dtype=torch.float64, device=self.device)
            
            # Set number of terms based on tau
            self.n = max(32, int(tau.item() * 63))
            
            modelPrice = self.price(S0, strike, r, q, tau)
            
            error_call = (modelPrice['call'] - torch.tensor(value['price']['call'], dtype=torch.float64, device=self.device)) ** 2
            error_put = (modelPrice['put'] - torch.tensor(value['price']['put'], dtype=torch.float64, device=self.device)) ** 2
            error = torch.cat([error, error_call, error_put])
        
        rmse = torch.sqrt(torch.mean(error))
        return rmse

    def calibrate(self, dict_path: str='Data/250901.json', tol=1e-3, lr: float=1e-2, epochs: int=1000) -> None:
        """
        Train model parameters using Adam optimizer to minimize objective function
        """
        init_kappa = torch.tensor(-8.9e-5, dtype=torch.float64, requires_grad=True)
        init_rho = torch.tensor(atanh(-0.704), dtype=torch.float64, requires_grad=True)
        init_theta = torch.tensor(-0.044, dtype=torch.float64, requires_grad=True)
        init_X0 = torch.tensor(0.113, dtype=torch.float64, requires_grad=True)
        init_c = torch.tensor([0.005815448596142969, 0.006291016742457756, 0.012358607277440904,
                               0.0052580164361077045, 0.052617189685476184, 0.04767847537874797, 
                               0.09552357002491417, 0.09287525814574296, 0.008354336875286783, 0.013264067215369858], 
                              dtype=torch.float64, requires_grad=True)
        init_gamma = torch.tensor([1.5705349751070807, 3.753670320505006, 8.425197308314868, 
                                   8.705010417836975, 3.7753295527055943, 6.124365889937913, 
                                   0.88109784914571, 7.011818919498007, 6.234030833542006, 4.372789977906521], 
                                  dtype=torch.float64, requires_grad=True)
        param = VSSParamLaplaceTorch()
        with torch.no_grad():
            init_c = torch.log(init_c)
            init_gamma = torch.log(init_gamma)

        optimizer = Adam([init_kappa, init_rho, init_theta, init_X0], lr=lr)
        last_loss = torch.tensor(torch.inf)

        for epoch in range(epochs):
            rho = torch.tanh(init_rho)
            param.kappa=init_kappa
            param.rho=rho
            param.X_0=init_X0
            param.theta=init_theta
            param.c = torch.exp(init_c)
            param.gamma = torch.exp(init_gamma)
            self.params = param


            optimizer.zero_grad()
            loss = self.objective(dict_path)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Objective: {loss.item():.6f}")
        
            if torch.abs(last_loss - loss) < 1e-3:
                print(f"Convergence reached. ({tol})")
                return {"suc": True, "loss": loss.item(), "param": self.params.to_dict()}
            last_loss = loss

        print(f"Convergence didn't reached after {epochs} epochs. ({tol})")
        return {"suc": False, "loss": loss.item(), "param": self.params.to_dict()}
    
def demo():
    import time
    # Test the precise PyTorch implementation
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    # Create parameters
    params = VSSParamLaplaceTorch(
        kappa=8.9e-5,
        # nu=0.176,
        rho=-0.704,
        theta=-0.044,
        X_0=0.113,
        terms=10,
        device=device
    )
    
    pricer = VSSPricerCOSTorch(params, n=252*2, device=device)
    
    # Test data
    S0 = torch.tensor(25617.42, dtype=torch.float64, device=device)
    r = torch.tensor(0.03, dtype=torch.float64, device=device)
    tau = torch.tensor(2.0, dtype=torch.float64, device=device)
    K = torch.linspace(16000.0, 35000.0, steps=100, dtype=torch.float64, device=device)
    
    print("\n=== Precise PyTorch Volterra Stein-Stein COS Pricer Test ===")
    print(f"Parameters: kappa={params.kappa.item()}, nu={params.nu.item()}, rho={params.rho.item()}, "
            f"theta={params.theta.item()}, X_0={params.X_0.item()}")
    print(f"Number of terms: {len(params.c)}")
    print(f"c: {params.c.tolist()}")
    print(f"gamma: {params.gamma.tolist()}")
    print(f"Market parameters: S0={S0.item()}, r={r.item()}")
    
    # Test call price
    print("\n=== Testing objective method ===")
    start = time.time()
    # call_price = pricer.call_price(S0, K, r, tau)
    error = pricer.objective(dict_path='Data/250901.json')
    print(f"Objective RMSE: {error.item():.6f}")
    # print("Strike\tCall Price")
    # for strike, price in zip(K.tolist(), call_price.tolist()):
    #     print(f"{strike:.2f}\t{price:.6f}")
    print(f"computed in {time.time() - start:.4f} seconds")


    
    # Test gradient computation
    print("\n=== Gradient Computation Test ===")
    # K_single = torch.tensor([25000.0], dtype=torch.float64, device=device)
    start = time.time()
    gradients = pricer.compute_gradients(error)
    print(f"Gradients computed in {time.time() - start:.4f} seconds")
    for param_name, grad in gradients.items():
        if grad is not None:
            if param_name in ['gamma', 'c']:
                print(f"Gradient w.r.t {param_name}: {grad}")
            else:
                print(f"Gradient w.r.t {param_name}: {grad.item():.6e}")
        else:
            print(f"Gradient w.r.t {param_name}: None")
    
    
    print("\n=== Test Completed Successfully ===")

def main():
    device = 'cpu'
    print(f"Using device: {device}")
    
    torch.manual_seed(42)

    calibrator = VSSPricerCOSTorch(VSSParamLaplaceTorch(), device=device)

    result = calibrator.calibrate()
    with open("Laplace_calibration_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f)

    print("Optimization result has been saved")


if __name__ == "__main__":
    main() 
    # demo()