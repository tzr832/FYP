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
    Heston model parameter data class
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
    
    def _compute_kernel_matrices(self, T):
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
            self._compute_kernel_matrices(T)
        elif self.g.size != self.n:
            self._compute_kernel_matrices(T)

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
        chi_k function, computes the cosine coefficients for function g(y) = e^y on interval [c,d]
        
        Parameters:
        c, d: integration interval endpoints
        a, b: overall integration interval endpoints
        k: order of cosine term
        
        Returns:
        χ_k(c,d) analytical solution
        """
        k = np.asarray(k)

        # Scalar case
        # Handle special case for k=0
        mask_k0 = (k == 0)
        mask_k_nonzero = ~mask_k0

        result = np.zeros_like(k, dtype=float)

        if np.any(mask_k_nonzero):
            # k ≠ 0 case
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
            # k = 0 case
            result[mask_k0] = np.exp(d) - np.exp(c)

        return result

    def _psi(self, c, d, a, b, k):
        """
        psi_k function, computes the cosine coefficients for function g(y) = 1 on interval [c,d]
        
        Parameters:
        c, d: integration interval endpoints
        a, b: overall integration interval endpoints
        k: order of cosine term
        
        Returns:
        ψ_k(c,d) analytical solution
        """
        k = np.asarray(k)

        # Scalar case
        # Handle special case for k=0
        mask_k0 = (k == 0)
        mask_k_nonzero = ~mask_k0
        
        result = np.zeros_like(k, dtype=float)
        
        if np.any(mask_k_nonzero):
            # k ≠ 0 case
            omega = k[mask_k_nonzero] * np.pi / (b - a)
            sin_term_d = np.sin(omega * (d - a))
            sin_term_c = np.sin(omega * (c - a))
            
            result[mask_k_nonzero] = (b - a) / (k[mask_k_nonzero] * np.pi) * (sin_term_d - sin_term_c)
        
        if np.any(mask_k0):
            # k = 0 case
            result[mask_k0] = d - c
        
        return result
    
    def _cal_integral_bounds(self, L: float, r:float , tau: float) -> tuple[float, float]:
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
        func_df1 = lambda x: -self._rss_cf(np.array([x])*1j, np.zeros(1), r, tau, 1.0).imag
        func_df2 = lambda x: -self._rss_cf(np.array([x])*1j, np.zeros(1), r, tau, 1.0).real
        func_df3 = lambda x:  self._rss_cf(np.array([x])*1j, np.zeros(1), r, tau, 1.0).imag
        func_df4 = lambda x:  self._rss_cf(np.array([x])*1j, np.zeros(1), r, tau, 1.0).real

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
        
        return a, b

    def call(self, S: float, K: Union[float, np.ndarray], r: float, tau: float,
                   N: int = 256, L: int = 10) -> np.ndarray:
        """
        Calculate European call option prices under Volterra Stein-Stein model using COS method (multiple strikes)
        
        Parameters:
            S: underlying asset price (scalar)
            K: strike price (scalar or array, shape (M,))
            r: risk-free rate
            tau: time to maturity
            N: number of cosine expansion terms (default 256)
            L: integration interval multiplier (default 10)
            
        Returns:
            call option prices (array with same shape as K)
        """
        # Ensure K is numpy array
        K = np.asarray(K)
        original_shape = K.shape
        K_flat = K.flatten()
        
        # Calculate integration interval [a, b]
        a,b = self._cal_integral_bounds(L, r, tau)
        b_minus_a = b - a
        
        # Calculate phi_{levy}
        k = np.arange(N)
        omega = k * np.pi / b_minus_a
        
        # Use Volterra Stein-Stein characteristic function to compute phi_levy
        # Use moneyness=1.0 to compute characteristic function here
        moneyness = 1.0
        u_array = omega * 1j
        w_array = np.zeros_like(omega, dtype=complex)  # 对于对数价格特征函数，w=0

        phi_levy = self._rss_cf(u_array, w_array, r, tau, moneyness)
        phi_levy[0] *= 0.5  # Zero-frequency term requires special scaling
        
        # Uk coefficients for call options
        Uk_call = (self._chi(0, b, a, b, k) - self._psi(0, b, a, b, k)) * 2 / b_minus_a
        
        Ck = phi_levy * Uk_call  # (N, )
        
        # Calculate strike-wise terms
        x_vec = np.log(S / K_flat)
        strike_bias = np.exp(1j * omega[np.newaxis, :] * (x_vec[:, np.newaxis] - a))
        
        call = K_flat * np.exp(-r * tau) * np.real(strike_bias @ Ck)
        
        return call.reshape(original_shape)

    def put(self, S: float, K: Union[float, np.ndarray], r: float, tau: float,
                  N: int = 256, L: int = 10) -> np.ndarray:
        """
        Calculate European put option prices under Volterra Stein-Stein model using COS method (multiple strikes)
        
        Parameters:
            S: underlying asset price (scalar)
            K: strike price (scalar or array, shape (M,))
            r: risk-free rate
            tau: time to maturity
            N: number of cosine expansion terms (default 256)
            L: integration interval multiplier (default 10)
            
        Returns:
            put option prices (array with same shape as K)
        """
        # Ensure K is numpy array
        K = np.asarray(K)
        original_shape = K.shape
        K_flat = K.flatten()
        
        a,b = self._cal_integral_bounds(L, r, tau)
        b_minus_a = b - a
        
        # Calculate phi_{levy}
        k = np.arange(N)
        omega = k * np.pi / b_minus_a
        
        # Use Volterra Stein-Stein characteristic function to compute phi_levy
        # Use moneyness=1.0 to compute characteristic function here
        moneyness = 1.0
        u_array = omega * 1j
        w_array = np.zeros_like(omega, dtype=complex)  # 对于对数价格特征函数，w=0

        phi_levy = self._rss_cf(u_array, w_array, r, tau, moneyness)
        phi_levy[0] *= 0.5  # Zero-frequency term requires special scaling
        
        # Uk coefficients for put options (different from call options)
        Uk_put = (self._psi(a, 0, a, b, k) - self._chi(a, 0, a, b, k)) * 2 / b_minus_a
        
        Ck = phi_levy * Uk_put  # (N, )
        
        # Calculate strike-wise terms
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
        Calculate call and put option prices for specified strikes
        Compute cf for union of call and put strikes, then calculate Uk and strike_bias for calls and puts separately
        
        Parameters:
            S0 (float): initial asset price
            strike (dict): strike price dictionary {'call': array, 'put': array}
            r (float): risk-free rate
            q (float): dividend rate
            tau (float): time to maturity
            N (int): number of cosine expansion terms (default 256)
            L (int): integration interval multiplier (default 10)
            
        Returns:
            dict: {'call': call option price array, 'put': put option price array}
        """
        assert list(strike.keys()) == ['call', 'put']
        
        # Combine all strikes and remove duplicates
        all_strikes = np.unique(np.concatenate([strike['call'], strike['put']]))
        all_strikes_flat = all_strikes.flatten()
        
        # Calculate integration interval [a, b]
        a,b = self._cal_integral_bounds(L, r, tau)
        b_minus_a = b - a
        
        # Calculate phi_{levy} - compute characteristic function once
        k = np.arange(N)
        omega = k * np.pi / b_minus_a
        
        # Use Volterra Stein-Stein characteristic function to compute phi_levy
        moneyness = 1.0
        u_array = omega * 1j
        w_array = np.zeros_like(omega, dtype=complex)  # 对于对数价格特征函数，w=0

        phi_levy = self._rss_cf(u_array, w_array, r, tau, moneyness)
        phi_levy[0] *= 0.5  # Zero-frequency term requires special scaling
        
        # Calculate Uk coefficients for calls and puts
        Uk_call = (self._chi(0, b, a, b, k) - self._psi(0, b, a, b, k)) * 2 / b_minus_a
        Uk_put = (self._psi(a, 0, a, b, k) - self._chi(a, 0, a, b, k)) * 2 / b_minus_a
        
        # Calculate Ck coefficients for calls and puts
        Ck_call = phi_levy * Uk_call  # (N, )
        Ck_put = phi_levy * Uk_put   # (N, )
        
        # Calculate strike-wise terms
        x_vec = np.log(S0 / all_strikes_flat)
        strike_bias = np.exp(1j * omega[np.newaxis, :] * (x_vec[:, np.newaxis] - a))
        
        # Calculate call and put prices for all strikes
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
    # Parameter settings
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
    # Test multiple strike prices
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



    
    print(f"\nVolterra Stein-Stein model COS pricing test:")
    print(f"Parameters: kappa={param.kappa}, nu={param.nu}, rho={param.rho}, theta={param.theta}, X_0={param.X_0}, H={param.H}")
    print(f"Market parameters: S0={S0}, r={r}, tau={tau}")
    print(f"Strike prices: {K_array}")
    
    # Test price method
    strike_dict = {'call': K_array, 'put': K_array}

    pricer.n = 32
    if DEBUG:
        start = time.time()
        prices_less = pricer.price(S0, strike_dict, r, q, tau)
        print(f"Pricing time (n = {pricer.n}): {time.time() - start}")
    else:
        prices_less = pricer.price(S0, strike_dict, r, q, tau)
    parity = prices_less['call'] + K_array * np.exp(-r * tau) - prices_less['put'] - S0
    print(f"tau = 1")
    print("Strike\tCall Price\tPut Price\tParity Error")
    for i, k in enumerate(K_array[::10]):
        print(f"{k}\t{prices_less['call'][i]:.6f}\t{prices_less['put'][i]:.6f}\t{parity[i]:.6e}")

    pricer.n *= 5
    # tau *= 2
    if DEBUG:
        start = time.time()
        prices_more = pricer.price(S0, strike_dict, r, q, tau)
        print(f"Pricing time (n = {pricer.n}): {time.time() - start}")
    else:
        prices_more = pricer.price(S0, strike_dict, r, q, tau)
    parity = prices_more['call'] + K_array * np.exp(-r * tau) - prices_more['put'] - S0
    # print(f"tau = 2")
    print("Strike\tCall Price\tPut Price\tParity Error")
    for i, k in enumerate(K_array[::10]):
        print(f"{k}\t{prices_more['call'][i]:.6f}\t{prices_more['put'][i]:.6f}\t{parity[i]:.6e}")
    
    if DEBUG:
        print("\nComparing pricing results with different n values:")
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        print(f"relative error: {np.max(np.abs((prices_less['call'] - prices_more['call']) / prices_more['call'])):.6e} (call),\
               {np.max(np.abs((prices_less['put'] - prices_more['put']) / prices_more['put'])):.6e} (put)")

        axs[0].plot(K_array, prices_less['call'], label='Call (smaller n)')
        axs[0].plot(K_array, prices_more['call'], '--', label='Call (larger n)')
        axs[0].plot(K_array, prices_less['put'], label='Put (smaller n)')
        axs[0].plot(K_array, prices_more['put'], '--', label='Put (larger n)')
        axs[0].legend()

        axs[1].plot(K_array, np.abs((prices_less['call'] - prices_more['call'])), label='Call abs error')
        axs[1].plot(K_array, np.abs((prices_less['put'] - prices_more['put'])), label='Put abs error')
        axs[1].legend()
        plt.show()
    print(f"\n=== Volterra Stein-Stein COS pricer test completed ===")
