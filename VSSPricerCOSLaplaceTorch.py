"""
Volterra Stein-Stein COS Pricer with exponential kernel implementation
"""

import torch
import json
from torch.optim import Adam
import numpy as np
import numdifftools as nd
from typing import Union, Tuple, Dict
from math import atanh

from VSSPricerCOSBase import VSSPricerCOSBase
from VSSParamLaplaceTorch import VSSParamLaplaceTorch


class VSSPricerCOSLaplaceTorch(VSSPricerCOSBase):
    """
    Volterra Stein-Stein COS Pricer with exponential kernel implementation
    This version aims to exactly replicate the NumPy implementation
    """
    
    def __init__(self, params: VSSParamLaplaceTorch, n: int = 252, device: str = 'cpu'):
        """
        初始化指数核实现
        
        Args:
            params: VSSParamLaplaceTorch 参数对象
            n: 时间离散化点数
            device: 计算设备
        """
        super().__init__(params, n, device)
    
    def set_params(self, params: VSSParamLaplaceTorch) -> None:
        """设置模型参数"""
        self.params = params

        self.KK = None
        self.SIG = None
        self.g = None
        self.KK_sum = None
        self.KK_mul = None
    
    def _compute_kernel_matrices(self, T: torch.Tensor) -> None:
        """
        计算核矩阵、协方差矩阵和调整输入向量
        使用指数衰减核的 PyTorch 实现

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
    
    def objective(self, dict_: Union[str, Dict] ='Data/250901.json') -> torch.Tensor:
        """
        校准目标函数：模型价格与市场价格之间的 RMSE
        """
        if isinstance(dict_, str):
            with open(dict_, 'r', encoding='utf-8') as f:
                optiondict = json.load(f)
        else:
            optiondict = dict_

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
            self.clear_kernel_matrices()
            error_call = (modelPrice['call'] - torch.tensor(value['price']['call'], dtype=torch.float64, device=self.device)) ** 2
            error_put = (modelPrice['put'] - torch.tensor(value['price']['put'], dtype=torch.float64, device=self.device)) ** 2
            error = torch.cat([error, error_call, error_put])
        
        rmse = torch.sqrt(torch.mean(error))
        return rmse
    
    def calibrate(self, dict_: Union[str, Dict], tol=1e-3, lr: float=1e-2, epochs: int=1000) -> None:
        """
        使用 Adam 优化器训练模型参数以最小化目标函数
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

        optimizer = Adam([init_kappa, init_rho, init_theta, init_X0, init_c, init_gamma], lr=lr)
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
            loss = self.objective(dict_)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Objective: {loss.item():.6f}")
        
            if torch.abs(last_loss - loss) < 1e-3:
                print(f"Convergence reached. ({tol})")
                return {"suc": True, "loss": loss.item(), "param": self.params.to_dict()}
            last_loss = loss

        print(f"Convergence didn't reached after {epochs} epochs. ({tol})")
        return {"suc": False, "loss": loss.item(), "param": self.params.to_dict()}
    
def main():
    device = 'cpu'
    print(f"Using device: {device}")
    
    torch.manual_seed(42)

    calibrator = VSSPricerCOSLaplaceTorch(VSSParamLaplaceTorch(), device=device)
    with open("Data/250901.json", 'r', encoding='utf-8') as f:
        option_dict = json.load(f)
        
    result = calibrator.calibrate(option_dict)
    with open("results/Laplace_calibration_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f)

    print("Optimization result has been saved")


if __name__ == "__main__":
    main() 
    # demo()