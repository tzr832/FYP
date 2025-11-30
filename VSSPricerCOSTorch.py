"""
Volterra Stein-Stein COS Pricer with standard kernel implementation
"""

import torch
import numpy as np
import json
from typing import Union, Tuple, Dict
from math import atanh, log
from torch.optim import Adam
from torch import sigmoid

from VSSPricerCOSBase import VSSPricerCOSBase
from VSSParamTorch import VSSParamTorch
from hyp2f1_numerical import hyp2f1


def inv_sigmoid(y):
    return log(y / (1 - y))


class VSSPricerCOSTorch(VSSPricerCOSBase):
    """
    Volterra Stein-Stein COS Pricer with standard kernel implementation
    This version aims to exactly replicate the NumPy implementation
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        初始化标准核实现
        
        Args:
            params: VSSParamTorch 参数对象
            n: 时间离散化点数
            device: 计算设备
        """
        super().__init__(VSSParamTorch(), n=252, device=device)
    
    def _compute_kernel_matrices(self, T: torch.Tensor) -> None:
        """
        计算核矩阵、协方差矩阵和调整输入向量
        使用精确的 PyTorch 操作匹配 NumPy 实现
        """
        alpha = self.params.H + 0.5
        
        # Time discretization from 0 to T
        t = torch.linspace(0, T.item(), self.n + 1, dtype=torch.float64, device=self.device)
        
        # Define indices for 2D matrices - exactly matching NumPy implementation
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
    
    def objective(self, option_dict: Dict) -> torch.Tensor:
        """
        校准目标函数：模型价格与市场价格之间的 RMSE
        
        Args:
            option_dict: 期权数据字典，包含市场价格信息
            
        Returns:
            torch.Tensor: RMSE 损失值
        """
        S0 = torch.tensor(option_dict['HSI'], dtype=torch.float64, device=self.device)
        r = torch.tensor(option_dict['rf'], dtype=torch.float64, device=self.device)
        
        error = torch.tensor([], dtype=torch.float64, device=self.device)
        
        for key, value in option_dict.items():
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
    
    def calibrate(self, option_dict: Dict, init_param: VSSParamTorch = None,
                 lr: float = 0.01, tol: float = 1e-3, epochs: int = 1000) -> Dict:
        """
        使用 Adam 优化器训练模型参数以最小化目标函数
        
        Args:
            option_dict: 期权数据字典
            init_param: 初始参数对象，如果为 None 则使用默认值
            lr: 学习率
            tol: 收敛容忍度
            epochs: 最大迭代次数
            
        Returns:
            Dict: 校准结果，包含成功标志、损失值和最终参数
        """
        if init_param is None:
            init_kappa = torch.tensor(-8.9e-5, dtype=torch.float64, requires_grad=True)
            init_nu = torch.tensor(0.176, dtype=torch.float64, requires_grad=True)
            init_rho = torch.tensor(atanh(-0.704), dtype=torch.float64, requires_grad=True)
            init_theta = torch.tensor(-0.044, dtype=torch.float64, requires_grad=True)
            init_X0 = torch.tensor(0.113, dtype=torch.float64, requires_grad=True)
            init_H = torch.tensor(inv_sigmoid(0.279), dtype=torch.float64, requires_grad=True)
        else:
            init_kappa = torch.tensor(init_param.kappa.item(), dtype=torch.float64, requires_grad=True)
            init_nu = torch.tensor(init_param.nu.item(), dtype=torch.float64, requires_grad=True)
            init_rho = torch.tensor(atanh(init_param.rho.item()), dtype=torch.float64, requires_grad=True)
            init_theta = torch.tensor(init_param.theta.item(), dtype=torch.float64, requires_grad=True)
            init_X0 = torch.tensor(init_param.X_0.item(), dtype=torch.float64, requires_grad=True)
            init_H = torch.tensor(inv_sigmoid(init_param.H.item()), dtype=torch.float64, requires_grad=True)

        optimizer = Adam(params=[init_kappa, init_nu, init_rho, init_theta, init_X0, init_H], lr=lr)
        prev_loss = torch.tensor(torch.inf)
        
        try:
            for epoch in range(epochs):
                rho = torch.tanh(init_rho)
                H = torch.sigmoid(init_H)

                param = VSSParamTorch()
                param.kappa = init_kappa
                param.nu = init_nu
                param.rho = rho
                param.theta = init_theta
                param.X_0 = init_X0
                param.H = H
                self.params = param

                optimizer.zero_grad()
                loss = self.objective(option_dict)
                loss.backward()
                optimizer.step()
                

                if (epoch + 1) % 1 == 0:
                    print(f"Epoch {epoch+1}: Loss={loss.item():.6f}", end=" | ")
                    print(f"Params: kappa={self.params.kappa.item():.8f}, nu={self.params.nu.item():.6f}, "
                        f"rho={rho.item():.6f}, theta={self.params.theta.item():.6f}, "
                        f"X_0={self.params.X_0.item():.6f}, H={H.item():.6f}")
                
                if abs(loss - prev_loss) < tol:
                    print(f"Optimization finished! The optimum parameters are found within given tolerance ({tol})")
                    return {"suc": True, "loss": loss.item(), "param": param.to_dict()}
                prev_loss = loss
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
            return {"suc": False, "loss": loss.item(), "param": param.to_dict()}

        print(f"Optimization finished! The optimum parameters are not found within given tolerance ({tol})")
        return {"suc": False, "loss": loss.item(), "param": param.to_dict()}
    

if __name__ == "__main__":
    device = 'cpu'
    calibrator = VSSPricerCOSTorch(device)
    # print(calibrator.objective())
    print(f"Using device: {device}. Starting calibration...")
    init_param = VSSParamTorch(kappa=0.00842271,
                               nu=0.12275754,
                               rho=0.84508075,
                               X_0=-0.02704256,
                               theta=0.16772171,
                               H=0.09164193)
    
    with open("Data/250901.json", 'r', encoding='utf-8') as f:
        option_dict = json.load(f)

    result = calibrator.calibrate(init_param=init_param, option_dict=option_dict)
    with open("results/VSS_calibration_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f)
    print("Optimization result has been saved")