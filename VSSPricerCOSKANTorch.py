"""
Volterra Stein-Stein COS Pricer with KAN implementation
"""

import torch
import torch.nn as nn
from typing import Union, Tuple, Dict
import torch.optim as opt
from math import atanh
import json

from VSSPricerCOSBase import VSSPricerCOSBase
from VSSParamsKANTorch import VSSParamKANTorch


class VSSPricerCOSKANTorch(VSSPricerCOSBase):
    """
    Volterra Stein-Stein COS Pricer with KAN implementation
    """
    
    def __init__(self, params: VSSParamKANTorch, n: int = 252, device: str = 'cpu'):
        """
        初始化KAN实现
        
        Args:
            params: VSSParamKANTorch 参数对象
            n: 时间离散化点数
            device: 计算设备
        """
        super().__init__(params, n, device)
    
    def _compute_kernel_matrices(self, T):
        """
        使用KAN计算核矩阵、协方差矩阵和调整输入向量
        """
        t = torch.linspace(0, T.item(), self.n + 1, dtype=torch.float64, device=self.device)

        tj_1 = t[:-1].unsqueeze(0).repeat(self.n, 1)
        ti_1 = tj_1.T
        tj = t[1:].unsqueeze(0).repeat(self.n, 1)
        
        # 创建KAN输入：ti 和 tj 的组合
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

        # 计算核矩阵
        KK_tj: torch.Tensor = self.params(inputs1).reshape(self.n, self.n)
        KK_tj_1: torch.Tensor = self.params(inputs0).reshape(self.n, self.n)
        
        self.KK = KK_tj - KK_tj_1
        self.KK_sum = self.KK + self.KK.T

        self.KK_mul = self.KK @ self.KK.T
        dt_inv = self.n / T
        self.SIG = self.params.nu**2 * self.KK_mul * dt_inv
        

        g_input = t[:-1].unsqueeze(1).repeat(1, 2).float()
        g_output: torch.Tensor = self.params(g_input)
        self.g = self.params.X_0 + self.params.theta * g_output
        self.g = torch.squeeze(self.g)
    
    def objective(self, option_dict: Dict) -> torch.Tensor:
        """
        校准目标函数：模型价格与市场价格之间的 RMSE + 单调性正则化损失
        
        Args:
            option_dict: 期权数据字典，包含市场价格信息
            
        Returns:
            torch.Tensor: RMSE 损失值 + 单调性正则化损失
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
            
            # 设置时间离散化点数
            self.n = max(32, int(tau.item() * 63))
            
            modelPrice = self.price(S0, strike, r, q, tau)
            self.clear_kernel_matrices()
            error_call = (modelPrice['call'] - torch.tensor(value['price']['call'], dtype=torch.float64, device=self.device)) ** 2
            error_put = (modelPrice['put'] - torch.tensor(value['price']['put'], dtype=torch.float64, device=self.device)) ** 2
            error = torch.cat([error, error_call, error_put])
        
        rmse = torch.sqrt(torch.mean(error))
        
        return rmse
    
    def calibrate(self, option_dict: Dict, init_param: VSSParamKANTorch = None,
                 lr: float = 0.01, tol: float = 1e-3, epochs: int = 1000) -> Dict:
        """
        使用 Adam 优化器训练模型参数以最小化目标函数
        注意：需要处理 _compute_kernel_matrices 中的单调性损失梯度流
        
        Args:
            option_dict: 期权数据字典
            init_param: 初始参数对象，如果为 None 则使用默认值
            lr: 学习率
            tol: 收敛容忍度
            epochs: 最大迭代次数
            
        Returns:
            Dict: 校准结果，包含成功标志、损失值和最终参数
        """
        
        if init_param != None:
            self.params = init_param
        
        self.params.train()

        # 创建优化器，优化所有参数（包括网络参数和神经网络权重）
        # optimizer = opt.LBFGS(self.params.parameters(), lr=lr)
        # optimizer = opt.SGD(self.params.parameters(), lr=lr)
        optimizer = opt.Adam(self.params.parameters(), lr=lr)
        
        
        prev_loss = torch.tensor(torch.inf)

        def closure():          
            return loss
        
        try:
            for epoch in range(epochs):                       
                # 计算目标函数（包含单调性损失）

                loss = self.objective(option_dict)
                
                # 反向传播
                loss.backward()
                # optimizer.step()
                optimizer.step(closure)
                optimizer.zero_grad()
                

                if (epoch + 1) % 1 == 0:
                    print(f"Epoch {epoch+1}: Loss={loss.item():.6f}", end=" | ")
                    print(f"Params: kappa={self.params.kappa.item():.6f}, "
                        f"rho={self.params.rho.item():.6f}, theta={self.params.theta.item():.6f}, "
                        f"X_0={self.params.X_0.item():.6f}")
                
                if abs(loss.item() - prev_loss.item()) < tol:
                    print(f"Optimization finished! The optimum parameters are found within given tolerance ({tol})")
                    # 设置网络为评估模式
                    self.params.eval()
                    return {"suc": True, "loss": loss.item(), "param": self.params.to_dict()}
                prev_loss = loss.detach()

        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
            return {"suc": False, "loss": loss.item(), "param": self.params.to_dict()}
        
        print(f"Optimization finished! The optimum parameters are not found within given tolerance ({tol})")
        # 设置网络为评估模式
        self.params.eval()
        return {"suc": False, "loss": loss.item(), "param": self.params.to_dict()}
    
def main(epochs: int):
    device = 'cpu'
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    params = VSSParamKANTorch(hidden_dim=10, num_grids=64, t_range=(0,6), device=device)
    calibrator = VSSPricerCOSKANTorch(params, device=device)

    with open('Data/250901.json', 'r') as f:
        option_data = json.load(f)
    result = calibrator.calibrate(option_data, lr=0.01, epochs=epochs)
    torch.save(calibrator.params.state_dict(), 'results/KAN_calibrated_network.pth', _use_new_zipfile_serialization=False)

    print("Optimization result has been saved")

def test():
    import time
    device = 'cpu'
    # device = 'cuda'
    params = VSSParamKANTorch(hidden_dim=10, num_grids=64, t_range=(0,6), device=device)
    pricer = VSSPricerCOSKANTorch(params, n=252, device=device)
    with open('Data/250901.json', 'r') as f:
        option_data = json.load(f)
    start = time.time()
    rmse = pricer.objective(option_data)
    print(f"RMSE computied in {time.time() - start:.6f}s")
    print(f"RMSE: {rmse}")

    start = time.time()
    rmse.backward()
    
    print(f'backward done in {time.time() - start:.6f}s')

if __name__ == "__main__":
    main(epochs=100)
    # test()