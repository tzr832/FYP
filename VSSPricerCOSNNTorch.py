"""
Volterra Stein-Stein COS Pricer with neural network implementation
"""

import torch
import torch.nn as nn
from typing import Union, Tuple, Dict
from torch.optim import Adam
from math import atanh
import json

from VSSPricerCOSBase import VSSPricerCOSBase
from VSSParamsNNTorch import VSSParamNNTorch


class VSSPricerCOSNNTorch(VSSPricerCOSBase):
    """
    Volterra Stein-Stein COS Pricer with neural network implementation
    """
    
    def __init__(self, params: VSSParamNNTorch, n: int = 252, device: str = 'cpu'):
        """
        初始化神经网络实现
        
        Args:
            params: VSSParamNNTorch 参数对象
            n: 时间离散化点数
            device: 计算设备
        """
        super().__init__(params, n, device)
    
    def _compute_kernel_matrices(self, T):
        """
        使用神经网络计算核矩阵、协方差矩阵和调整输入向量
        """
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

        # 计算核矩阵
        KK_tj: torch.Tensor = self.params(inputs1).reshape(self.n, self.n)
        KK_tj_1: torch.Tensor = self.params(inputs0).reshape(self.n, self.n)
        
        KK = KK_tj - KK_tj_1
        self.KK = KK * torch.tensor(0.1)
        self.KK_sum = self.KK + self.KK.T

        self.KK_mul = self.KK @ self.KK.T
        dt_inv = self.n / T
        self.SIG = self.KK_mul * dt_inv
        

        g_input = t[:-1].unsqueeze(1).repeat(1, 2).float()
        g_output: torch.Tensor = self.params(g_input)
        self.g = self.params.X_0 + self.params.theta * g_output
        self.g = torch.squeeze(self.g)

    def _compute_monotonicity_loss(self, T: torch.Tensor = None) -> torch.Tensor:
        """
        计算神经网络对第二个输入参数的单调性损失
        
        Args:
            T: 到期时间，如果为 None 则使用默认值 1.0
            
        Returns:
            torch.Tensor: 单调性损失值
        """
        if T is None:
            # 如果没有提供 T，使用默认值
            T = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        
        # 生成测试点来检查单调性
        t = torch.linspace(0, T.item(), self.n + 1, dtype=torch.float64, device=self.device)
        
        # 创建测试输入：固定第一个参数，变化第二个参数
        # 使用多个固定时间点作为第一个参数
        num_test_points = min(10, self.n)  # 使用较少的测试点以提高效率
        test_times = t[torch.linspace(0, self.n, num_test_points, dtype=torch.long)]
        
        # 为每个固定时间点创建变化的第二个参数
        monotonic_loss = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        
        for t_fixed in test_times:
            # 创建变化的第二个参数，确保 t > s
            # 第二个参数 s 应该小于第一个参数 t
            s_max = t_fixed.item() * 0.99  # 确保 s < t，留一点余量
            if s_max > 0:
                s_values = torch.linspace(0, s_max, 20, dtype=torch.float64, device=self.device)
            else:
                # 如果 t_fixed 太小，跳过这个测试点
                continue
            
            # 构建输入：固定第一个参数，变化第二个参数
            inputs = torch.stack([
                torch.full_like(s_values, t_fixed.item()),
                s_values
            ], dim=1).float()
            
            # 验证输入满足 t > s 条件
            assert torch.all(inputs[:, 0] > inputs[:, 1]), "输入必须满足 t > s 条件"
            
            # 启用梯度计算
            inputs.requires_grad_(True)
            
            # 计算网络输出
            outputs = self.params(inputs).squeeze()
            
            # 计算输出对第二个输入的梯度
            # 使用自动微分计算梯度
            grad_outputs = torch.autograd.grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=torch.ones_like(outputs),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # 提取对第二个输入的梯度
            grad_wrt_s = grad_outputs[:, 1]
            
            # 单调性损失：惩罚负梯度（即非单调递减）
            # 使用 ReLU 惩罚负梯度值
            negative_grad_penalty = torch.relu(-grad_wrt_s)
            monotonic_loss += negative_grad_penalty.mean()
        
        # 返回平均单调性损失
        if monotonic_loss > 0:
            return monotonic_loss / len(test_times)
        else:
            return torch.tensor(0.0, dtype=torch.float64, device=self.device)
    
    def objective(self, option_dict: Dict, monotonic_weight: float = 0.1) -> torch.Tensor:
        """
        校准目标函数：模型价格与市场价格之间的 RMSE + 单调性正则化损失
        
        Args:
            option_dict: 期权数据字典，包含市场价格信息
            monotonic_weight: 单调性正则化权重
            
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
    
        # 计算单调性损失（使用最后一个期权的到期时间）
        monotonic_loss = self._compute_monotonicity_loss(tau)
        
        # 总损失 = RMSE + 单调性正则化损失
        total_loss = rmse + monotonic_weight * monotonic_loss
        
        return total_loss
    
    def calibrate(self, option_dict: Dict, init_param: VSSParamNNTorch = None,
                 lr: float = 0.01, tol: float = 1e-3, epochs: int = 1000,
                 monotonic_weight: float = 0.1) -> Dict:
        """
        使用 Adam 优化器训练模型参数以最小化目标函数
        注意：需要处理 _compute_kernel_matrices 中的单调性损失梯度流
        
        Args:
            option_dict: 期权数据字典
            init_param: 初始参数对象，如果为 None 则使用默认值
            lr: 学习率
            tol: 收敛容忍度
            epochs: 最大迭代次数
            monotonic_weight: 单调性正则化权重
            
        Returns:
            Dict: 校准结果，包含成功标志、损失值和最终参数
        """
        if init_param is None:
            # 使用默认参数
            self.params = VSSParamNNTorch(device=self.device)
        else:
            self.params = init_param
        
        self.params.train()

        # 创建优化器，优化所有参数（包括网络参数和神经网络权重）
        optimizer = Adam(self.params.parameters(), lr=lr)
        
        prev_loss = torch.tensor(torch.inf)
        
        for epoch in range(epochs):                       
            # 计算目标函数（包含单调性损失）
            loss = self.objective(option_dict, monotonic_weight)
            
            # 反向传播
            loss.backward()
            optimizer.step()
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
        
        print(f"Optimization finished! The optimum parameters are not found within given tolerance ({tol})")
        # 设置网络为评估模式
        self.params.eval()
        return {"suc": False, "loss": loss.item(), "param": self.params.to_dict()}
    
def main():
    device = 'cpu'
    print(f"Using device: {device}")
    
    torch.manual_seed(42)

    calibrator = VSSPricerCOSNNTorch(VSSParamNNTorch(), device=device)

    result = calibrator.calibrate()
    torch.save(calibrator.params.state_dict(), 'results/NN_calibrated_network.pth', _use_new_zipfile_serialization=False)

    print("Optimization result has been saved")