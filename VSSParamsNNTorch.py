"""
Network parameters for neural network-based Volterra Stein-Stein model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from math import atanh


class MonotonicNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device='cuda'):
        super(MonotonicNetwork, self).__init__()
        
        # 定义网络结构
        self.q_fc1 = nn.Linear(input_dim, hidden_dim, device=device)  # 输入层到隐层
        self.q_fc2 = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.q_fc3 = nn.Linear(hidden_dim, output_dim, device=device)  # 隐层到输出层
        # self.p_fc1 = nn.Linear(1, hidden_dim, device=device)  # 输入层到隐层
        # self.p_fc2 = nn.Linear(hidden_dim, hidden_dim, device=device)
        # self.p_fc3 = nn.Linear(hidden_dim, output_dim, device=device)  # 隐层到输出层
        self.B = nn.Parameter(torch.tensor(1. + 1e-6, device=device))
        self.K = nn.Parameter(torch.tensor(0.5, device=device))
        # self.alpha = nn.Parameter(torch.tensor(1., device=device))
    def forward(self, input):
        # 隐层使用 ReLU 激活函数

        q = input
        q = F.softplus(self.q_fc1(q))
        q = F.softplus(self.q_fc2(q))
        q = F.softplus(self.q_fc3(q))
        # 输出层使用 Softplus 激活函数，保证输出为正

        s_input = input[:, 1:2]  # 保持维度，shape: (batch_size, 1)
        # p = s_input.clone()
        # p = F.softplus(self.p_fc1(p))
        # p = F.softplus(self.p_fc2(p))
        # p = F.softplus(self.p_fc3(p))  # Softplus 保证输出恒正
        
        # 确保输入的第二列在计算图中被正确使用
        # 通过显式地创建依赖关系
        
        # result = self.B * torch.tanh(s_input * p * q * self.K)
        result = self.B * torch.tanh(s_input * q * self.K)
        # result = q * p
        
        return result

    # # 定义一个损失函数，强制网络对第二个输入特征单调递增
    # def output_and_monotonicity_loss(self, inputs):
    #     # 假设我们希望确保网络对第一个输入（x[:, 0]）单调递增
    #     inputs.requires_grad_(True)

    #     outputs = self.forward(inputs)

    #     # 计算输出对第一个输入的梯度
    #     grad_outputs = torch.autograd.grad(outputs.sum(), inputs, create_graph=True, allow_unused=True)[0]
        
    #     # 通过正则化强制梯度为正（即确保对该输入单调递增）
    #     return outputs, torch.mean(torch.relu(-grad_outputs[:,1]))  # 梯度应该为正，若不正则化则违反单调性 


class VSSParamNNTorch(MonotonicNetwork):
    """
    Network parameters for neural network-based Volterra Stein-Stein model
    """
    def __init__(self, kappa: float = -8.9e-5,
                 rho: float = -0.704,
                 theta: float = -0.044,
                 X_0: float = 0.113,
                 input_size: int = 2,
                 hidden_size: int = 32,
                 output_size: int = 1,
                 device='cpu'):
        super().__init__(input_size, hidden_size, output_size, device=device)
        self.device = device

        self.kappa = nn.Parameter(torch.tensor(kappa, device=device), requires_grad=True)
        self.nu = torch.tensor(1., dtype=torch.float64, device=device, requires_grad=False) 
        self._raw_rho = nn.Parameter(torch.tensor(atanh(rho), device=device), requires_grad=True)
        self.theta = nn.Parameter(torch.tensor(theta, device=device), requires_grad=True)
        self.X_0 = nn.Parameter(torch.tensor(X_0, device=device), requires_grad=True)
        # super().load_state_dict(torch.load('results/pretrain_network.pth', weights_only=False))
        self.eval()
    
    @property
    def rho(self):
        """返回经过 tanh 约束的 rho 值，确保在 (-1, 1) 范围内"""
        return torch.tanh(self._raw_rho)
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Return parameters as dictionary"""
        return {
            'kappa': self.kappa.item(),
            'nu': self.nu.item(),
            'rho': self.rho.item(),
            'theta': self.theta.item(),
            'X_0': self.X_0.item()
        }

if __name__ == "__main__":
    # 测试 NetworkParams 类
    net_params = VSSParamNNTorch(device='cuda')
    print("Network Parameters:")
    for name, value in net_params.to_dict().items():
        print(f"{name}: {value}")