"""
Network parameters for neural network-based Volterra Stein-Stein model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MonotonicNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MonotonicNetwork, self).__init__()
        
        # 定义网络结构
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层到隐层
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)
        self.fc22 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 隐层到输出层
        self.B = nn.Parameter(torch.tensor(1. + 1e-6))

    def forward(self, input):
        # 隐层使用 ReLU 激活函数

        x = input
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc21(x))
        x = F.softplus(self.fc22(x))
        # 输出层使用 Softplus 激活函数，保证输出为正
        x = F.softplus(self.fc3(x))  # Softplus 保证输出恒正
        
        # 确保输入的第二列在计算图中被正确使用
        # 通过显式地创建依赖关系
        s_input = input[:, 1:2]  # 保持维度，shape: (batch_size, 1)
        result = self.B * torch.tanh(s_input * x)
        
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
        super().__init__(input_size, hidden_size, output_size)
        self.device = device

        self.kappa = nn.Parameter(torch.tensor(kappa), requires_grad=True)
        self.nu = torch.tensor(1., dtype=torch.float64, requires_grad=False) 
        self.rho = nn.Parameter(torch.tensor(rho), requires_grad=True)
        self.theta = nn.Parameter(torch.tensor(theta), requires_grad=True)
        self.X_0 = nn.Parameter(torch.tensor(X_0), requires_grad=True)
        super().load_state_dict(torch.load('results/pretrain_network.pth', weights_only=False))
        self.eval()
    
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
    net_params = VSSParamNNTorch()
    print("Network Parameters:")
    for name, value in net_params.to_dict().items():
        print(f"{name}: {value}")