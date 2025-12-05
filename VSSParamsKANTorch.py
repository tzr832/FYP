"""
Network parameters for KAN-based Volterra Stein-Stein model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from math import atanh

from monotonic_kan_layer import MonotonicKANLayer


class VSSParamKANTorch(MonotonicKANLayer):
    """
    Network parameters for KAN-based Volterra Stein-Stein model
    """
    def __init__(self, kappa: float = -8.9e-5,
                 rho: float = -0.704,
                 theta: float = -0.044,
                 X_0: float = 0.113,
                 hidden_dim: int = 32,
                 num_grids: int = 10,
                 k: int = 3,
                 s_range: tuple = (0, 5),
                 t_range: tuple = (0, 5),
                 device='cpu'):
        # 调用 MonotonicKANLayer 的初始化
        super().__init__(hidden_dim=hidden_dim, num_grids=num_grids, k=k,
                         t_range=t_range, device=device)
        self.device = device

        self.kappa = nn.Parameter(torch.tensor(kappa, device=device), requires_grad=True)
        self.nu = torch.tensor(0.1, dtype=torch.float64, device=device, requires_grad=False) 
        self._raw_rho = nn.Parameter(torch.tensor(atanh(rho), device=device), requires_grad=True)
        self.theta = nn.Parameter(torch.tensor(theta, device=device), requires_grad=True)
        self.X_0 = nn.Parameter(torch.tensor(X_0, device=device), requires_grad=True)
        # 可选：加载预训练权重
        super().load_state_dict(torch.load('results/monotonic_kan_model.pth', weights_only=False))
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
    # 测试 VSSParamKANTorch 类
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    net_params = VSSParamKANTorch(device=device)
    print("KAN Network Parameters:")
    for name, value in net_params.to_dict().items():
        print(f"{name}: {value}")
    
    # 测试前向传播
    batch = 5
    x = torch.randn(batch, 2, device=device)
    output = net_params(x)
    print(f"Forward output shape: {output.shape}")
    print(f"Sample output: {output[:3]}")