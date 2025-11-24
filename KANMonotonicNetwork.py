import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.special import gamma
from typing import Tuple, Optional
import math


class BSplineActivation(nn.Module):
    """
    B样条基函数激活层
    实现KAN网络中的B样条基函数作为激活函数
    使用向量化实现提高效率
    """
    
    def __init__(self, grid_size: int = 5, k: int = 3, grid_range: Tuple[float, float] = (-1, 1)):
        """
        初始化B样条激活函数
        
        Args:
            grid_size: 网格点数量
            k: B样条阶数
            grid_range: 输入范围
        """
        super(BSplineActivation, self).__init__()
        self.grid_size = grid_size
        self.k = k
        self.grid_range = grid_range
        
        # 创建均匀分布的网格点（扩展边界以支持B样条计算）
        extended_grid_size = grid_size + 2 * k
        extended_range = (
            grid_range[0] - k * (grid_range[1] - grid_range[0]) / (grid_size - 1),
            grid_range[1] + k * (grid_range[1] - grid_range[0]) / (grid_size - 1)
        )
        
        self.extended_grid = nn.Parameter(
            torch.linspace(extended_range[0], extended_range[1], extended_grid_size),
            requires_grad=False
        )
        
        # 可学习的B样条系数
        self.coefficients = nn.Parameter(torch.randn(grid_size + k - 1) * 0.1)
        
    def _compute_bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        向量化计算B样条基函数
        
        Args:
            x: 输入张量，形状为 (batch_size, 1)
            
        Returns:
            B样条基函数矩阵，形状为 (batch_size, num_coefficients)
        """
        batch_size = x.shape[0]
        num_coefficients = len(self.coefficients)
        
        # 初始化基函数矩阵
        basis_matrix = torch.zeros(batch_size, num_coefficients, device=x.device)
        
        # 对每个系数计算对应的B样条基函数
        for i in range(num_coefficients):
            # 使用简化的B样条计算（避免递归）
            basis = self._simple_bspline(x, i, self.k)
            basis_matrix[:, i] = basis.squeeze()
            
        return basis_matrix
    
    def _simple_bspline(self, x: torch.Tensor, i: int, k: int) -> torch.Tensor:
        """
        简化的B样条基函数计算
        使用分段线性近似避免递归
        
        Args:
            x: 输入张量
            i: 基函数索引
            k: 样条阶数
            
        Returns:
            近似的B样条基函数值
        """
        # 检查索引是否在有效范围内
        if i + k >= len(self.extended_grid):
            return torch.zeros_like(x)
            
        # 对于原型实现，使用分段线性函数近似B样条
        # 在实际应用中可以使用更精确的实现
        if k == 0:
            # 零阶B样条（矩形函数）
            if i + 1 >= len(self.extended_grid):
                return torch.zeros_like(x)
            return ((x >= self.extended_grid[i]) &
                    (x < self.extended_grid[i+1])).float()
        else:
            # 使用分段线性函数近似
            if i + k >= len(self.extended_grid):
                return torch.zeros_like(x)
                
            center = (self.extended_grid[i] + self.extended_grid[i+k]) / 2
            width = (self.extended_grid[i+k] - self.extended_grid[i]) / 2
            
            # 避免除零
            if width == 0:
                return torch.zeros_like(x)
                
            # 三角波函数近似
            distance = torch.abs(x - center)
            basis = torch.clamp(1 - distance / width, 0, 1)
            
            return basis
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, 1)
            
        Returns:
            B样条激活后的输出，形状为 (batch_size, 1)
        """
        # 将输入缩放到网格范围内
        x_scaled = torch.clamp(x, self.grid_range[0], self.grid_range[1])
        
        # 计算B样条基函数矩阵
        basis_matrix = self._compute_bspline_basis(x_scaled)
        
        # 计算加权和
        output = torch.matmul(basis_matrix, self.coefficients)
        
        return output.unsqueeze(-1)


class KANLayer(nn.Module):
    """
    KAN网络层
    实现Kolmogorov-Arnold表示中的单层网络
    使用更高效的向量化实现
    """
    
    def __init__(self, input_dim: int, output_dim: int, grid_size: int = 5, k: int = 3):
        """
        初始化KAN层
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            grid_size: B样条网格大小
            k: B样条阶数
        """
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.k = k
        
        # 为每个输入维度创建B样条激活函数
        # 每个输入维度对应output_dim个激活函数
        self.activations = nn.ModuleList([
            nn.ModuleList([
                BSplineActivation(grid_size, k) for _ in range(output_dim)
            ]) for _ in range(input_dim)
        ])
        
        # 线性组合权重
        self.linear_weights = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, input_dim)
            
        Returns:
            输出张量，形状为 (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        # 对每个输出维度
        for j in range(self.output_dim):
            # 对每个输入维度
            for i in range(self.input_dim):
                # 获取对应的B样条激活函数
                activation_output = self.activations[i][j](x[:, i:i+1])
                
                # 应用线性权重
                output[:, j] += self.linear_weights[j, i] * activation_output.squeeze()
                
        return output


class KANMonotonicNetwork(nn.Module):
    """
    基于Kolmogorov-Arnold Networks的单调网络
    保持与原始MonotonicNetwork相同的API接口
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 grid_size: int = 5, k: int = 3):
        """
        初始化KAN单调网络
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            grid_size: B样条网格大小
            k: B样条阶数
        """
        super(KANMonotonicNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # KAN网络层
        self.kan1 = KANLayer(input_dim, hidden_dim, grid_size, k)
        self.kan2 = KANLayer(hidden_dim, hidden_dim, grid_size, k)
        self.kan_out = KANLayer(hidden_dim, output_dim, grid_size, k)
        
        # 保持与原始网络相同的参数
        self.B = nn.Parameter(torch.tensor(1.))
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input: 输入张量，形状为 (batch_size, input_dim)
            
        Returns:
            输出张量，形状为 (batch_size, output_dim)
        """
        x = input
        
        # 通过KAN层
        x = self.kan1(x)  # 使用Softplus保持正输出
        x = self.kan2(x)
        x = F.softplus(self.kan_out(x))  # 输出层使用Softplus保证输出恒正
        
        # 保持与原始网络相同的输出结构
        s_input = input[:, 1:2]  # 保持维度，shape: (batch_size, 1)
        result = self.B * torch.tanh(s_input * x)
        
        return result
    
    def output_and_monotonicity_loss(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算输出和单调性损失
        保持与原始网络相同的API
        
        Args:
            inputs: 输入张量
            
        Returns:
            Tuple[输出张量, 单调性损失]
        """
        inputs.requires_grad_(True)
        
        outputs = self.forward(inputs)
        
        # 计算输出对输入的梯度
        grad_outputs = torch.autograd.grad(
            outputs.sum(), inputs, create_graph=True, allow_unused=True
        )[0]
        
        # 通过正则化强制梯度为正（即确保对第二个输入单调递增）
        monotonic_loss = torch.mean(torch.relu(-grad_outputs[:, 1]))
        
        return outputs, monotonic_loss


def train_kan(model: KANMonotonicNetwork, train_data: Tuple[torch.Tensor, torch.Tensor], 
              epochs: int = 100, lr: float = 0.001):
    """
    训练KAN单调网络
    保持与原始训练函数相同的API
    
    Args:
        model: KAN单调网络模型
        train_data: 训练数据元组 (inputs, targets)
        epochs: 训练轮数
        lr: 学习率
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    inputs, targets = train_data
    inputs.requires_grad_(True)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 正常的前向传播
        outputs, monotonic_loss = model.output_and_monotonicity_loss(inputs)
        
        # 计算标准的损失
        loss = criterion(outputs, targets)
        
        # 总损失 = 传统损失 + 单调性损失
        total_loss = loss + monotonic_loss
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        if epoch % 1 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {total_loss.item():.6f}, '
                  f'Mono Loss: {monotonic_loss.item():.6f}')
    
    torch.save(model.state_dict(), 'kan_model.pth')


def gen_test_training_data(n: int = 252, T: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成测试训练数据
    保持与原始函数相同的API
    
    Args:
        n: 时间步数
        T: 总时间
        
    Returns:
        Tuple[输入张量, 输出张量]
    """
    t = np.linspace(0, T, T * n + 1)
    
    tj_1 = np.tile(t[:-1], T * n).reshape(T * n, T * n)  # Times tj excluding the final point
    ti_1 = tj_1.T  # Transpose to create a grid of ti values
    tj = np.tile(t[1:], T * n).reshape(T * n, T * n)  # Times tj excluding the initial point
    
    alpha = 0.279 + 0.5  # H=0.279
    
    mask = tj <= ti_1
    KK = np.zeros((T * n, T * n))
    KK[mask] = (ti_1 ** alpha - (ti_1 - tj) ** alpha)[mask] / gamma(1 + alpha)
    
    t_train = ti_1[mask].reshape(-1, 1)
    s_train = tj[mask].reshape(-1, 1)
    
    inputs = np.concatenate([t_train, s_train], axis=1)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = KK[mask].reshape(-1, 1)
    outputs = torch.tensor(outputs, dtype=torch.float32)
    
    return inputs, outputs


if __name__ == "__main__":
    # 测试KAN单调网络
    input_dim = 2
    hidden_dim = 8
    output_dim = 1
    
    load_exist_model = True
    training = False
    # 创建KAN模型
    if load_exist_model:
        kan_model = KANMonotonicNetwork(input_dim, hidden_dim, output_dim)
    else:
        kan_model = KANMonotonicNetwork(input_dim, hidden_dim, output_dim)
        kan_model.load_state_dict(torch.load('kan_model.pth', weights_only=False))
        kan_model.eval()
    # 生成训练数据
    train_data = gen_test_training_data(n=252, T=1)
    inputs, outputs = train_data
    
    print(f"输入形状: {inputs.shape}, 输出形状: {outputs.shape}")
    print(f"KAN模型参数数量: {sum(p.numel() for p in kan_model.parameters())}")
    
    # 训练模型
    if training:
        train_kan(kan_model, train_data, epochs=50, lr=0.01)

    # out = kan_model(inputs[:5])
    # print(out)
    
    test_input = torch.cat([torch.ones(100,1), torch.linspace(0,1,100).unsqueeze(1)], dim=1).float()
    test_output = kan_model(test_input)
    alpha = torch.tensor(0.779)
    test_target = (test_input[:,0] ** alpha - (test_input[:,0] - test_input[:,1]) ** alpha) / gamma(1 + alpha)

    import matplotlib.pyplot as plt

    plt.plot(test_input[:,1].detach(), test_output.detach(), label='NN')
    plt.plot(test_input[:,1].detach(), test_target, label='exact')
    plt.legend()
    plt.show()
