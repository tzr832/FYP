import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from scipy.special import gamma

class MonotonicNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MonotonicNetwork, self).__init__()
        
        # 定义网络结构
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层到隐层
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)
        self.fc22 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 隐层到输出层
        self.B = nn.Parameter(torch.tensor(1.))

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

    # 定义一个损失函数，强制网络对第二个输入特征单调递增
    def output_and_monotonicity_loss(self, inputs):
        # 假设我们希望确保网络对第一个输入（x[:, 0]）单调递增
        inputs.requires_grad_(True)

        outputs = self.forward(inputs)

        # 计算输出对第一个输入的梯度
        grad_outputs = torch.autograd.grad(outputs.sum(), inputs, create_graph=True, allow_unused=True)[0]
        
        # 通过正则化强制梯度为正（即确保对该输入单调递增）
        return outputs, torch.mean(torch.relu(-grad_outputs[:,1]))  # 梯度应该为正，若不正则化则违反单调性 

def train(model: MonotonicNetwork, train_data, epochs=100, lr=0.001):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # 目标损失函数（可以根据需要更改）
    # 输入数据
    inputs, targets = train_data  # 假设train_data是一个元组，包括输入和目标输出
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
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {total_loss.item()}, Monotonicity loss: {monotonic_loss}')

    torch.save(model, 'model.pth')

def gen_test_trainnig_data(n=252, T=1):
    t = np.linspace(0, T, T*n+1)

    tj_1 = np.tile(t[:-1], T*n).reshape(T*n, T*n)  # Times tj excluding the final point
    ti_1 = tj_1.T  # Transpose to create a grid of ti values
    tj = np.tile(t[1:], T*n).reshape(T*n, T*n)  # Times tj excluding the initial point

    alpha = 0.279+0.5 ## H=0.279

    mask = tj <= ti_1
    KK = np.zeros((T*n,T*n))
    KK[mask] = (ti_1 ** alpha - (ti_1 - tj) ** alpha)[mask] / gamma(1 + alpha)

    t_train = ti_1[mask].reshape(-1,1)
    s_train = tj[mask].reshape(-1,1)

    inputs = np.concatenate([t_train, s_train], axis=1)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = KK[mask].reshape(-1,1)
    outputs = torch.tensor(outputs, dtype=torch.float32)

    return inputs, outputs


if __name__ == "__main__":
    input_dim = 2  # 输入维度
    hidden_dim = 64  # 隐藏层大小
    output_dim = 1  # 输出维度

    # 创建模型
    model = MonotonicNetwork(input_dim, hidden_dim, output_dim)

    train_data = gen_test_trainnig_data(n=252*4, T=6)
    inputs, outputs = train_data

    train(model, train_data, epochs=300, lr=0.01)
    print(model.B)
    # monoloss = model.monotonicity_loss(inputs)