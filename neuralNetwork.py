import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim

class MonotonicNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MonotonicNetwork, self).__init__()
        
        # 定义网络结构
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层到隐层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 隐层到输出层
        self.B = Parameter(torch.tensor(1.), requires_grad=True)
    

    def forward(self, input):
        # 隐层使用 ReLU 激活函数

        x = input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # 输出层使用 Softplus 激活函数，保证输出为正
        x = F.softplus(self.fc3(x))  # Softplus 保证输出恒正
        
        return self.B * torch.tanh(torch.unsqueeze(input[:,1], 1) * x)

    # 定义一个损失函数，强制 fc2 的输出对输入单调递增
    def monotonicity_loss(self, outputs):
        # 假设我们希望确保网络对第一个输入（x[:, 0]）单调递增
        target_input = inputs[:, 1]  # 获取目标输入（假设是第一个特征）
        
        # 计算输出对第一个输入的梯度
        grad_outputs = torch.autograd.grad(outputs.sum(), target_input, create_graph=True)[0]
        
        # 通过正则化强制梯度为正（即确保对该输入单调递增）
        return torch.mean(torch.relu(-grad_outputs))  # 梯度应该为正，若不正则化则违反单调性    

def train(model, train_data, epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # 目标损失函数（可以根据需要更改）

    for epoch in range(epochs):
        model.train()
        
        # 输入数据
        inputs, targets = train_data  # 假设train_data是一个元组，包括输入和目标输出
        
        optimizer.zero_grad()
        
        # 正常的前向传播
        outputs = model(inputs)
        
        # 计算标准的损失
        loss = criterion(outputs, targets)
        
        # 计算单调性损失并加到总损失中
        monotonic_loss = model.monotonicity_loss(outputs)
        
        # 总损失 = 传统损失 + 单调性损失
        total_loss = loss + monotonic_loss
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {total_loss.item()}')



if __name__ == "__main__":
    input_dim = 2  # 输入维度
    hidden_dim = 64  # 隐藏层大小
    output_dim = 1  # 输出维度

    # 创建模型
    model = MonotonicNetwork(input_dim, hidden_dim, output_dim)

    # 假设训练数据
    inputs = torch.zeros(100, input_dim)  # 随机输入数据
    targets = torch.randn(100, output_dim)  # 随机目标输出

    print(model.forward(inputs[:5]))