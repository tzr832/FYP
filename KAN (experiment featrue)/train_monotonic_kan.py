import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import sys
import os

# 添加当前目录以导入monotonic_kan_layer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from VSSParamsKANTorch_experiment import VSSParamKANTorch


def gen_pretrain_data_noneven(n=252, T=1):
    """
    返回输入(t,s)和目标核函数值。
    """

    dt = T / n
    T_grid = torch.linspace(dt, T, n)  # 时间点从dt到T，共n个点
    for i, t in enumerate(T_grid):
        s_grid = torch.linspace(0, t, n)        
        inputs_temp = torch.stack([t.repeat(n), s_grid], dim=1)

        alpha = 0.279 + 0.5  # H=0.279
        gamma_value = gamma(1 + alpha)
        target_temp = (t.item() ** alpha - (t.item() - s_grid) ** alpha) / gamma_value
        
        if i == 0:
            inputs = inputs_temp
            outputs = target_temp.unsqueeze(1)
        else:
            inputs = torch.cat([inputs, inputs_temp], dim=0)
            outputs = torch.cat([outputs, target_temp.unsqueeze(1)], dim=0)

    return inputs, outputs


def train(model: VSSParamKANTorch, inputs, targets, epochs=500, lr=0.001, smoothpanalty=True):
    """
    训练模型。
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    targets = targets / inputs[:,0:1]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(inputs)
        outputs = outputs / inputs[:,0:1]
        loss = criterion(outputs, targets)

        if smoothpanalty:
            spline_coef_H = F.softplus(model.H_s_func.d[:,:,1:])
            smooth_loss_H = torch.sum((spline_coef_H[1:] - spline_coef_H[:-1]) ** 2)
            spline_coef_a = F.softplus(model.a_s_func.d[:,:,1:])
            smooth_loss_a = torch.sum((spline_coef_a[1:] - spline_coef_a[:-1]) ** 2)
            total_loss = loss + 0.001 * (smooth_loss_H + smooth_loss_a)
        else:
            total_loss = loss

        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {total_loss.item()}')

    return model


def visualize(trained_model, device='cuda'):
    """
    可视化模型输出与精确值的对比。
    """
    fig, ax = plt.subplots(2, 3, figsize=(21, 14))
    ax = ax.flatten()
    for i, t_val in enumerate([0.1, 0.5, 1, 2, 3, 5]):
        t_test = torch.ones(800, 1, device=device) * t_val
        s_test = torch.linspace(0, t_val, 800, device=device).unsqueeze(1)

        dt = s_test[1] - s_test[0]

        trained_model = trained_model.to(device)

        input_test = torch.cat([t_test, s_test], dim=1).float()
        output_test = trained_model(input_test)
        alpha = torch.tensor(0.279 + 0.5, device=device)  # H=0.279
        gamma_value = gamma(1 + alpha.cpu().numpy())
        gamma_value = torch.tensor(gamma_value, device=device)
        target_test = (t_test ** alpha - (t_test - s_test) ** alpha) / gamma_value

        output_kernal = (output_test[1:] - output_test[:-1]) / dt
        target_kernal = (target_test[1:] - target_test[:-1]) / dt

        
        ax[i].plot(s_test[1:].cpu().detach().numpy(), target_kernal.cpu().detach().numpy(), label='exact value')
        ax[i].plot(s_test[1:].cpu().detach().numpy(), output_kernal.cpu().detach().numpy(), label='network')
        ax[i].legend()
        ax[i].set_title(f'Kernel Approximation (T={t_val})')
    plt.show()


if __name__ == "__main__":
    # 超参数
    hidden_dim = 10          # 隐藏维度 H
    num_grids = 64           # B样条网格数
    k = 3                    # B样条阶数
    t_range = (0, 6)         # t的范围
    epochs = 500
    learning_rate = 0.05

    # 训练标志（可根据需要修改）
    train_the_model = True
    # train_the_model = False
    save_the_model = True
    # save_the_model = False
    load_the_model = True 
    # load_the_model = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 生成数据
    inputs, targets = gen_pretrain_data_noneven(1024, 6)
    inputs = inputs.to(device)
    targets = targets.to(device)
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")

    # 创建模型
    model = VSSParamKANTorch(
        hidden_dim=hidden_dim,
        num_grids=num_grids,
        k=k,
        t_range=t_range,
        device=device
    )
    model.to(device)

    # 计算参数量
    param_num = sum(p.numel() for p in model.parameters())
    print(f"Model created with {param_num} parameters")

    model_save_path = 'KAN (experiment featrue)/monotonic_kan_model_new.pth'

    if load_the_model and os.path.exists(model_save_path):
        print(f"Loading model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()
        trained_model = model
    if train_the_model:
        print("Training model...")
        trained_model = train(model, inputs, targets, epochs=epochs, lr=learning_rate)
        if save_the_model:
            torch.save(trained_model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
    else:
        trained_model = model
        print("Using untrained model for visualization.")

    # import torch.nn.functional as F
    # spline_coef = F.softplus(trained_model.H_s_func.d)
    # print(spline_coef.max(), spline_coef.min())
    # 可视化
    visualize(trained_model, device=device)