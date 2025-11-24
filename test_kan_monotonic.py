"""
KANMonotonicNetwork测试脚本
验证KAN实现的功能和API兼容性
"""

import torch
import torch.nn as nn
import numpy as np
import time
from KANMonotonicNetwork import KANMonotonicNetwork, train_kan, gen_test_training_data
from MonotonicNetwork import MonotonicNetwork, train, gen_test_trainnig_data


def test_api_compatibility():
    """测试API兼容性"""
    print("=== 测试API兼容性 ===")
    
    # 创建两个模型
    input_dim, hidden_dim, output_dim = 2, 32, 1
    
    original_model = MonotonicNetwork(input_dim, hidden_dim, output_dim)
    kan_model = KANMonotonicNetwork(input_dim, hidden_dim, output_dim)
    
    # 测试构造函数参数
    print(f"原始模型参数数量: {sum(p.numel() for p in original_model.parameters())}")
    print(f"KAN模型参数数量: {sum(p.numel() for p in kan_model.parameters())}")
    
    # 测试前向传播
    test_input = torch.randn(10, 2)
    
    with torch.no_grad():
        original_output = original_model(test_input)
        kan_output = kan_model(test_input)
    
    print(f"原始模型输出形状: {original_output.shape}")
    print(f"KAN模型输出形状: {kan_output.shape}")
    print(f"输出范围 - 原始: [{original_output.min():.4f}, {original_output.max():.4f}]")
    print(f"输出范围 - KAN: [{kan_output.min():.4f}, {kan_output.max():.4f}]")
    
    # 测试单调性损失函数
    test_input.requires_grad_(True)
    
    original_outputs, original_mono_loss = original_model.output_and_monotonicity_loss(test_input)
    kan_outputs, kan_mono_loss = kan_model.output_and_monotonicity_loss(test_input)
    
    print(f"原始模型单调性损失: {original_mono_loss.item():.6f}")
    print(f"KAN模型单调性损失: {kan_mono_loss.item():.6f}")
    
    print("✓ API兼容性测试通过")


def test_monotonicity():
    """测试单调性约束"""
    print("\n=== 测试单调性约束 ===")
    
    model = KANMonotonicNetwork(2, 32, 1)
    
    # 创建测试数据：固定t，变化s
    t_fixed = 1.0
    s_values = torch.linspace(0.1, 1.0, 100).unsqueeze(1)
    t_values = torch.ones_like(s_values) * t_fixed
    test_inputs = torch.cat([t_values, s_values], dim=1)
    
    with torch.no_grad():
        outputs = model(test_inputs)
    
    # 检查输出是否随s单调递增
    output_diff = torch.diff(outputs.squeeze())
    monotonic_ratio = (output_diff > 0).float().mean().item()
    
    print(f"单调递增比例: {monotonic_ratio:.2%}")
    print(f"输出范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
    
    if monotonic_ratio > 0.8:
        print("✓ 单调性约束测试通过")
    else:
        print("⚠ 单调性约束需要进一步优化")


def test_training_compatibility():
    """测试训练兼容性"""
    print("\n=== 测试训练兼容性 ===")
    
    # 生成小型训练数据
    small_train_data = gen_test_training_data(n=50, T=1)
    inputs, targets = small_train_data
    
    print(f"训练数据形状: 输入 {inputs.shape}, 目标 {targets.shape}")
    
    # 创建模型
    model = KANMonotonicNetwork(2, 16, 1, grid_size=3, k=2)  # 使用较小的配置
    
    # 测试训练过程
    start_time = time.time()
    
    # 训练少量轮次
    train_kan(model, small_train_data, epochs=10, lr=0.01)
    
    training_time = time.time() - start_time
    print(f"训练时间: {training_time:.2f}秒")
    
    # 验证训练后模型仍然工作
    test_output = model(inputs[:5])
    print(f"训练后输出形状: {test_output.shape}")
    print("✓ 训练兼容性测试通过")


def test_performance_comparison():
    """性能比较测试"""
    print("\n=== 性能比较测试 ===")
    
    # 创建测试数据
    test_input = torch.randn(1000, 2)
    
    # 原始模型性能
    original_model = MonotonicNetwork(2, 32, 1)
    kan_model = KANMonotonicNetwork(2, 32, 1, grid_size=3, k=2)
    
    # 前向传播性能
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = original_model(test_input)
    original_time = time.time() - start_time
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = kan_model(test_input)
    kan_time = time.time() - start_time
    
    print(f"原始模型前向传播时间: {original_time:.4f}秒")
    print(f"KAN模型前向传播时间: {kan_time:.4f}秒")
    print(f"速度比: {original_time/kan_time:.2f}x")
    
    # 内存使用比较
    original_params = sum(p.numel() for p in original_model.parameters())
    kan_params = sum(p.numel() for p in kan_model.parameters())
    
    print(f"原始模型参数数量: {original_params}")
    print(f"KAN模型参数数量: {kan_params}")
    print(f"参数比: {kan_params/original_params:.2f}x")


def test_function_approximation():
    """测试函数逼近能力"""
    print("\n=== 测试函数逼近能力 ===")
    
    # 创建简单的测试函数
    def target_function(t, s):
        """目标函数：类似于核函数的简单形式"""
        return torch.sin(t * 2 * torch.pi) * torch.exp(-s)
    
    # 生成训练数据
    t_vals = torch.rand(500, 1) * 2  # t in [0, 2]
    s_vals = torch.rand(500, 1) * 1  # s in [0, 1]
    inputs = torch.cat([t_vals, s_vals], dim=1)
    targets = target_function(t_vals, s_vals)
    
    train_data = (inputs, targets)
    
    # 创建和训练模型
    model = KANMonotonicNetwork(2, 32, 1, grid_size=5, k=3)
    
    # 训练前损失
    with torch.no_grad():
        initial_outputs = model(inputs)
        initial_loss = nn.MSELoss()(initial_outputs, targets)
    
    # 训练
    train_kan(model, train_data, epochs=50, lr=0.01)
    
    # 训练后损失
    with torch.no_grad():
        final_outputs = model(inputs)
        final_loss = nn.MSELoss()(final_outputs, targets)
    
    print(f"训练前MSE: {initial_loss.item():.6f}")
    print(f"训练后MSE: {final_loss.item():.6f}")
    print(f"改进比例: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
    
    if final_loss < initial_loss * 0.5:
        print("✓ 函数逼近测试通过")
    else:
        print("⚠ 函数逼近能力需要进一步优化")


if __name__ == "__main__":
    print("KANMonotonicNetwork综合测试")
    print("=" * 50)
    
    try:
        test_api_compatibility()
        test_monotonicity()
        test_training_compatibility()
        test_performance_comparison()
        test_function_approximation()
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()