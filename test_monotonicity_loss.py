"""
测试单调性损失函数的实现
"""

import torch
from VSSPricerCOSNNTorch import VSSPricerCOSNNTorch
from VSSParamsNNTorch import VSSParamNNTorch

def test_monotonicity_loss():
    """测试单调性损失函数"""
    device = 'cpu'
    print(f"使用设备: {device}")
    
    # 创建神经网络参数和定价器
    params = VSSParamNNTorch(device=device)
    pricer = VSSPricerCOSNNTorch(params, device=device)
    
    # 测试单调性损失计算
    print("测试单调性损失函数...")
    
    # 使用不同的到期时间测试
    test_T_values = [torch.tensor(0.5), torch.tensor(1.0), torch.tensor(2.0)]
    
    for T in test_T_values:
        print(f"\n测试到期时间 T={T.item()}:")
        
        # 计算单调性损失
        monotonic_loss = pricer._compute_monotonicity_loss(T)
        print(f"单调性损失值: {monotonic_loss.item():.6f}")
        
        # 检查损失是否为标量且非负
        assert monotonic_loss.dim() == 0, "单调性损失应该是标量"
        assert monotonic_loss.item() >= 0, "单调性损失应该非负"
        
        # 只有当损失不为0时才检查梯度
        if monotonic_loss.item() > 0:
            # 检查梯度是否可以计算
            monotonic_loss.backward()
            print("梯度计算成功")
            
            # 清除梯度
            params.zero_grad()
        else:
            print("单调性损失为0，跳过梯度检查")
    
    print("\n单调性损失函数测试通过！")

def test_input_condition():
    """测试输入是否满足 t > s 条件"""
    print("\n测试输入条件 t > s...")
    
    device = 'cpu'
    params = VSSParamNNTorch(device=device)
    pricer = VSSPricerCOSNNTorch(params, device=device)
    
    # 手动创建一些测试输入来验证条件
    T = torch.tensor(1.0)
    t = torch.linspace(0, T.item(), pricer.n + 1, dtype=torch.float64, device=device)
    
    # 测试几个固定时间点
    test_times = t[torch.linspace(0, pricer.n, 5, dtype=torch.long)]
    
    for t_fixed in test_times:
        if t_fixed > 0:
            s_max = t_fixed.item() * 0.99
            s_values = torch.linspace(0, s_max, 5, dtype=torch.float64, device=device)
            
            inputs = torch.stack([
                torch.full_like(s_values, t_fixed.item()),
                s_values
            ], dim=1).float()
            
            # 验证 t > s 条件
            print(f"t_fixed={t_fixed.item():.3f}, s_values={s_values.tolist()}")
            assert torch.all(inputs[:, 0] > inputs[:, 1]), f"输入不满足 t > s 条件: t={inputs[:, 0]}, s={inputs[:, 1]}"
            print("✓ 输入满足 t > s 条件")
    
    print("所有输入都满足 t > s 条件！")

def test_with_sample_data():
    """使用样本数据测试"""
    print("\n使用样本数据测试...")
    
    # 创建样本期权数据
    sample_option_data = {
        'HSI': 18000.0,
        'rf': 0.03,
        'option1': {
            'tau': 0.25,
            'strike': {
                'call': [17500, 18000, 18500],
                'put': [17500, 18000, 18500]
            },
            'price': {
                'call': [800, 500, 300],
                'put': [300, 500, 800]
            }
        }
    }
    
    device = 'cpu'
    params = VSSParamNNTorch(device=device)
    pricer = VSSPricerCOSNNTorch(params, device=device)
    
    # 测试目标函数（包含单调性损失）
    total_loss = pricer.objective(sample_option_data, monotonic_weight=0.1)
    print(f"总损失值: {total_loss.item():.6f}")
    
    # 检查梯度计算
    total_loss.backward()
    print("总损失梯度计算成功")
    
    print("样本数据测试通过！")

if __name__ == "__main__":
    test_monotonicity_loss()
    test_input_condition()
    test_with_sample_data()