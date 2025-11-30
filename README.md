# Volterra Stein-Stein 模型实现
中文 | [English](./README_EN.md)

本项目实现了基于 Volterra Stein-Stein 模型的欧式期权定价，使用 COS Fourier 方法 [(Fang & Oosterlee, 2009)](https://doi.org/10.1137/080718061)。该模型基于 [Abi Jaber (2022)](https://doi.org/10.1007/s00780-022-00489-4) 提出的高斯随机波动率模型特征函数解析表达式。

## 项目概述

本项目提供了三种不同的 Volterra Stein-Stein 模型实现：

1. **标准核实现** ([`VSSPricerCOSTorch.py`](VSSPricerCOSTorch.py)) - 使用标准分数阶核函数
2. **指数核实现** ([`VSSPricerCOSLaplaceTorch.py`](VSSPricerCOSLaplaceTorch.py)) - 使用指数衰减核函数
3. **神经网络实现** ([`VSSPricerCOSNNTorch.py`](VSSPricerCOSNNTorch.py)) - 使用神经网络学习核函数

所有实现都基于 PyTorch 框架，支持 GPU 加速和自动微分。

## 主要特性

- ✅ 欧式看涨/看跌期权定价
- ✅ 模型参数校准
- ✅ 支持多种核函数实现
- ✅ 基于 COS Fourier 方法的快速定价
- ✅ PyTorch 自动微分支持
- ✅ 数值稳定性优化（旋转计数算法）
- ✅ 参数约束验证（0 < H < 1, -1 < rho < 1）

## 安装和依赖

### 必需依赖

```bash
pip install torch numpy scipy numdifftools pandas exchange-calendars matplotlib
```

### 可选依赖

```bash
# 用于 Jupyter notebook 实验
pip install jupyter
```

### 系统要求

- Python 3.12
- PyTorch 2.9.1
- NumPy, SciPy, Pandas
- 支持 CUDA 的 GPU（可选，用于加速计算）

## 项目结构

```
.
├── README.md                          # 项目说明文档
├── VSSPricerCOSBase.py               # 基类实现，包含共同功能
├── VSSPricerCOSTorch.py              # 标准核实现
├── VSSPricerCOSLaplaceTorch.py       # 指数核实现
├── VSSPricerCOSNNTorch.py            # 神经网络实现
├── VSSParamTorch.py                  # 标准模型参数类
├── VSSParamLaplaceTorch.py           # 指数核参数类
├── VSSParamsNNTorch.py               # 神经网络参数类
├── hyp2f1_numerical.py               # 超几何函数数值实现
├── BS_IV.py                          # Black-Scholes 隐含波动率计算
├── formatData.py                     # 数据格式化工具
├── training_NN.ipynb                 # 神经网络训练实验
├── Data/                             # 数据目录
│   ├── 250901.json                   # 期权数据（JSON格式）
│   └── 20250901.csv                  # 原始期权数据
├── results/                          # 结果输出目录
│   ├── VSS_calibration_result.json   # 标准模型校准结果
│   ├── VSS_NN_calibration_result.json # 神经网络模型校准结果
│   ├── Laplace_calibration_result.json # 指数核模型校准结果
│   └── pretrain_network.pth          # 预训练神经网络权重
├── KAN (experiment featrue)/         # KAN网络实验特性
│   ├── KANMonotonicNetwork.py        # KAN单调网络实现
│   └── kan_model.pth                 # KAN模型权重
└── old/                              # 旧版本实现
    ├── VSS_COS_torch.py              # 旧版标准实现
    ├── VSS_COS_exp_ker_torch.py      # 旧版指数核实现
    └── VSS_NN_torch.py               # 旧版神经网络实现
```

## 快速开始

### 1. 数据准备

首先准备期权数据：

```python
from formatData import formatData
# 数据会自动处理并保存为 Data/250901.json
```

### 2. 标准模型定价

```python
from VSSPricerCOSTorch import VSSPricerCOSTorch
from VSSParamTorch import VSSParamTorch
import torch

# 初始化定价器
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pricer = VSSPricerCOSTorch(device=device)

# 设置模型参数
params = VSSParamTorch(
    kappa=-8.9e-5,
    nu=0.176,
    rho=-0.704,
    theta=-0.044,
    X_0=0.113,
    H=0.279,
    device=device
)

# 计算看涨期权价格
S0 = torch.tensor(25617.42, device=device)  # 标的资产价格
K = torch.tensor([25000, 26000], device=device)  # 行权价
r = torch.tensor(0.03, device=device)  # 无风险利率
tau = torch.tensor(0.5, device=device)  # 到期时间

call_prices = pricer.call_price(S0, K, r, tau)
print(f"看涨期权价格: {call_prices}")
```

### 3. 模型校准

```python
import json

# 加载期权数据
with open("Data/250901.json", 'r', encoding='utf-8') as f:
    option_dict = json.load(f)

# 执行校准
result = pricer.calibrate(option_dict, lr=0.01, epochs=1000)

if result["suc"]:
    print("校准成功!")
    print(f"最终损失: {result['loss']:.6f}")
    print(f"校准参数: {result['param']}")
```

### 4. 神经网络模型

```python
from VSSPricerCOSNNTorch import VSSPricerCOSNNTorch
from VSSParamsNNTorch import VSSParamNNTorch

# 初始化神经网络定价器
nn_pricer = VSSPricerCOSNNTorch(VSSParamNNTorch(), device=device)

# 校准神经网络模型
nn_result = nn_pricer.calibrate(option_dict, monotonic_weight=0.1)
```

## 模型参数

### 标准模型参数 ([`VSSParamTorch`](VSSParamTorch.py))

- `kappa`: 均值回归速度
- `nu`: 波动率波动率
- `rho`: 资产价格与波动率的相关系数 (-1 < rho < 1)
- `theta`: 长期均值水平
- `X_0`: 初始波动率
- `H`: Hurst 指数 (0 < H < 1)

### 指数核模型参数 ([`VSSParamLaplaceTorch`](VSSParamLaplaceTorch.py))

- `c`: 指数核系数向量
- `gamma`: 指数核衰减率向量

### 神经网络模型参数 ([`VSSParamsNNTorch`](VSSParamsNNTorch.py))

- 包含标准参数和神经网络权重
- 支持单调性约束

## 配置选项

### COS 方法参数

- `N`: COS 级数项数 (默认: 256)
- `L`: 积分区间扩展因子 (默认: 10.0)
- `n`: 时间离散化点数 (默认: 252)

### 优化参数

- `lr`: 学习率 (默认: 0.01)
- `tol`: 收敛容忍度 (默认: 1e-3)
- `epochs`: 最大迭代次数 (默认: 1000)

## 数值稳定性特性

1. **旋转计数算法**: 处理特征函数中的相位缠绕问题
2. **高精度复数运算**: 使用 `torch.complex128` 确保精度
3. **参数约束**: 自动验证参数范围约束
4. **零频项处理**: `phi_levy[0] *= 0.5` 特殊缩放

## 贡献指南

### 开发环境设置

1. Fork 本仓库
2. 创建特性分支: `git checkout -b feature/新特性`
3. 提交更改: `git commit -am '添加新特性'`
4. 推送分支: `git push origin feature/新特性`
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 代码风格
- 为所有函数和类添加文档字符串
- 包含单元测试
- 更新相关文档

### 测试

运行现有测试用例：

```bash
python -m pytest tests/
```

## 参考文献

1. Abi Jaber, E. (2022). The characteristic function of Gaussian stochastic volatility models: An analytic expression. *Finance and Stochastics*, 26(4), 733–769. https://doi.org/10.1007/s00780-022-00489-4

2. Fang, F., & Oosterlee, C. W. (2009). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. *SIAM Journal on Scientific Computing*, 31(2), 826–848. https://doi.org/10.1137/080718061

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建 GitHub Issue
- 发送邮件至项目维护者
