# Volterra Stein-Stein Model Implementation
English | [中文](./README.md)

This project implements European option pricing under the Volterra Stein-Stein model using the COS Fourier method [(Fang & Oosterlee, 2009)](https://doi.org/10.1137/080718061). The model is based on the analytic expression for the characteristic function of Gaussian stochastic volatility models proposed by [Abi Jaber (2022)](https://doi.org/10.1007/s00780-022-00489-4).

## Project Overview

This project provides three different implementations of the Volterra Stein-Stein model:

1. **Standard Kernel Implementation** ([`VSSPricerCOSTorch.py`](VSSPricerCOSTorch.py)) - Uses standard fractional kernel functions
2. **Exponential Kernel Implementation** ([`VSSPricerCOSLaplaceTorch.py`](VSSPricerCOSLaplaceTorch.py)) - Uses exponential decay kernel functions
3. **Neural Network Implementation** ([`VSSPricerCOSNNTorch.py`](VSSPricerCOSNNTorch.py)) - Uses neural networks to learn kernel functions

All implementations are based on the PyTorch framework, supporting GPU acceleration and automatic differentiation.

## Key Features

- ✅ European call/put option pricing
- ✅ Model parameter calibration
- ✅ Support for multiple kernel function implementations
- ✅ Fast pricing based on COS Fourier method
- ✅ PyTorch automatic differentiation support
- ✅ Numerical stability optimization (rotation count algorithm)
- ✅ Parameter constraint validation (0 < H < 1, -1 < rho < 1)

## Installation and Dependencies

### Required Dependencies

```bash
pip install torch numpy scipy numdifftools pandas exchange-calendars matplotlib
```

### Optional Dependencies

```bash
# For Jupyter notebook experiments
pip install jupyter
```

### System Requirements

- Python 3.12
- PyTorch 2.9.1
- NumPy, SciPy, Pandas
- CUDA-enabled GPU (optional, for accelerated computation)

## Project Structure

```
.
├── README.md                          # Project documentation (Chinese)
├── README_EN.md                       # Project documentation (English)
├── VSSPricerCOSBase.py               # Base class implementation with common functionality
├── VSSPricerCOSTorch.py              # Standard kernel implementation
├── VSSPricerCOSLaplaceTorch.py       # Exponential kernel implementation
├── VSSPricerCOSNNTorch.py            # Neural network implementation
├── VSSParamTorch.py                  # Standard model parameter class
├── VSSParamLaplaceTorch.py           # Exponential kernel parameter class
├── VSSParamsNNTorch.py               # Neural network parameter class
├── hyp2f1_numerical.py               # Numerical hypergeometric function implementation
├── BS_IV.py                          # Black-Scholes implied volatility calculation
├── formatData.py                     # Data formatting utilities
├── training_NN.ipynb                 # Neural network training experiments
├── Data/                             # Data directory
│   ├── 250901.json                   # Option data (JSON format)
│   └── 20250901.csv                  # Raw option data
├── results/                          # Results output directory
│   ├── VSS_calibration_result.json   # Standard model calibration results
│   ├── VSS_NN_calibration_result.json # Neural network model calibration results
│   ├── Laplace_calibration_result.json # Exponential kernel model calibration results
│   └── pretrain_network.pth          # Pre-trained neural network weights
├── KAN (experiment featrue)/         # KAN network experimental features
│   ├── KANMonotonicNetwork.py        # KAN monotonic network implementation
│   └── kan_model.pth                 # KAN model weights
└── old/                              # Legacy implementations
    ├── VSS_COS_torch.py              # Legacy standard implementation
    ├── VSS_COS_exp_ker_torch.py      # Legacy exponential kernel implementation
    └── VSS_NN_torch.py               # Legacy neural network implementation
```

## Quick Start

### 1. Data Preparation

First, prepare the option data:

```python
from formatData import formatData
# Data will be automatically processed and saved as Data/250901.json
```

### 2. Standard Model Pricing

```python
from VSSPricerCOSTorch import VSSPricerCOSTorch
from VSSParamTorch import VSSParamTorch
import torch

# Initialize pricer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pricer = VSSPricerCOSTorch(device=device)

# Set model parameters
params = VSSParamTorch(
    kappa=-8.9e-5,
    nu=0.176,
    rho=-0.704,
    theta=-0.044,
    X_0=0.113,
    H=0.279,
    device=device
)

# Calculate call option prices
S0 = torch.tensor(25617.42, device=device)  # Underlying asset price
K = torch.tensor([25000, 26000], device=device)  # Strike prices
r = torch.tensor(0.03, device=device)  # Risk-free rate
tau = torch.tensor(0.5, device=device)  # Time to maturity

call_prices = pricer.call_price(S0, K, r, tau)
print(f"Call option prices: {call_prices}")
```

### 3. Model Calibration

```python
import json

# Load option data
with open("Data/250901.json", 'r', encoding='utf-8') as f:
    option_dict = json.load(f)

# Perform calibration
result = pricer.calibrate(option_dict, lr=0.01, epochs=1000)

if result["suc"]:
    print("Calibration successful!")
    print(f"Final loss: {result['loss']:.6f}")
    print(f"Calibrated parameters: {result['param']}")
```

### 4. Neural Network Model

```python
from VSSPricerCOSNNTorch import VSSPricerCOSNNTorch
from VSSParamsNNTorch import VSSParamNNTorch

# Initialize neural network pricer
nn_pricer = VSSPricerCOSNNTorch(VSSParamNNTorch(), device=device)

# Calibrate neural network model
nn_result = nn_pricer.calibrate(option_dict, monotonic_weight=0.1)
```

## Model Parameters

### Standard Model Parameters ([`VSSParamTorch`](VSSParamTorch.py))

- `kappa`: Mean reversion speed
- `nu`: Volatility of volatility
- `rho`: Correlation between asset price and volatility (-1 < rho < 1)
- `theta`: Long-term mean level
- `X_0`: Initial volatility
- `H`: Hurst index (0 < H < 1)

### Exponential Kernel Model Parameters ([`VSSParamLaplaceTorch`](VSSParamLaplaceTorch.py))

- `c`: Exponential kernel coefficient vector
- `gamma`: Exponential kernel decay rate vector

### Neural Network Model Parameters ([`VSSParamsNNTorch`](VSSParamsNNTorch.py))

- Includes standard parameters and neural network weights
- Supports monotonicity constraints

## Configuration Options

### COS Method Parameters

- `N`: Number of COS series terms (default: 256)
- `L`: Integration interval expansion factor (default: 10.0)
- `n`: Number of time discretization points (default: 252)

### Optimization Parameters

- `lr`: Learning rate (default: 0.01)
- `tol`: Convergence tolerance (default: 1e-3)
- `epochs`: Maximum number of iterations (default: 1000)

## Numerical Stability Features

1. **Rotation Count Algorithm**: Handles phase wrapping issues in characteristic function
2. **High-Precision Complex Arithmetic**: Uses `torch.complex128` for precision
3. **Parameter Constraints**: Automatic validation of parameter range constraints
4. **Zero-Frequency Handling**: Special scaling `phi_levy[0] *= 0.5` for k=0 term

## Contribution Guidelines

### Development Environment Setup

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push the branch: `git push origin feature/new-feature`
5. Create a Pull Request

### Code Standards

- Follow PEP 8 code style
- Add docstrings for all functions and classes
- Include unit tests
- Update relevant documentation

### Testing

Run existing test cases:

```bash
python -m pytest tests/
```

## References

1. Abi Jaber, E. (2022). The characteristic function of Gaussian stochastic volatility models: An analytic expression. *Finance and Stochastics*, 26(4), 733–769. https://doi.org/10.1007/s00780-022-00489-4

2. Fang, F., & Oosterlee, C. W. (2009). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. *SIAM Journal on Scientific Computing*, 31(2), 826–848. https://doi.org/10.1137/080718061

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or suggestions, please contact:
- Create a GitHub Issue
- Email the project maintainer

## Acknowledgments

Thanks to all developers and researchers who have contributed to this project.