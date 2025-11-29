"""
Volterra Stein-Stein model parameters with exponential kernel for PyTorch
"""

import torch
from typing import Dict


class VSSParamLaplaceTorch:
    """
    Volterra Stein-Stein model parameters with PyTorch tensors for exponential kernel
    """
    def __init__(self, 
                 kappa: float = -8.9e-5,
                 rho: float = -0.704,
                 theta: float = -0.044,
                 X_0: float = 0.113,
                 terms: int = 10,
                 device: str = 'cpu'):
        
        self.device = device
        
        # Convert parameters to tensors with requires_grad=True for gradient computation
        self.kappa = torch.tensor(kappa, dtype=torch.float64, device=device, requires_grad=True)
        self.nu = torch.tensor(1, dtype=torch.float64, device=device)
        self.rho = torch.tensor(rho, dtype=torch.float64, device=device, requires_grad=True)
        self.theta = torch.tensor(theta, dtype=torch.float64, device=device, requires_grad=True)
        self.X_0 = torch.tensor(X_0, dtype=torch.float64, device=device, requires_grad=True)
        
        self.c = torch.rand(terms, dtype=torch.float64, device=device, requires_grad=True)
        self.gamma = torch.rand(terms, dtype=torch.float64, device=device, requires_grad=True)
        with torch.no_grad():
            self.gamma.mul_(10.0) # Scale gamma to a reasonable range
            self.c.mul_(0.1)
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate parameter constraints"""
        if not (-1 < self.rho < 1):
            raise ValueError("rho must be in the interval (-1, 1).")
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Return parameters as dictionary"""
        return {
            'kappa': self.kappa.item(),
            'nu': self.nu.item(),
            'rho': self.rho.item(),
            'theta': self.theta.item(),
            'X_0': self.X_0.item(),
            'c': self.c.tolist(),
            'gamma': self.gamma.tolist()
        }
    
    def set_requires_grad(self, requires_grad: bool = True):
        """Set requires_grad for all parameters"""
        for param_name, param in [('kappa', self.kappa), ('nu', self.nu), ('rho', self.rho),
                                 ('theta', self.theta), ('X_0', self.X_0), ('c', self.c), ('gamma', self.gamma)]:
            try:
                param.requires_grad = requires_grad
            except RuntimeError as e:
                print(f"ERROR: Failed to set requires_grad for {param_name}: {e}")
                raise