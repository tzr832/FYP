"""
Volterra Stein-Stein model parameters with PyTorch tensors
"""

import torch
from typing import Dict


class VSSParamTorch:
    """
    Volterra Stein-Stein model parameters with PyTorch tensors
    """
    def __init__(self, 
                 kappa: float = -8.9e-5,
                 nu: float = 0.176,
                 rho: float = -0.704,
                 theta: float = -0.044,
                 X_0: float = 0.113,
                 H: float = 0.279,
                 device: str = 'cpu'):
        
        self.device = device
        
        # Convert parameters to tensors with requires_grad=True for gradient computation
        self.kappa = torch.tensor(kappa, dtype=torch.float64, device=device, requires_grad=True)
        self.nu = torch.tensor(nu, dtype=torch.float64, device=device, requires_grad=True)
        self.rho = torch.tensor(rho, dtype=torch.float64, device=device, requires_grad=True)
        self.theta = torch.tensor(theta, dtype=torch.float64, device=device, requires_grad=True)
        self.X_0 = torch.tensor(X_0, dtype=torch.float64, device=device, requires_grad=True)
        self.H = torch.tensor(H, dtype=torch.float64, device=device, requires_grad=True)
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate parameter constraints"""
        if not (0 < self.H < 1):
            raise ValueError("Hurst index H must be in the interval (0, 1).")
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
            'H': self.H.item()
        }
    
    def set_requires_grad(self, requires_grad: bool = True):
        """Set requires_grad for all parameters"""
        for param in [self.kappa, self.nu, self.rho, self.theta, self.X_0, self.H]:
            param.requires_grad = requires_grad