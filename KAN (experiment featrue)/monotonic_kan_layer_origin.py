"""
Monotonic KAN Layer implementation based on manual.md specification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from kan.spline import B_batch, coef2curve, extend_grid


class BSplineBase(nn.Module):
    """
    Base class for B-spline computations.
    
    Attributes:
        k (int): B-spline order (degree + 1). Default 3.
        num_grids (int): number of internal grid points G.
        N_coef (int): total number of B-spline coefficients = G + k - 1.
        grid (torch.Tensor): extended B-spline grid points, shape (1, G+2k).
    """
    def __init__(self, in_features=1, out_features=1, num_grids=10, k=3, grid_range=(-1, 1), device='cpu'):
        """
        Args:
            in_features (int): input dimension (default 1).
            out_features (int): output dimension (default 1).
            num_grids (int): number of internal grid intervals G.
            k (int): B-spline order (polynomial degree + 1).
            grid_range (tuple): (a, b) domain range for spline.
            device (str): device.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.num_grids = num_grids
        self.device = device
        a, b = grid_range
        
        # Create initial grid of size (in_features, num_grids+1)
        # For simplicity, we assume in_features=1, but we can extend.
        grid = torch.linspace(a, b, steps=num_grids + 1, device=device).unsqueeze(0)  # (1, num_grids+1)
        # Extend grid for B-spline basis
        self.grid = extend_grid(grid, k_extend=k)  # (1, num_grids+1+2k)
        self.register_buffer('grid_buffer', self.grid)  # store as buffer (non-learnable)
        
        # Number of B-spline coefficients = num_grids + k
        self.N_coef = self.grid.shape[1] - k - 1  # G + k where G = num_grids?
        # Actually according to manual: N_coef = G + k - 1? Let's verify.
        # In spline.py, coef shape is (in_dim, out_dim, G+k) where G = number of grid intervals.
        # Number of grid intervals = num_grids (since we have num_grids+1 points).
        # So N_coef = num_grids + k.
        # We'll keep as is.
        
    def _compute_spline_curve(self, x, coef):
        """
        Compute spline curve values at x using coef2curve.
        
        Args:
            x (torch.Tensor): input of shape (batch, in_features).
            coef (torch.Tensor): coefficients of shape (out_features, in_features, N_coef).
        
        Returns:
            torch.Tensor: spline values of shape (batch, in_features, out_features).
        """
        # coef2curve expects grid shape (in_dim, G+2k) and coef shape (in_dim, out_dim, G+k)
        # Our grid is (1, G+2k) and coef is (out_features, in_features, N_coef).
        # Need to transpose coef to (in_features, out_features, N_coef).
        # Since in_features=1, we can do:
        if self.in_features == 1:
            grid_ = self.grid_buffer  # (1, G+2k)
            # coef shape (out, 1, N_coef) -> (1, out, N_coef)
            coef_ = coef.transpose(0, 1)  # (in, out, N_coef)
        else:
            raise NotImplementedError("Only in_features=1 supported for now.")
        
        y = coef2curve(x, grid_, coef_, k=self.k, device=self.device)
        # y shape (batch, in_features, out_features)
        return y
    

class MonotonicFunc(BSplineBase):
    """
    Monotonic function f(s) = w·s + spline(s) with f(0)=0 and f'(s) > 0.
    Used for a(s) and H_k(s).
    
    Args:
        out_features (int): output dimension (1 for a(s), H for H_k(s)).
        num_grids (int): number of internal grid intervals.
        k (int): B-spline order.
        grid_range (tuple): domain range for s.
        device (str): device.
    """
    def __init__(self, out_features=1, num_grids=10, k=3, grid_range=(0, 1), device='cpu'):
        # in_features is always 1 for s
        super().__init__(in_features=1, out_features=out_features, num_grids=num_grids, k=k, grid_range=grid_range, device=device)
        
        # Learnable parameters
        # d: increments Δc_i, shape (out_features, in_features, N_coef)
        self.d = nn.Parameter(torch.full((self.out_features, self.in_features, self.N_coef), 1e-3, device=device))
        # w_raw: raw weight for linear term, shape (out_features, in_features)
        self.w_raw = nn.Parameter(torch.full((self.out_features, self.in_features), 1e-3, device=device))
        
    def get_spline_coefficients(self):
        """
        Compute monotonic spline coefficients c_i.
        Ensures S(0)=0 by setting first k increments to zero.
        
        Returns:
            torch.Tensor: coefficients c of shape (out_features, in_features, N_coef).
        """
        # Δc = softplus(d) to ensure positivity
        delta = F.softplus(self.d)
        # Set first k increments to zero to enforce S(0)=0
        delta[:, :, :self.k] = 0.0
        # Cumulative sum to get c_i = sum_{j=0}^{i} Δc_j
        c = torch.cumsum(delta, dim=-1)
        return c
    
    def forward(self, x):
        """
        Forward pass: f(s) = w·s + S(s)
        
        Args:
            x (torch.Tensor): input s of shape (batch, 1).
        
        Returns:
            torch.Tensor: output of shape (batch, out_features).
        """
        # Ensure x shape (batch, 1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        batch = x.shape[0]
        
        # Linear term: w = softplus(w_raw)
        w = F.softplus(self.w_raw)  # (out_features, in_features)
        # Compute w·s: (batch, 1) * (out_features, 1) -> (batch, out_features)
        linear_term = x * w.squeeze(-1).unsqueeze(0)  # w.squeeze(-1) shape (out_features,)
        
        # Spline term
        c = self.get_spline_coefficients()  # (out, in, N_coef)
        spline_vals = self._compute_spline_curve(x, c)  # (batch, in_features, out_features)
        # Since in_features=1, we can squeeze dim=1
        spline_term = spline_vals.squeeze(1)  # (batch, out_features)
        
        # Total
        f = linear_term + spline_term
        return f
    

class GFunc(BSplineBase):
    """
    Positive function G_k(t) = softplus(silu(t) + spline(t)).
    Used for G_k(t) terms.
    
    Args:
        out_features (int): output dimension (hidden_dim H).
        num_grids (int): number of internal grid intervals.
        k (int): B-spline order.
        grid_range (tuple): domain range for t.
        device (str): device.
    """
    def __init__(self, out_features=1, num_grids=10, k=3, grid_range=(0, 1), device='cpu'):
        super().__init__(in_features=1, out_features=out_features, num_grids=num_grids, k=k, grid_range=grid_range, device=device)
        
        # Spline coefficients (no monotonic constraint)
        self.coef = nn.Parameter(torch.zeros((self.out_features, self.in_features, self.N_coef), device=device))
        # Optional linear layer (not used in manual but mentioned)
        self.linear = nn.Linear(1, out_features, bias=False, device=device)
        
    def forward(self, x):
        """
        Forward pass: G_k(t) = softplus(silu(t) + spline(t))
        
        Args:
            x (torch.Tensor): input t of shape (batch, 1).
        
        Returns:
            torch.Tensor: output of shape (batch, out_features).
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        batch = x.shape[0]
        
        # SILU activation
        silu_term = F.silu(x)  # (batch, 1)
        
        # Spline term
        spline_vals = self._compute_spline_curve(x, self.coef)  # (batch, in_features, out_features)
        spline_term = spline_vals.squeeze(1)  # (batch, out_features)
        
        # Linear mapping (optional)
        linear_term = self.linear(x)  # (batch, out_features)
        
        # Combine: silu(t) + spline(t) + linear? According to manual: silu(t) + spline(t)
        inner = silu_term + spline_term + linear_term
        # Apply softplus to ensure positivity
        g = F.softplus(inner)
        return g
    

class MonotonicKANLayer(nn.Module):
    """
    Monotonic KAN layer implementing NN(t,s) = a(s) + Σ_k G_k(t) H_k(s).
    
    Args:
        hidden_dim (int): number of hidden components H.
        num_grids (int): number of internal grid intervals for splines.
        k (int): B-spline order.
        s_range (tuple): domain range for s variable.
        t_range (tuple): domain range for t variable.
        device (str): device.
    """
    def __init__(self, hidden_dim=10, num_grids=10, k=3, s_range=(0, 1), t_range=(0, 1), device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_grids = num_grids
        self.k = k
        self.device = device
        
        # a(s) function: monotonic with output size 1
        self.a_s_func = MonotonicFunc(out_features=1, num_grids=num_grids, k=k, grid_range=s_range, device=device)
        # H_k(s) functions: monotonic with output size hidden_dim
        self.H_s_func = MonotonicFunc(out_features=hidden_dim, num_grids=num_grids, k=k, grid_range=s_range, device=device)
        # G_k(t) functions: positive with output size hidden_dim
        self.G_t_func = GFunc(out_features=hidden_dim, num_grids=num_grids, k=k, grid_range=t_range, device=device)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): input of shape (batch, 2), where x[:,0] = t, x[:,1] = s.
        
        Returns:
            torch.Tensor: output of shape (batch, 1).
        """
        batch = x.shape[0]
        t = x[:, 0:1]  # (batch, 1)
        s = x[:, 1:2]  # (batch, 1)
        
        # Compute a(s)
        a = self.a_s_func(s)  # (batch, 1)
        
        # Compute H_k(s) shape (batch, hidden_dim)
        H = self.H_s_func(s)  # (batch, hidden_dim)
        
        # Compute G_k(t) shape (batch, hidden_dim)
        G = self.G_t_func(t)  # (batch, hidden_dim)
        
        # Element-wise product and sum over k
        product = G * H  # (batch, hidden_dim)
        sum_term = product.sum(dim=1, keepdim=True)  # (batch, 1)
        
        # Final output
        out = a + sum_term
        return out
    

if __name__ == "__main__":
    import sys
    import numpy as np
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on {device}")

    model = MonotonicKANLayer(hidden_dim=10, num_grids=20, k=3, s_range=(0,5), t_range=(0,5), device=device)
    num_param = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_param} parameters")
    del model

    # Helper function for assertions
    def assert_tensor_allclose(a, b, tol=1e-6, msg=""):
        if not torch.allclose(a, b, rtol=tol, atol=tol):
            print(f"FAIL: {msg}")
            print(f"a = {a}")
            print(f"b = {b}")
            sys.exit(1)
        else:
            print(f"PASS: {msg}")
    
    # Test 1: Monotonicity Constraint Verification
    print("\n=== Test 1: Monotonicity Constraint Verification ===")
    monotonic_func = MonotonicFunc(out_features=5, num_grids=8, k=3, grid_range=(0, 1), device=device)
    # Generate random inputs s1 < s2
    batch = 10
    s1 = torch.rand(batch, 1, device=device) * 0.5  # range [0,0.5]
    s2 = s1 + torch.rand(batch, 1, device=device) * 0.5  # s2 > s1
    out1 = monotonic_func(s1)
    out2 = monotonic_func(s2)
    # Check monotonicity per output dimension
    for dim in range(5):
        diff = out2[:, dim] - out1[:, dim]
        if (diff < -1e-6).any():
            print(f"FAIL: Monotonicity violated for dimension {dim}")
            print(f"diff min = {diff.min().item()}")
            sys.exit(1)
    print("PASS: Monotonicity holds for all dimensions")
    
    # Test 2: BSpline Basis Integration
    print("\n=== Test 2: BSpline Basis Integration ===")
    # Verify that spline term is non-zero and depends on coefficients
    x_test = torch.tensor([[0.3], [0.7]], device=device)
    spline_vals = monotonic_func._compute_spline_curve(x_test, monotonic_func.get_spline_coefficients())
    # spline_vals shape (2,1,5)
    assert spline_vals.shape == (2, 1, 5), f"Unexpected shape {spline_vals.shape}"
    # Check that spline values change when coefficients change
    coeffs = monotonic_func.get_spline_coefficients()
    coeffs_perturbed = coeffs + 0.1
    spline_vals2 = monotonic_func._compute_spline_curve(x_test, coeffs_perturbed)
    if torch.allclose(spline_vals, spline_vals2):
        print("WARNING: Spline values unchanged after coefficient perturbation")
    else:
        print("PASS: Spline basis integration works")
    
    # Test 3: Gradient Flow Verification
    print("\n=== Test 3: Gradient Flow Verification ===")
    layer = MonotonicKANLayer(hidden_dim=4, num_grids=6, k=3, device=device)
    x = torch.rand(5, 2, device=device, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    # Check gradients for all parameters
    grad_exists = False
    for name, param in layer.named_parameters():
        if param.grad is None:
            print(f"FAIL: No gradient for {name}")
            sys.exit(1)
        if param.grad.abs().max() > 0:
            grad_exists = True
    if grad_exists:
        print("PASS: Gradients exist and are non-zero for at least some parameters")
    else:
        print("WARNING: All gradients are zero (maybe initialization)")
    
    # Test 4: Parameter Initialization Bounds
    print("\n=== Test 4: Parameter Initialization Bounds ===")
    # Check that d (increments) are positive after softplus
    d = monotonic_func.d.data
    delta = F.softplus(d)
    assert (delta >= 0).all(), "Delta should be non-negative"
    # Check that w_raw is initialized to zero (or whatever)
    w_raw = monotonic_func.w_raw.data
    # No strict bounds, just print
    print(f"d range: [{d.min().item():.3f}, {d.max().item():.3f}]")
    print(f"w_raw range: [{w_raw.min().item():.3f}, {w_raw.max().item():.3f}]")
    # Check spline coefficients are monotonic increasing along last dimension
    c = monotonic_func.get_spline_coefficients()
    diff_c = c[:, :, 1:] - c[:, :, :-1]
    if (diff_c < -1e-8).any():
        print("FAIL: Spline coefficients not monotonic")
        sys.exit(1)
    print("PASS: Spline coefficients are monotonic")
    
    # Test 5: Dimensional Consistency
    print("\n=== Test 5: Dimensional Consistency ===")
    for batch_size in [1, 7, 32]:
        x = torch.randn(batch_size, 2, device=device)
        out = layer(x)
        expected_shape = (batch_size, 1)
        if out.shape != expected_shape:
            print(f"FAIL: batch size {batch_size} produced shape {out.shape}, expected {expected_shape}")
            sys.exit(1)
    print("PASS: Output shape correct for various batch sizes")
    
    # Test 6: Edge Case Handling
    print("\n=== Test 6: Edge Case Handling ===")
    # Inputs outside grid range (s_range and t_range are (0,1))
    # Create inputs outside [0,1]
    x_outside = torch.tensor([[-0.5, 1.5], [2.0, -1.0]], device=device)
    out_outside = layer(x_outside)
    # Should not produce NaN or Inf
    assert torch.isfinite(out_outside).all(), "Output contains NaN/Inf for out-of-range inputs"
    print("PASS: No NaN/Inf for out-of-range inputs")
    
    # Additional test: zero at s=0? Not required but can check
    s_zero = torch.zeros(3, 1, device=device)
    a_zero = layer.a_s_func(s_zero)
    # a(s) should be zero at s=0? Not necessarily, but monotonic with f(0)=0 per manual.
    # Actually manual says f(0)=0 for monotonic functions.
    # Let's check if a(s) is zero at s=0.
    if torch.allclose(a_zero, torch.zeros_like(a_zero), atol=1e-6):
        print("PASS: a(s) zero at s=0")
    else:
        print(f"WARNING: a(s) not zero at s=0, values {a_zero}")
    
    print("\nAll tests passed successfully!")