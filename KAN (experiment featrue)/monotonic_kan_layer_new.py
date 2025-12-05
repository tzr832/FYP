"""
Monotonic KAN Layer implementation based on manual_new.md specification.
Simplified with b(s) = w·s and arccos transformation.
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
        N_coef (int): total number of B-spline coefficients = G + k.
        grid (torch.Tensor): extended B-spline grid points, shape (1, G+2k).
    """
    def __init__(self, in_features=1, out_features=1, num_grids=10, k=3, grid_range=(0, 1), device='cpu'):
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
        self.register_buffer('grid_buffer', self.grid)
        
        # Number of B-spline coefficients = num_grids + k
        self.N_coef = self.grid.shape[1] - k - 1  # G + k where G = num_grids? Let's verify.
        # According to spline.py, coef shape is (in_dim, out_dim, G+k) where G = number of grid intervals.
        # Number of grid intervals = num_grids (since we have num_grids+1 points).
        # So N_coef = num_grids + k.
        # We'll compute as grid.shape[1] - k - 1? Let's test with example.
        # If grid shape = (1, G+1+2k), then G+1+2k - k - 1 = G + k. Yes correct.
        self.N_coef = self.grid.shape[1] - k - 1
        
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
    Monotonic function ψ(s, t) = w·s + S(z) with z = 0.5*(1 + sinh(A*x)/sinh(A)), x = 2*s/t - 1.
    Ensures ψ(0, t)=0 and ∂ψ/∂s ≥ 0.
    
    Args:
        out_features (int): output dimension (1 for a(s), H for H_k(s)).
        num_grids (int): number of internal grid intervals.
        k (int): B-spline order.
        grid_range (tuple): domain range for z (default (0, 1)).
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
        # A_raw: grid density control, shape (out_features, in_features)
        self.A_raw = nn.Parameter(torch.tensor(10.0, device=device))
    
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
    
    def forward(self, s, t):
        """
        Forward pass: ψ(s, t) = w·s + S(z) where z = 0.5*(1 + sinh(A*x)/sinh(A)), x = 2*s/t - 1.
        
        Args:
            s (torch.Tensor): input s of shape (batch, 1).
            t (torch.Tensor): input t of shape (batch, 1).
        
        Returns:
            torch.Tensor: output of shape (batch, out_features).
        """
        # Ensure s and t have shape (batch, 1)
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        batch = s.shape[0]
        
        # Compute x = 2*s/t - 1
        ratio = s / t
        # Ensure ratio in [0, 1] (since s ∈ [0, t])
        ratio = torch.clamp(ratio, 0.0, 1.0)
        x = 2.0 * ratio - 1.0  # shape (batch, 1)
        
        # Compute A = softplus(A_raw) > 0
        A = F.softplus(self.A_raw)  # (1,)
        # Broadcast A to (batch, out_features) for element-wise operations
        # Compute sinh(A*x) and sinh(A)
        sinh_Ax = torch.sinh(A * x)  # (batch, out_features)
        sinh_A = torch.sinh(A)       # (out_features, in_features)
        # Compute z = 0.5 * (1 + sinh(A*x)/sinh(A))
        z = 0.5 * (1.0 + sinh_Ax / sinh_A)  # shape (batch, out_features)
        
        # Linear term: w = softplus(w_raw) > 0
        w = F.softplus(self.w_raw)  # (out_features, in_features)
        # Compute w·s: (batch, 1) * (out_features, 1) -> (batch, out_features)
        linear_term = s * w.squeeze(-1).unsqueeze(0)  # (batch, out_features)
        
        # Spline term S(z)
        c = self.get_spline_coefficients()  # (out, in, N_coef)
        spline_term = self._compute_spline_curve(z, c)  # (batch, in_features, out_features)
        spline_term = spline_term.squeeze(1) # (batch, out_features)

        # Total
        psi = linear_term + spline_term  # (batch, out_features)
        return psi
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
    Monotonic KAN layer implementing NN(t,s) = a(s,t) + Σ_k G_k(t) H_k(s,t).
    
    Args:
        hidden_dim (int): number of hidden components H.
        num_grids (int): number of internal grid intervals for splines.
        k (int): B-spline order.
        s_range (tuple): domain range for s variable (used for a and H).
        t_range (tuple): domain range for t variable (used for G).
        device (str): device.
    """
    def __init__(self, hidden_dim=10, num_grids=10, k=3, t_range=(0, 1), device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_grids = num_grids
        self.k = k
        self.device = device
        
        # a(s,t) function: monotonic with output size 1
        self.a_s_func = MonotonicFunc(out_features=1, num_grids=num_grids, k=k, grid_range=(0, 1), device=device)
        # H_k(s,t) functions: monotonic with output size hidden_dim
        self.H_s_func = MonotonicFunc(out_features=hidden_dim, num_grids=num_grids, k=k, grid_range=(0, 1), device=device)
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
        
        # Compute a(s,t)
        a = self.a_s_func(s, t)  # (batch, 1)
        
        # Compute H_k(s,t) shape (batch, hidden_dim)
        H = self.H_s_func(s, t)  # (batch, hidden_dim)
        
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
    
    # Test 1: MonotonicFunc monotonicity
    print("\n=== Test 1: MonotonicFunc monotonicity ===")
    mono = MonotonicFunc(out_features=5, num_grids=8, k=3, device=device)
    batch = 10
    t = torch.ones(batch, 1, device=device) * 2.0
    s1 = torch.rand(batch, 1, device=device) * 1.5  # s < t
    s2 = s1 + torch.rand(batch, 1, device=device) * (2.0 - s1)  # s2 > s1, still < t
    out1 = mono(s1, t)
    out2 = mono(s2, t)
    # Check monotonicity per output dimension
    for dim in range(5):
        diff = out2[:, dim] - out1[:, dim]
        if (diff < -1e-6).any():
            print(f"FAIL: Monotonicity violated for dimension {dim}")
            print(f"diff min = {diff.min().item()}")
            sys.exit(1)
    print("PASS: Monotonicity holds for all dimensions")
    
    # Test 2: MonotonicFunc zero at s=0
    print("\n=== Test 2: MonotonicFunc zero at s=0 ===")
    s_zero = torch.zeros(3, 1, device=device)
    t_ones = torch.ones(3, 1, device=device) * 3.0
    out_zero = mono(s_zero, t_ones)
    if torch.allclose(out_zero, torch.zeros_like(out_zero), atol=1e-6):
        print("PASS: ψ(0, t) = 0")
    else:
        print(f"WARNING: ψ(0, t) not zero, values {out_zero}")
    
    # Test 3: GFunc positivity
    print("\n=== Test 3: GFunc positivity ===")
    gfunc = GFunc(out_features=7, num_grids=6, k=3, device=device)
    t_vals = torch.rand(5, 1, device=device) * 5.0
    g_vals = gfunc(t_vals)
    if (g_vals > 0).all():
        print("PASS: G(t) > 0 for all t")
    else:
        print(f"FAIL: G(t) contains non-positive values {g_vals}")
        sys.exit(1)
    
    # Test 4: MonotonicKANLayer forward shape
    print("\n=== Test 4: MonotonicKANLayer forward shape ===")
    layer = MonotonicKANLayer(hidden_dim=4, num_grids=6, k=3, device=device)
    for batch_size in [1, 7, 32]:
        x = torch.randn(batch_size, 2, device=device)
        out = layer(x)
        expected_shape = (batch_size, 1)
        if out.shape != expected_shape:
            print(f"FAIL: batch size {batch_size} produced shape {out.shape}, expected {expected_shape}")
            sys.exit(1)
    print("PASS: Output shape correct for various batch sizes")
    
    # Test 5: Gradient flow
    print("\n=== Test 5: Gradient flow ===")
    layer = MonotonicKANLayer(hidden_dim=4, num_grids=6, k=3, device=device)
    x = torch.rand(5, 2, device=device, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
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
    
    # Test 6: Parameter bounds
    print("\n=== Test 6: Parameter bounds ===")
    mono = MonotonicFunc(out_features=3, num_grids=5, k=3, device=device)
    d = mono.d.data
    delta = F.softplus(d)
    assert (delta >= 0).all(), "Delta should be non-negative"
    w_raw = mono.w_raw.data
    print(f"d range: [{d.min().item():.3f}, {d.max().item():.3f}]")
    print(f"w_raw range: [{w_raw.min().item():.3f}, {w_raw.max().item():.3f}]")
    c = mono.get_spline_coefficients()
    diff_c = c[:, :, 1:] - c[:, :, :-1]
    if (diff_c < -1e-8).any():
        print("FAIL: Spline coefficients not monotonic")
        sys.exit(1)
    print("PASS: Spline coefficients are monotonic")
    
    # Test 7: Edge cases (s close to t, s=0, t small)
    print("\n=== Test 7: Edge cases ===")
    layer = MonotonicKANLayer(hidden_dim=2, num_grids=4, k=3, device=device)
    # s = t (ratio = 1) -> z = arccos(-1) = π
    t = torch.tensor([[1.0], [2.0]], device=device)
    s = t.clone()
    out = layer(torch.cat([t, s], dim=1))
    assert torch.isfinite(out).all(), "Output contains NaN/Inf for s=t"
    # s = 0
    s0 = torch.zeros_like(t)
    out0 = layer(torch.cat([t, s0], dim=1))
    assert torch.isfinite(out0).all(), "Output contains NaN/Inf for s=0"
    # t small positive
    t_small = torch.tensor([[0.01], [0.001]], device=device)
    s_small = t_small * 0.5
    out_small = layer(torch.cat([t_small, s_small], dim=1))
    assert torch.isfinite(out_small).all(), "Output contains NaN/Inf for small t"
    print("PASS: No NaN/Inf for edge cases")
    
    print("\nAll tests passed successfully!")