import torch
from torch.autograd import Function
import numpy as np
from scipy.special import hyp2f1 as hyp2f1_scipy

class Hyp2F1(Function):
    # ---- hyperparameters for finite difference ----
    h = 1e-6    # central diff step

    @staticmethod
    def forward(ctx, a, b, c, z):
        # convert to numpy float
        a_np = a.detach().cpu().numpy().astype(float)
        b_np = b.detach().cpu().numpy().astype(float)
        c_np = c.detach().cpu().numpy().astype(float)
        z_np = z.detach().cpu().numpy().astype(float)

        # forward value using scipy
        out_np = hyp2f1_scipy(a_np, b_np, c_np, z_np)
        out = torch.tensor(out_np, dtype=z.dtype, device=z.device)

        # save tensors for backward
        ctx.save_for_backward(a, b, c, z, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, c, z, out = ctx.saved_tensors

        # convert to numpy
        a_np = a.detach().cpu().numpy().astype(float)
        b_np = b.detach().cpu().numpy().astype(float)
        c_np = c.detach().cpu().numpy().astype(float)
        z_np = z.detach().cpu().numpy().astype(float)
        h = Hyp2F1.h

        # ======== ∂F/∂z (analytic) ========
        # d/dz F(a,b;c;z) = ab/c * F(a+1, b+1; c+1; z)
        dF_dz_np = (a_np * b_np / c_np) * hyp2f1_scipy(a_np + 1, b_np + 1, c_np + 1, z_np)
        dF_dz = torch.tensor(dF_dz_np, dtype=z.dtype, device=z.device)

        # ======== central diff for a,b,c ========
        # ∂F/∂a
        Fa_plus = hyp2f1_scipy(a_np + h, b_np, c_np, z_np)
        Fa_minus = hyp2f1_scipy(a_np - h, b_np, c_np, z_np)
        dF_da_np = (Fa_plus - Fa_minus) / (2 * h)
        dF_da = torch.tensor(dF_da_np, dtype=z.dtype, device=z.device)

        # ∂F/∂b
        Fb_plus = hyp2f1_scipy(a_np, b_np + h, c_np, z_np)
        Fb_minus = hyp2f1_scipy(a_np, b_np - h, c_np, z_np)
        dF_db_np = (Fb_plus - Fb_minus) / (2 * h)
        dF_db = torch.tensor(dF_db_np, dtype=z.dtype, device=z.device)

        # ∂F/∂c
        Fc_plus = hyp2f1_scipy(a_np, b_np, c_np + h, z_np)
        Fc_minus = hyp2f1_scipy(a_np, b_np, c_np - h, z_np)
        dF_dc_np = (Fc_plus - Fc_minus) / (2 * h)
        dF_dc = torch.tensor(dF_dc_np, dtype=z.dtype, device=z.device)

        # chain rule: grad_output * ∂out/∂param
        grad_a = grad_output * dF_da
        grad_b = grad_output * dF_db
        grad_c = grad_output * dF_dc
        grad_z = grad_output * dF_dz

        return grad_a, grad_b, grad_c, grad_z


# ======== helper function ========
def hyp2f1(a, b, c, z) -> torch.Tensor:
    return Hyp2F1.apply(a, b, c, z)


if __name__ == "__main__":

    a = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    z = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

    y = hyp2f1(a, b, c, z)
    y.sum().backward()

    print("value:", y.item())
    print("grads:", a.grad.item(), b.grad.item(), c.grad.item(), z.grad.item())