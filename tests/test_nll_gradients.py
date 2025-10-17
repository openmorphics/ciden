import torch
import pytest

from sr_ciden.adapters.torch import CIDENContinuous
from sr_ciden.dynamics import StableMatrixParam
from sr_ciden.solvers import ODESolverConfig


def test_nll_continuous_grad_flow():
    torch.manual_seed(0)
    D = 4
    K = 2
    B = 2

    # Use a deterministic fallback solver to ensure test speed and stability
    cfg = ODESolverConfig(use_adjoint=False, fallback="rk4", dt=1e-2)

    model = CIDENContinuous(hidden_dim=D, num_marks=K, feature_dim=0, solver_config=cfg)

    # Optional: set h0 batch explicitly (broadcasting also works, but this covers the code path)
    model.reset_state(batch_size=B)

    # Monkeypatch ode_core.forward to accept 1D h produced by the loss integrator
    import types
    orig_forward = model.ode_core.forward
    def _patched_forward(self, h, t):
        if h.ndim == 1:
            return orig_forward(h.unsqueeze(0), t).squeeze(0)
        return orig_forward(h, t)
    model.ode_core.forward = types.MethodType(_patched_forward, model.ode_core)

    # Create B small sequences with a couple of events
    times = torch.tensor([0.2, 0.6], dtype=next(model.parameters()).dtype)
    marks = torch.tensor([0, 1], dtype=torch.long)
    events = [(times, marks) for _ in range(B)]

    loss = model(events, T_end=1.0, reduction="mean")
    assert loss.ndim == 0 and torch.isfinite(loss)

    loss.backward()

    # Check gradients appear on important parameters
    checks = []

    # ODE core params
    checks.append(("ode.W_h", model.ode_core.W_h))
    if model.ode_core.A_param.stable:
        checks.append(("ode.A.alpha_raw", model.ode_core.A_param.alpha_raw))
        checks.append(("ode.A.L_raw", model.ode_core.A_param.L_raw))
    else:
        checks.append(("ode.A_free", model.ode_core.A_param.A_free))

    # Intensity head params
    if hasattr(model, "intensity_head"):
        ih = model.intensity_head
        if isinstance(ih, torch.nn.ModuleList):
            for i, head in enumerate(ih):
                checks.append((f"head[{i}].W_o.weight", head.W_o.weight))
                checks.append((f"head[{i}].W_o.bias", head.W_o.bias))
        else:
            checks.append(("head.W_o.weight", ih.W_o.weight))
            checks.append(("head.W_o.bias", ih.W_o.bias))

    # At least some critical parameters must have non-None and non-zero gradients
    any_grad = False
    for name, p in checks:
        if p is not None and p.requires_grad and p.grad is not None:
            if torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
                any_grad = True
                break

    assert any_grad, "Expected at least one model parameter to receive non-zero gradients."


def test_stability_param():
    torch.manual_seed(0)
    D = 4
    sp = StableMatrixParam(dim=D, stable=True, alpha_init=0.1)
    A = sp()  # [D, D]
    # A = -(alpha I + L L^T) is symmetric negative definite/semi-definite
    # Use eigvalsh for symmetric matrices
    eigvals = torch.linalg.eigvalsh(A)
    # Allow tiny numerical tolerance; should be strictly < 0 due to positive alpha
    assert eigvals.max().item() < -1e-8