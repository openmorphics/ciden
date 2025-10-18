import torch
from sr_ciden.adapters import CIDENContinuous
from sr_ciden.solvers import ODESolverConfig


def test_get_intensity_fn_side_effect_free_and_shape():
    torch.set_grad_enabled(True)
    model = CIDENContinuous(hidden_dim=4, num_marks=2, solver_config=ODESolverConfig(dt=1e-3, use_adjoint=False))
    model.eval()
    # Initialize/reset state
    model.reset_state(batch_size=1)
    intensity_fn = model.get_intensity_fn()

    t0 = 0.0
    t1 = 0.1
    lam0 = intensity_fn(t0)
    lam1 = intensity_fn(t1)

    # Shape and nonnegativity
    assert isinstance(lam0, torch.Tensor)
    assert lam0.shape == (2,)
    assert torch.all(lam0 >= 0)
    assert lam1.shape == (2,)
    assert torch.all(lam1 >= 0)

    # Ensure no autograd tracking (side-effect free)
    assert lam0.requires_grad is False
    assert lam1.requires_grad is False
