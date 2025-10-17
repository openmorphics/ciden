import torch
import pytest
from sr_ciden.dynamics import LinearNonlinearODE, IntensityHead, JumpMLP
from sr_ciden.losses import nll_continuous
from sr_ciden.readout import sample_ogata, sample_bernoulli
from sr_ciden.adapters.torch import CIDENContinuous


def test_dynamics_shapes():
    D = 6
    B = 3
    X = 2

    ode = LinearNonlinearODE(dim=D, x_dim=X, nonlinearity="tanh", use_stable_A=True)
    h = torch.randn(B, D)
    t = torch.tensor(0.25)
    dh = ode(h, t)
    assert dh.shape == (B, D)

    head = IntensityHead(state_dim=D, activation="softplus")
    lam = head(h)
    assert lam.shape == (B, 1)
    assert torch.all(lam >= 0)

    jump = JumpMLP(state_dim=D, input_dim=X, hidden_dim=16, nonlinearity="tanh")
    x_i = torch.randn(B, X)
    delta = jump(x_i, h)
    assert delta.shape == (B, D)


def test_nll_continuous_shapes():
    torch.manual_seed(0)
    D = 4
    K = 2

    def f_h(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.1 * h

    def lambda_fn(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Constant nonnegative intensities [K]
        return torch.full((K,), 0.5, device=h.device, dtype=h.dtype)

    h0 = torch.zeros(D)
    times = torch.tensor([0.2, 0.5, 0.8], dtype=h0.dtype)
    marks = torch.tensor([0, 1, 0], dtype=torch.long)

    loss = nll_continuous(
        f_h=f_h,
        lambda_fn=lambda_fn,
        h0=h0,
        events=(times, marks),
        T_end=1.0,
        K=K,
        reduction="mean",
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_readout_shapes():
    K = 3

    def intensity_fn(t: float) -> torch.Tensor:
        # Constant intensities on CPU
        return torch.tensor([0.5, 0.25, 0.75])

    T = 0.5
    lam_max = 2.0
    seed = 123

    spikes_og = sample_ogata(intensity_fn, T=T, lam_max=lam_max, seed=seed)
    assert isinstance(spikes_og, list)
    for item in spikes_og:
        assert isinstance(item, tuple) and len(item) == 2
        t_val, m_val = item
        assert isinstance(t_val, float)
        assert isinstance(m_val, int)
        assert 0.0 < t_val < T
        assert 0 <= m_val < K

    spikes_dt = sample_bernoulli(intensity_fn, T=T, dt=0.05, seed=seed)
    assert isinstance(spikes_dt, list)
    for item in spikes_dt:
        assert isinstance(item, tuple) and len(item) == 2
        t_val, m_val = item
        assert isinstance(t_val, float)
        assert isinstance(m_val, int)
        assert 0.0 <= t_val < T
        assert 0 <= m_val < K


def test_adapter_shapes():
    D = 5
    K = 2
    B = 2
    model = CIDENContinuous(hidden_dim=D, num_marks=K, feature_dim=0)

    # Monkeypatch ode_core.forward to accept 1D h produced by the loss integrator
    import types
    orig_forward = model.ode_core.forward
    def _patched_forward(self, h, t):
        if h.ndim == 1:
            return orig_forward(h.unsqueeze(0), t).squeeze(0)
        return orig_forward(h, t)
    model.ode_core.forward = types.MethodType(_patched_forward, model.ode_core)

    # Two sequences with a couple of events each
    times_b = torch.tensor([0.2, 0.9], dtype=next(model.parameters()).dtype)
    marks_b = torch.tensor([0, 1], dtype=torch.long)
    events = [(times_b, marks_b) for _ in range(B)]

    loss = model(events, T_end=1.0, reduction="mean")
    assert loss.ndim == 0
    assert torch.isfinite(loss)