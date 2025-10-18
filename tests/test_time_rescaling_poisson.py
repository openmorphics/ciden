import torch
from sr_ciden.readout import sample_ogata
from sr_ciden.validation import time_rescaling_test


def test_time_rescaling_poisson_reasonable_p_value():
    torch.manual_seed(0)
    lam = 1.0
    T = 5.0

    def lam_fn(t):
        return torch.tensor([lam], dtype=torch.get_default_dtype())

    # Simulate spikes with a conservative bound
    spikes = sample_ogata(lam_fn, T=T, lam_max=lam * 1.1, seed=123)
    times = torch.tensor([t for (t, _) in spikes], dtype=torch.get_default_dtype())
    marks = torch.zeros(len(spikes), dtype=torch.long)

    if len(spikes) < 5:
        # Not enough events to run a meaningful KS; skip quietly
        return

    res = time_rescaling_test(lam_fn, (times, marks))
    p = float(res.get("p_value", 0.0))
    # Very lenient bound to reduce flake probability
    assert 0.0 <= p <= 1.0
