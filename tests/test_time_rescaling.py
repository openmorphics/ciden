import torch
from sr_ciden.validation import time_rescaling_test


def test_time_rescaling_on_known_process():
    # Deterministic RNG
    gen = torch.Generator(device="cpu")
    gen.manual_seed(12345)

    # Known simple process: homogeneous Poisson with constant total intensity
    rate = 30.0  # total rate λ (K=1); choose large to keep total time small
    N = 200      # number of events to simulate; balance stability and speed

    # Generate exponential inter-arrivals Δt ~ Exp(rate), then cumulative sum for event times
    U = torch.rand(N, generator=gen)
    inter = -torch.log(torch.clamp(U, min=1e-12)) / rate
    times = inter.cumsum(0).to(torch.get_default_dtype())
    marks = torch.zeros_like(times, dtype=torch.long)  # marks unused by validation

    # Intensity function λ(t) = [rate] with required signature: t is a scalar 0-d tensor
    def intensity_fn(t: torch.Tensor) -> torch.Tensor:
        return torch.tensor([rate], dtype=t.dtype, device=t.device)

    # Run time-rescaling validation
    out = time_rescaling_test(intensity_fn, (times, marks))
    assert isinstance(out, dict) and "p_value" in out and "D" in out

    # For a well-specified model, K-S p-value should be high (fail to reject H0)
    assert out["p_value"] > 0.1, f"Expected high KS p-value, got {out['p_value']}"