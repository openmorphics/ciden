import torch
from sr_ciden.readout import sample_ogata, sample_ogata_with_stats


def test_thinning_determinism():
    K = 3

    def intensity_fn(t: float) -> torch.Tensor:
        # Constant rates per mark
        return torch.tensor([0.5, 0.25, 0.75])

    T = 1.0
    lam_max = 2.0  # >= sum(intensity)=1.5
    seed = 42

    spikes1 = sample_ogata(intensity_fn, T=T, lam_max=lam_max, seed=seed)
    spikes2 = sample_ogata(intensity_fn, T=T, lam_max=lam_max, seed=seed)
    assert spikes1 == spikes2, "Ogata sampler must be deterministic given a fixed seed"


def test_thinning_acceptance_ratio():
    # Define constant intensity with known total sum
    lam_per = 0.8
    K = 3
    def intensity_fn(t: float) -> torch.Tensor:
        return torch.full((K,), lam_per)

    total = lam_per * K  # expected sum over marks
    T = 3.0
    seed = 123

    # Loose bound: lam_max = 2 * total -> expected acceptance ratio ~ 0.5
    lam_max_loose = 2.0 * total
    _, stats_loose = sample_ogata_with_stats(intensity_fn, T=T, lam_max=lam_max_loose, seed=seed)
    acc_loose = float(stats_loose["acceptance_ratio"])
    # Tight bound: lam_max = total -> expected acceptance ratio ~ 1.0
    lam_max_tight = 1.0 * total
    _, stats_tight = sample_ogata_with_stats(intensity_fn, T=T, lam_max=lam_max_tight, seed=seed)
    acc_tight = float(stats_tight["acceptance_ratio"])

    # Reasonable tolerances given finite proposals
    assert abs(acc_loose - 0.5) <= 0.2, f"Expected ~0.5 acceptance, got {acc_loose}"
    assert abs(acc_tight - 1.0) <= 0.05, f"Expected ~1.0 acceptance, got {acc_tight}"
    # Monotonicity: tighter bound should not have worse acceptance
    assert acc_tight >= acc_loose - 1e-6