import torch
from sr_ciden.readout import sample_ogata_with_stats


def test_acceptance_increases_with_tighter_bound():
    def lam_fn(t: float):
        # Two-mark constant intensities: total 0.7
        return torch.tensor([0.5, 0.2], dtype=torch.get_default_dtype())

    T = 3.0
    true_max = 0.7
    ratios = []
    for multiple in [5.0, 2.0, 1.5, 1.2, 1.0]:
        _, stats = sample_ogata_with_stats(lam_fn, T=T, lam_max=true_max * multiple, seed=999)
        ratios.append(float(stats["acceptance_ratio"]))
    # As lam_max decreases (tighter), acceptance should not decrease overall
    # We check that the sequence is non-decreasing after reversing (coarse check).
    assert all(ratios[i] <= ratios[i + 1] + 1e-6 for i in range(len(ratios) - 1))
