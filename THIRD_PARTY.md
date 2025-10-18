# Third-Party Notices

This document lists third-party dependencies and their licenses. For an exhaustive, reproducible list:

- Install dev tooling and run:
  - `pip-licenses --format=markdown --with-authors --with-urls --with-license-file --output-file THIRD_PARTY_GENERATED.md`

Key runtime dependencies (see requirements and pyproject):
- PyTorch (BSD-style)
- NumPy (BSD)
- SciPy (BSD)
- Matplotlib (PSF-based)
- tqdm (MPL-2.0)
- torchdiffeq (MIT)

Datasets:
- SHD/SSC/DVS Gesture/NCARS — see dataset providers' licenses. Update [src/sr_ciden/data/fetchers.py](src/sr_ciden/data/fetchers.py:1) with authoritative URLs and checksums, and include their licenses in this file upon bundling or redistribution.

Note:
- This repository is licensed under Apache-2.0 (see LICENSE).
- If vendoring third-party code, update [NOTICE](NOTICE:1) accordingly.
