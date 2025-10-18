from __future__ import annotations

import hashlib
import os
import pathlib
import urllib.request
from dataclasses import dataclass
from typing import Optional

from sr_ciden.utils.determinism import get_data_root

__all__ = [
    "ensure_file",
    "fetch_shd",
    "fetch_ssc",
    "fetch_dvs_gesture",
    "fetch_ncars",
]


@dataclass
class DatasetSpec:
    url: str
    sha256: str
    filename: str
    subdir: str


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_file(url: str, sha256: str, filename: str, subdir: str) -> str:
    """
    Download a file to SR_CIDEN_DATA/datasets/{subdir}/{filename} and verify SHA-256.
    Returns the absolute path to the file.
    """
    root = os.path.join(get_data_root(), "datasets", subdir)
    os.makedirs(root, exist_ok=True)
    dest = os.path.join(root, filename)
    if not os.path.exists(dest):
        tmp = dest + ".tmp"
        urllib.request.urlretrieve(url, tmp)  # nosec - controlled URL provided by dataset spec
        os.replace(tmp, dest)
    # Verify checksum
    digest = _sha256(dest)
    if digest.lower() != sha256.lower():
        raise ValueError(f"Checksum mismatch for {dest}: expected {sha256}, got {digest}")
    return os.path.abspath(dest)


# Placeholder specs — replace with authoritative URLs and checksums for camera-ready
_SPECS = {
    "shd": DatasetSpec(
        url="https://example.com/placeholder/shd.zip",
        sha256="REPLACE_WITH_REAL_SHA256",
        filename="shd.zip",
        subdir="shd",
    ),
    "ssc": DatasetSpec(
        url="https://example.com/placeholder/ssc.zip",
        sha256="REPLACE_WITH_REAL_SHA256",
        filename="ssc.zip",
        subdir="ssc",
    ),
    "dvs_gesture": DatasetSpec(
        url="https://example.com/placeholder/dvs_gesture.zip",
        sha256="REPLACE_WITH_REAL_SHA256",
        filename="dvs_gesture.zip",
        subdir="dvs_gesture",
    ),
    "ncars": DatasetSpec(
        url="https://example.com/placeholder/ncars.zip",
        sha256="REPLACE_WITH_REAL_SHA256",
        filename="ncars.zip",
        subdir="ncars",
    ),
}


def _fetch_from_spec(name: str) -> str:
    spec = _SPECS[name]
    if "REPLACE_WITH_REAL_SHA256" in spec.sha256 or "example.com" in spec.url:
        raise NotImplementedError(
            f"{name} dataset spec placeholder — update URL and sha256. "
            "See THIRD_PARTY.md for licensing and source."
        )
    return ensure_file(spec.url, spec.sha256, spec.filename, spec.subdir)


def fetch_shd() -> str:
    """Return local path to SHD archive (download if needed)."""
    return _fetch_from_spec("shd")


def fetch_ssc() -> str:
    """Return local path to SSC archive (download if needed)."""
    return _fetch_from_spec("ssc")


def fetch_dvs_gesture() -> str:
    """Return local path to DVS Gesture archive (download if needed)."""
    return _fetch_from_spec("dvs_gesture")


def fetch_ncars() -> str:
    """Return local path to NCARS archive (download if needed)."""
    return _fetch_from_spec("ncars")
