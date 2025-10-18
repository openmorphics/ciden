import json
import os
import tempfile

import pytest

from sr_ciden.utils.config import load_config, apply_overrides


def test_load_config_json_and_overrides_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cfg.json")
        data = {"seed": 0, "bench": {"runs": 10, "name": "base"}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        cfg = load_config(path)
        assert cfg["seed"] == 0
        assert cfg["bench"]["runs"] == 10

        cfg2 = apply_overrides(cfg, ["seed=123", 'bench.name="modified"', "bench.runs=100"])
        assert cfg2["seed"] == 123
        assert cfg2["bench"]["runs"] == 100
        assert cfg2["bench"]["name"] == "modified"


def test_load_config_yaml_requires_pyyaml():
    pytest.importorskip("yaml")
