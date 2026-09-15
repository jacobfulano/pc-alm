from __future__ import annotations

import json
import os
from dataclasses import replace

from pcalm.config import ExperimentConfig
from pcalm.training import train_one


def test_train_one_synthetic_writes_compatible_outputs(tmp_path):
    base = ExperimentConfig()
    config = replace(
        base,
        output_dir=str(tmp_path),
        model=replace(base.model, width=4, depth=3, input_dim=8, output_dim=2),
        method=replace(base.method, budget=2),
        training=replace(base.training, batch_size=4, train_subset=8, test_subset=4),
    )
    summary = train_one(config, device=os.environ.get("PCALM_TEST_DEVICE", "cpu"))

    assert summary["method"] == "pcalm"
    assert summary["steps"] == 2
    assert {path.name for path in tmp_path.iterdir()} == {
        "config.json",
        "metrics.csv",
        "summary.json",
    }
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
