"""Light tests for ``run_pipeline`` (step list + CLI parsing)."""

from __future__ import annotations

import sys

import pytest

import run_pipeline as rp


def test_build_steps_includes_training_by_default():
    steps = rp.build_steps(skip_training=False, with_tests=False)
    flat = " ".join(" ".join(step) for step in steps)
    assert "modelTraining.py" in flat
    assert "package_production_model.py" in flat


def test_build_steps_skip_training():
    steps = rp.build_steps(skip_training=True, with_tests=False)
    flat = " ".join(" ".join(step) for step in steps)
    assert "modelTraining.py" not in flat


def test_build_steps_with_tests_appends_pytest():
    steps = rp.build_steps(skip_training=True, with_tests=True)
    assert steps[-1][:3] == [sys.executable, "-m", "pytest"]


def test_parse_args_dry_run(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--dry-run"])
    ns = rp.parse_args()
    assert ns.dry_run is True
    assert ns.skip_training is False
