#!/usr/bin/env python3
"""v0.1 tests: calibrated rule on the corpora plus both adapters.

Runs standalone (python3 tests/test_v01.py) or under pytest.
Gymnasium tests skip cleanly when gymnasium is not installed.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))

from rcfuzz import adapters, audit_spec  # noqa: E402
from rcfuzz.mdp import Spec  # noqa: E402


def _specs(module_name, builder):
    module = __import__(module_name)
    return getattr(module, builder)()


def test_spike_corpus_regression():
    from corpus import build_corpus
    for spec in build_corpus():
        # The package Spec is a different class than the spike Spec;
        # rebuild through the shared protocol fields.
        pkg_spec = Spec(
            id=spec.id, description=spec.description, horizon=spec.horizon,
            initial_state=spec.initial_state, step_fn=spec.step_fn,
            actions_fn=spec.actions_fn, is_terminal_fn=spec.is_terminal_fn,
            intended_actions=spec.intended_actions)
        result = audit_spec(pkg_spec)
        assert result.verified, spec.id
        assert result.flagged == (spec.label == "exploitable"), (
            spec.id, result.flagged, result.gap, result.margin)


def test_borderline_corpus():
    from borderline import borderline_corpus
    for spec in borderline_corpus():
        pkg_spec = Spec(
            id=spec.id, description=spec.description, horizon=spec.horizon,
            initial_state=spec.initial_state, step_fn=spec.step_fn,
            actions_fn=spec.actions_fn, is_terminal_fn=spec.is_terminal_fn,
            intended_actions=spec.intended_actions)
        result = audit_spec(pkg_spec)
        assert result.verified, spec.id
        assert result.flagged == (spec.label == "exploitable"), (
            spec.id, result.flagged, result.gap, result.margin)


def test_stateful_adapter_catches_sensor_farm():
    path = os.path.join(ROOT, "examples", "sensor-farm", "env.py")
    spec = adapters.stateful_spec(path)
    result = audit_spec(spec)
    assert result.verified
    assert result.flagged
    assert result.intended_return == 8.0
    assert result.optimal_return == 194.0


def test_gymnasium_adapter_frozenlake_clean():
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        return  # skip when the extra is not installed
    spec = adapters.gymnasium_spec(
        "FrozenLake-v1", intended_actions=["D", "D", "R", "R", "D", "R"],
        horizon=16, kwargs={"is_slippery": False},
        action_names=["L", "D", "R", "U"])
    result = audit_spec(spec)
    assert result.verified
    assert not result.flagged
    assert result.gap == 0.0


def test_gymnasium_adapter_rejects_stochastic():
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        return
    try:
        adapters.gymnasium_spec(
            "FrozenLake-v1", intended_actions=[], horizon=16,
            kwargs={"is_slippery": True},
            action_names=["L", "D", "R", "U"])
    except adapters.AuditNotApplicable:
        return
    raise AssertionError("stochastic env was not rejected")


TESTS = [
    test_spike_corpus_regression,
    test_borderline_corpus,
    test_stateful_adapter_catches_sensor_farm,
    test_gymnasium_adapter_frozenlake_clean,
    test_gymnasium_adapter_rejects_stochastic,
]


if __name__ == "__main__":
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)
