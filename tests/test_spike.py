"""Spike tests: corpus integrity, optimiser exactness, and the kill criterion.

Runs under pytest, or standalone: python3 tests/test_spike.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from corpus import build_corpus  # noqa: E402
from fuzzer import fuzz_spec  # noqa: E402
import optimizer  # noqa: E402
from mdp import intended_policy, rollout  # noqa: E402


def test_corpus_shape():
    corpus = build_corpus()
    assert len(corpus) == 12
    assert len({s.id for s in corpus}) == 12
    assert sum(1 for s in corpus if s.label == "exploitable") == 6
    assert sum(1 for s in corpus if s.label == "clean") == 6


def test_replay_verification_exact():
    for spec in build_corpus():
        values, policy, _ = optimizer.solve(spec)
        traj = optimizer.optimal_trajectory(spec, policy)
        assert abs(traj.total_return - optimizer.optimal_value(spec, values)) < 1e-6, spec.id
        # Deterministic replay: same policy, same return, twice.
        a = rollout(spec, intended_policy(spec)).total_return
        b = rollout(spec, intended_policy(spec)).total_return
        assert a == b, spec.id


def test_clean_specs_intended_is_optimal():
    for spec in build_corpus():
        if spec.label != "clean":
            continue
        result = fuzz_spec(spec)
        assert abs(result.diff) < 1e-6, (spec.id, result.diff)
        assert not result.flagged, spec.id


def test_exploitable_specs_beat_intended_by_margin():
    for spec in build_corpus():
        if spec.label != "exploitable":
            continue
        result = fuzz_spec(spec)
        assert result.diff > result.margin_threshold, (spec.id, result.diff)
        assert result.flagged, spec.id


def test_kill_criterion():
    corpus = build_corpus()
    results = {s.id: fuzz_spec(s) for s in corpus}
    planted = [s for s in corpus if s.label == "exploitable"]
    clean = [s for s in corpus if s.label == "clean"]
    hit_rate = sum(1 for s in planted if results[s.id].flagged) / len(planted)
    false_rate = sum(1 for s in clean if results[s.id].flagged) / len(clean)
    assert hit_rate >= 0.50, hit_rate
    assert false_rate <= 0.20, false_rate


def test_fuzzer_is_blind_to_labels():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "..", "src", "fuzzer.py")) as fh:
        source = fh.read()
    assert "spec.label" not in source
    assert "exploit_type" not in source


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
