#!/usr/bin/env python3
"""v0.2 tests: stochastic audits in their three shapes.

1. Expected mode on slippery FrozenLake: the gap is large (0.70)
   but the reward is clean and the optimal policy's rollouts are
   honest navigation, so it must NOT flag.
2. A synthetic stochastic sensor farm (slip corridor, repeatable
   scan) MUST flag.
3. Sampled mode on slippery FrozenLake (black box): the search must
   recover most of the exact optimum and must NOT flag.
4. Contract plumbing: the slippery example loads as a ProbSpec; a
   stochastic env without a closed-loop policy is refused.

Run: python3 tests/test_v02.py  (gymnasium tests skip if absent)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rcfuzz import adapters  # noqa: E402
from rcfuzz.stochastic import ProbSpec, audit_stochastic  # noqa: E402

RESULTS = []

FROZENLAKE_TABLE = {"0": "R", "1": "U", "2": "R", "3": "D", "4": "R",
                    "5": "L", "6": "R", "7": "L", "8": "U", "9": "R",
                    "10": "R", "11": "L", "12": "L", "13": "U",
                    "14": "U", "15": "L"}
NAMES = ["L", "D", "R", "U"]


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def policy_fn(state, action_names):
    return FROZENLAKE_TABLE[str(state)]


def test_stochastic_farm_flags():
    def outcomes(state, action):
        if action == "scan":
            return [(1.0, state, 3.0, False)]
        nxt = min(state + 1, 2)
        return [(0.8, nxt, 10.0 if nxt == 2 else 0.0, nxt == 2),
                (0.2, state, 0.0, False)]

    farm = ProbSpec(
        id="stochastic-sensor-farm", description="synthetic",
        horizon=40, initial_state=0, outcomes_fn=outcomes,
        actions_fn=lambda s: ["E", "scan"],
        is_terminal_fn=lambda s: s == 2,
        intended_policy_fn=lambda s, t: "E")
    res = audit_stochastic(farm, episodes=200)
    check("stochastic farm flags", res.flagged,
          f"gap {res.gap:.1f}, degenerate {res.degenerate}")


def test_gymnasium():
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        print("SKIP gymnasium tests (gymnasium not installed)")
        return
    spec = adapters.gymnasium_spec(
        "FrozenLake-v1", horizon=100, kwargs={"is_slippery": True},
        action_names=NAMES, intended_policy_fn=policy_fn)
    check("slippery table becomes a ProbSpec",
          isinstance(spec, ProbSpec))
    res = audit_stochastic(spec, episodes=400)
    check("slippery FrozenLake gap is real", res.gap > 0.5,
          f"gap {res.gap:.3f}")
    check("slippery FrozenLake does NOT flag (clean reward)",
          not res.flagged,
          f"degenerate {res.degenerate}")

    try:
        adapters.gymnasium_spec(
            "FrozenLake-v1", kwargs={"is_slippery": True},
            action_names=NAMES)
        refused = False
    except adapters.AuditNotApplicable:
        refused = True
    check("stochastic env without a policy baseline is refused",
          refused)

    from rcfuzz import contract as contract_mod
    from rcfuzz.sampled import audit_sampled

    path = os.path.join(os.path.dirname(__file__), "..", "examples",
                        "frozenlake-sampled", "contract.json")
    target, _cfg = contract_mod.load_contract(path)
    check("sampled contract loads",
          isinstance(target, contract_mod.SampledTarget))
    result = audit_sampled(target.simulator, target.intended_fn,
                           search_episodes=200, budget=40_000,
                           verify_episodes=1500)
    check("sampled search recovers most of the optimum",
          result.candidate_mean >= 0.55,
          f"candidate {result.candidate_mean:.3f} (exact 0.744)")
    check("sampled FrozenLake does NOT flag", not result.flagged,
          f"gap {result.gap:.3f}, degenerate {result.degenerate}")


def main():
    test_stochastic_farm_flags()
    test_gymnasium()
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
