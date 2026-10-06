#!/usr/bin/env python3
"""Reward Contract Fuzzer spike runner.

Runs the two-stage discovery loop over the seeded corpus and reports
hit rate on planted exploits, false-alarm rate on clean specs, and
cost per run, against the pre-registered kill criterion:

  KILL if hit rate < 50% on planted exploits, or false alarms > 20%
  of clean specs. Otherwise the spike passes and a full build is worth
  discussing.

Usage: python3 run_spike.py [--json results.json]
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from corpus import build_corpus  # noqa: E402
from fuzzer import fuzz_spec  # noqa: E402


def main() -> int:
    json_path = None
    if "--json" in sys.argv:
        json_path = sys.argv[sys.argv.index("--json") + 1]

    corpus = build_corpus()
    label_by_id = {s.id: s.label for s in corpus}

    rows = []
    print(f"{'spec':<20} {'label':<12} {'intended':>9} {'optimal':>9} "
          f"{'diff':>8} {'margin':>7} {'loop':>5} {'topAct':>7} "
          f"{'term':>5} {'flag':>5} {'time':>7}")
    print("-" * 108)
    for spec in corpus:
        result = fuzz_spec(spec)
        rows.append(result)
        print(f"{result.spec_id:<20} {label_by_id[result.spec_id]:<12} "
              f"{result.intended_return:>9.1f} {result.optimal_return:>9.1f} "
              f"{result.diff:>8.1f} {result.margin_threshold:>7.1f} "
              f"{result.loop_ratio:>5.2f} {result.top_action_share:>7.2f} "
              f"{str(result.terminated):>5} {str(result.flagged):>5} "
              f"{result.seconds * 1000:>6.1f}m")

    planted = [r for r in rows if label_by_id[r.spec_id] == "exploitable"]
    clean = [r for r in rows if label_by_id[r.spec_id] == "clean"]
    hits = sum(1 for r in planted if r.flagged)
    misses = len(planted) - hits
    false_alarms = sum(1 for r in clean if r.flagged)
    hit_rate = hits / max(1, len(planted))
    false_rate = false_alarms / max(1, len(clean))
    total_seconds = sum(r.seconds for r in rows)
    all_verified = all(r.verified for r in rows)

    kill = hit_rate < 0.50 or false_rate > 0.20
    print("-" * 108)
    print(f"Planted exploits : {len(planted)}  hits={hits} misses={misses} "
          f"hit rate={hit_rate:.0%}")
    print(f"Clean specs      : {len(clean)}  false alarms={false_alarms} "
          f"rate={false_rate:.0%}")
    print(f"Replay verified  : {all_verified} (DP value == replayed return, exact)")
    print(f"Cost             : {total_seconds:.3f}s total, "
          f"{total_seconds / max(1, len(rows)) * 1000:.1f} ms/spec, $0.00 LLM spend")
    print(f"KILL CRITERION   : {'KILL - stop and report failure' if kill else 'PASS - separation holds'} "
          f"(kill if hit rate < 50% or false alarms > 20%)")

    if json_path:
        payload = {
            "specs": [
                {
                    "id": r.spec_id,
                    "label": label_by_id[r.spec_id],
                    "intended_return": r.intended_return,
                    "optimal_return": r.optimal_return,
                    "diff": r.diff,
                    "margin_threshold": r.margin_threshold,
                    "loop_ratio": r.loop_ratio,
                    "top_action_share": r.top_action_share,
                    "terminated": r.terminated,
                    "flagged": r.flagged,
                    "verified": r.verified,
                    "seconds": r.seconds,
                    "optimal_actions": r.optimal_actions,
                }
                for r in rows
            ],
            "hit_rate": hit_rate,
            "false_alarm_rate": false_rate,
            "total_seconds": total_seconds,
            "kill": kill,
        }
        with open(json_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote {json_path}")
    return 1 if kill else 0


if __name__ == "__main__":
    raise SystemExit(main())
