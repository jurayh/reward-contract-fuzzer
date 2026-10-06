#!/usr/bin/env python3
"""Build the head-to-head comparison matrix.

Reads ground_truth.json, ratctl_out/*.json, fuzzer_out.json and emits
comparison.json plus a markdown table. A tool "flags" an env when:
  ratctl : gameability_score >= 30 (its own CI gate is gameability>0.3)
           or any finding at high/critical severity
  fuzzer : fuzz_spec flagged (verified margin + degeneracy gates)
Tamper envs have no MDP adapter, so the fuzzer cell is out-of-model.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)


def load(name):
    with open(os.path.join(HERE, name)) as fh:
        return json.load(fh)


def ratctl_flag(report):
    if report["gameability_score"] >= 30:
        return True
    for cs in report["class_scores"].values():
        for f in cs["findings"]:
            if f["severity"] in ("high", "critical"):
                return True
    return False


def main():
    truth = load("ground_truth.json")
    fuzz = load("fuzzer_out.json")
    rows = []
    for name in sorted(truth):
        t = truth[name]
        rat = load(os.path.join("ratctl_out", f"{name}.json"))
        findings = [
            {"class": c, "severity": f["severity"], "title": f["title"]}
            for c, cs in rat["class_scores"].items()
            for f in cs["findings"]
        ]
        row = {
            "env": name,
            "format": t["format"],
            "tamper_vulnerable": t["tamper_vulnerable"],
            "design_exploitable": t["design_exploitable"],
            "mdp_adapter": t["mdp_adapter"],
            "ratctl": {
                "gameability": rat["gameability_score"],
                "findings": findings,
                "flagged": ratctl_flag(rat),
            },
            "fuzzer": None,
        }
        if t["mdp_adapter"]:
            fz = fuzz[name]
            row["fuzzer"] = {
                "intended_return": fz["intended_return"],
                "optimal_return": fz["optimal_return"],
                "diff": fz["diff"],
                "flagged": fz["flagged"],
                "verified": fz["verified"],
                "seconds": fz["seconds"],
            }
        rows.append(row)

    with open(os.path.join(HERE, "comparison.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    print("| Env | Truth | ratctl | Fuzzer |")
    print("|---|---|---|---|")
    for r in rows:
        if r["tamper_vulnerable"]:
            truth_label = "tamper"
        elif r["design_exploitable"]:
            truth_label = "design exploit"
        else:
            truth_label = "clean"
        rat_cell = (f"gameability {r['ratctl']['gameability']}, "
                    f"{len(r['ratctl']['findings'])} findings, "
                    f"{'FLAG' if r['ratctl']['flagged'] else 'pass'}")
        if r["fuzzer"] is None:
            fuzz_cell = "out of model (no MDP)"
        else:
            fz = r["fuzzer"]
            fuzz_cell = (f"intended {fz['intended_return']:.0f} vs optimal "
                         f"{fz['optimal_return']:.0f} "
                         f"({'FLAG' if fz['flagged'] else 'pass'}, "
                         f"verified={fz['verified']})")
        print(f"| {r['env']} | {truth_label} | {rat_cell} | {fuzz_cell} |")
    print("\nWrote comparison.json")


if __name__ == "__main__":
    main()
