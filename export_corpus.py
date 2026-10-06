#!/usr/bin/env python3
"""Export the seed corpus as declarative JSON files.

corpus/<id>.json holds the blind spec (world + reward terms + intended
script) -- exactly what the fuzzer consumes in spirit. Ground-truth labels
live only in corpus/ground_truth.json, read by evaluation, never by
discovery.

Usage: python3 export_corpus.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "src")

from corpus import build_corpus  # noqa: E402


def main():
    os.makedirs("corpus", exist_ok=True)
    ground_truth = {}
    for spec in build_corpus():
        decl = getattr(spec, "declaration", {})
        doc = {
            "id": spec.id,
            "description": spec.description,
            "horizon": spec.horizon,
            "world": decl.get("world", ""),
            "reward_terms": decl.get("reward_terms", {}),
            "intended_actions": spec.intended_actions,
        }
        with open(f"corpus/{spec.id}.json", "w") as fh:
            json.dump(doc, fh, indent=2)
        ground_truth[spec.id] = {
            "label": spec.label,
            "exploit_type": spec.exploit_type,
            "notes": spec.notes,
        }
    with open("corpus/ground_truth.json", "w") as fh:
        json.dump(ground_truth, fh, indent=2)
    print(f"Exported {len(ground_truth)} specs to corpus/ (+ ground_truth.json)")


if __name__ == "__main__":
    main()
