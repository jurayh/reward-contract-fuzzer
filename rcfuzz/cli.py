"""Command line: rcfuzz audit CONTRACT [--out report.md] [--json out.json]

Exit codes: 0 no exploit found, 1 exploit found, 2 cannot audit.
That makes the audit a one-line CI gate on reward changes.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import adapters, audit, contract as contract_mod, report
from .mdp import Spec
from .sampled import audit_sampled
from .stochastic import ProbSpec, audit_stochastic


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="rcfuzz",
        description="Find the policy your reward function actually "
                    "incentivizes, before an agent does.")
    sub = parser.add_subparsers(dest="command", required=True)
    audit_p = sub.add_parser("audit", help="audit one reward contract")
    audit_p.add_argument("contract", help="path to a contract JSON file")
    audit_p.add_argument("--out", help="write the markdown report here")
    audit_p.add_argument("--json", help="write the JSON report here")
    args = parser.parse_args(argv)

    if args.command == "audit":
        try:
            target, _cfg = contract_mod.load_contract(args.contract)
            if isinstance(target, Spec):
                result = audit.audit_spec(target)
                text = report.markdown(result, target.description)
                payload = report.result_dict(result, target.description)
            elif isinstance(target, ProbSpec):
                result = audit_stochastic(target)
                text = report.stochastic_markdown(
                    result, target.description)
                payload = report.stochastic_dict(result)
            elif isinstance(target, contract_mod.SampledTarget):
                cfg = target.search_cfg
                result = audit_sampled(
                    target.simulator, target.intended_fn,
                    search_episodes=cfg.get("episodes_per_candidate",
                                             200),
                    budget=cfg.get("budget", 60_000),
                    verify_episodes=cfg.get("verify_episodes", 2000))
                text = report.stochastic_markdown(
                    result, "Sampled audit (black-box simulator)")
                payload = report.stochastic_dict(result)
            else:  # pragma: no cover - defensive
                raise adapters.AuditNotApplicable("unknown audit target")
        except adapters.AuditNotApplicable as exc:
            print(f"cannot audit: {exc}", file=sys.stderr)
            return 2
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"cannot audit: {exc}", file=sys.stderr)
            return 2
        if args.out:
            with open(args.out, "w") as fh:
                fh.write(text)
            print(f"wrote {args.out}")
        else:
            print(text)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"wrote {args.json}")
        return 1 if result.flagged else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
