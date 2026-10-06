"""Command line: rcfuzz audit CONTRACT [--out report.md] [--json out.json]

Exit codes: 0 no exploit found, 1 exploit found, 2 cannot audit.
That makes the audit a one-line CI gate on reward changes.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import adapters, audit, contract as contract_mod, report


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
            spec, _cfg = contract_mod.load_contract(args.contract)
            result = audit.audit_spec(spec)
        except adapters.AuditNotApplicable as exc:
            print(f"cannot audit: {exc}", file=sys.stderr)
            return 2
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"cannot audit: {exc}", file=sys.stderr)
            return 2
        text = report.markdown(result, spec.description)
        if args.out:
            with open(args.out, "w") as fh:
                fh.write(text)
            print(f"wrote {args.out}")
        else:
            print(text)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(report.result_dict(result, spec.description),
                          fh, indent=2)
            print(f"wrote {args.json}")
        return 1 if result.flagged else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
