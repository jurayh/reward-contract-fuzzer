"""Audit reports: what an env author pastes into an issue or PR."""

from __future__ import annotations

from collections import Counter


def summarise_actions(actions, limit=60):
    if len(actions) <= limit:
        return " ".join(actions) if actions else "(no actions)"
    counts = Counter(actions)
    parts = [f"{a} x{n}" for a, n in counts.most_common()]
    return (", ".join(parts)
            + f"  (in order, first {limit}: "
            + " ".join(actions[:limit]) + " ...)")


def result_dict(result, spec_description=""):
    return {
        "env": result.spec_id,
        "description": spec_description,
        "verdict": "exploit-found" if result.flagged else "no-exploit-found",
        "bounded_claim": (
            "No exploit found under the stated model, horizon, optimizer "
            "and margin. This is not a certificate that the reward is safe."
            if not result.flagged else
            "Exploit verified by exact replay under the stated model and "
            "horizon."
        ),
        "intended_return": result.intended_return,
        "optimal_return": result.optimal_return,
        "pessimal_return": result.pessimal_return,
        "return_range": result.return_range,
        "gap": result.gap,
        "margin_required": result.margin,
        "degenerate": result.degenerate,
        "loop_ratio": result.loop_ratio,
        "top_action_share": result.top_action_share,
        "verified": result.verified,
        "terminated": result.terminated,
        "horizon": result.horizon,
        "optimal_actions": result.optimal_actions,
        "seconds": result.seconds,
    }


def markdown(result, spec_description=""):
    r = result
    if r.flagged:
        verdict = "## Verdict: EXPLOIT FOUND (verified by replay)"
        claim = ("An optimizing policy earns materially more than the "
                 "intended behaviour without doing the intended job. "
                 "The trajectory below is the proof; replay it.")
    else:
        verdict = "## Verdict: no exploit found"
        claim = ("No exploit found under the stated model, horizon, "
                 "optimizer and margin. This is not a certificate that "
                 "the reward is safe.")
    lines = [
        f"# Reward contract audit: {r.spec_id}",
        "",
        spec_description,
        "",
        verdict,
        "",
        claim,
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Intended return | {r.intended_return:.4g} |",
        f"| Optimal return | {r.optimal_return:.4g} |",
        f"| Gap | {r.gap:.4g} |",
        f"| Margin required to flag | {r.margin:.4g} |",
        f"| Return range (optimal - worst) | {r.return_range:.4g} |",
        f"| Degenerate trajectory | {r.degenerate} "
        f"(loop ratio {r.loop_ratio:.2f}, top action share "
        f"{r.top_action_share:.2f}) |",
        f"| Replay verified | {r.verified} |",
        f"| Optimal policy terminated | {r.terminated} |",
        f"| Horizon | {r.horizon} |",
        "",
        "## Optimal policy",
        "",
        summarise_actions(r.optimal_actions),
        "",
        "## How to read this",
        "",
        "The gap is what an optimizer earns over the intended behaviour. "
        "It only counts as an exploit when it clears a scale-aware "
        "margin (the larger of 25% of the intended return and 10% of "
        "the env's return range) and the winning trajectory is "
        "degenerate: looping, farming one action, or refusing to "
        "finish. A merely better honest route does not flag.",
        "",
    ]
    return "\n".join(lines)
