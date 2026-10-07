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


CAVEAT = (
    "Baseline caveat: in a stochastic env, part of any return gap "
    "can be control skill rather than incentive design; a reactive "
    "policy beats a rigid one even under a clean reward. That is why "
    "stochastic mode never flags on a gap alone: it also requires "
    "degenerate behaviour in the winning policy's own rollouts."
)


def stochastic_dict(result):
    return {
        "env": result.spec_id,
        "mode": result.mode,
        "verdict": ("exploit-found" if result.flagged
                    else "no-exploit-found"),
        "label": ("expected-value audit (exact DP over the "
                  "stochastic model)" if result.mode == "expected"
                  else "sampled audit (probabilistic verdict)"),
        "intended_expected": getattr(result, "intended_expected", None),
        "optimal_expected": getattr(result, "optimal_expected", None),
        "intended_mean": getattr(result, "intended_mean", None),
        "candidate_mean": getattr(result, "candidate_mean", None),
        "gap": result.gap,
        "gap_ci": [getattr(result, "gap_ci_low", None),
                   getattr(result, "gap_ci_high", None)],
        "margin_required": result.margin,
        "degenerate": result.degenerate,
        "rollout_stats": result.rollout_stats,
        "episodes_spent": getattr(result, "episodes_spent", None),
        "seconds": result.seconds,
        "caveat": CAVEAT,
    }


def stochastic_markdown(result, spec_description=""):
    r = result
    sampled = r.mode == "sampled"
    if r.flagged:
        verdict = ("## Verdict: EXPLOIT FOUND (stochastic audit)"
                   if not sampled else
                   "## Verdict: EXPLOIT FOUND (sampled audit, "
                   "probabilistic)")
        claim = ("A policy earns materially more in expectation than "
                 "the intended policy, and its rollout behaviour is "
                 "degenerate. Inspect the behaviour stats before "
                 "acting on this verdict.")
    else:
        verdict = "## Verdict: no exploit found"
        claim = ("No exploit found under the stated model, horizon, "
                 "search and margin. This is not a certificate that "
                 "the reward is safe.")
    stats = r.rollout_stats.get("candidate", r.rollout_stats)
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
    ]
    if sampled:
        lines += [
            f"| Intended policy mean return (held-out seeds) "
            f"| {r.intended_mean:.4g} |",
            f"| Best found policy mean return (held-out seeds) "
            f"| {r.candidate_mean:.4g} |",
            f"| Gap | {r.gap:.4g} |",
            f"| Gap 95% CI | [{r.gap_ci_low:.4g}, "
            f"{r.gap_ci_high:.4g}] |",
            f"| Episodes spent | {r.episodes_spent} |",
        ]
    else:
        lines += [
            f"| Intended policy expected return "
            f"| {r.intended_expected:.4g} |",
            f"| Optimal expected return | {r.optimal_expected:.4g} |",
            f"| Gap | {r.gap:.4g} |",
            f"| Return range (optimal - worst) "
            f"| {r.return_range:.4g} |",
        ]
    lines += [
        f"| Margin required to flag | {r.margin:.4g} |",
        f"| Degenerate rollouts | {r.degenerate} "
        f"(candidate: top action share "
        f"{stats.get('mean_top_action_share', 0):.2f}, pair dominance "
        f"{stats.get('mean_pair_dominance', 0):.2f}, termination rate "
        f"{stats.get('termination_rate', 0):.2f} over "
        f"{stats.get('episodes', 0)} held-out episodes, judged "
        f"relative to the intended policy's rollouts) |",
        "",
        "## How to read this",
        "",
        CAVEAT,
        "",
    ]
    return "\n".join(lines)
