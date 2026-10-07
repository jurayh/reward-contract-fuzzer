# v0.2 build plan — drafted Oct 7, 2026, from the stochastic spike verdict

Status: draft for Yuriy's go / no-go. Nothing here is built or pushed.
Source: v02_spike/STOCHASTIC_SPIKE.md (spike completed locally Oct 7,
nothing pushed, at Yuriy's choice of spike-only over a launch post).

## The one rule the spike bought

Never ship v0.1's flag rule on stochastic envs. On clean slippery
FrozenLake the expected return gap was 0.699 against a greedy intended
policy (0.046 vs the 0.744 optimum), so a gap alone measures baseline
competence, not a reward flaw. Any v0.2 flag needs behavioural
evidence on top of a gap.

## Slice 1 — expected-DP adapter (table-known envs)

- Accept stochastic envs whose transition table is exposed.
- Intended input becomes a closed-loop policy. Open-loop scripts stay
  accepted for deterministic envs only, and are rejected or warned in
  stochastic mode (a script scored 0.001 on FrozenLake: a strawman).
- Cross-check DP results with seeded Monte Carlo in the report
  (spike cross-check: 0.742 sampled vs 0.744 exact).

## Slice 2 — sampled search (small model-free envs)

- Hill-climbing search over the policy table from the intended policy,
  paired-seed scoring during search.
- Mandatory held-out-seed verification before any number is printed:
  paired-seed estimates ran about 0.03 optimistic in the spike
  (0.760 paired vs 0.732 fresh, still 98.4% of the exact optimum).
- Reports in this mode carry confidence intervals and are labelled
  probabilistic. The words "verified exploit" do not appear in
  stochastic mode.

## Slice 3 — the stochastic flag rule

Flag only when all of these hold:
- expected gap clears the margin, AND
- the optimal policy's rollout distribution shows degenerate
  behaviour (farming / repeated actions, non-termination mass, or
  state revisitation far above the intended policy's), AND
- the baseline-competence caveat is printed in the report.

Regression guard from the spike: the wildfire env must stay clean
(best alternative family scored 1.704–1.708 vs the intended rule's
1.717, within noise; wait-only scored -0.392). If a build flags
wildfire, the flag rule is wrong, not the env.

## Out of scope until these land

- Large or continuous envs (search quality beyond small policy tables
  is unproven).
- Any launch post. Yuriy chose spike-only on Oct 7; posting stays a
  separate call.

## Kill / ship criteria for the build

- Ship if: FrozenLake clean envs do not flag, wildfire does not flag,
  and a planted stochastic exploit does flag under the Slice 3 rule.
- Stop and re-spike if the behavioural test cannot separate a planted
  exploit from a merely weak baseline.
