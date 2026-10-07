# v0.2 stochastic spike: what breaks, what survives

Date: 2026-10-07. Question: can the v0.1 audit extend to stochastic
envs? v0.1 refuses them (exit 2) because exact DP over a deterministic
model would misreport a stochastic one. Three tracks, one afternoon,
$0 spend. Code and raw numbers in this directory.

## Track 1: expected-value DP where the model is known — works, and
## breaks the flag semantics

Gymnasium stochastic envs expose a transition table with
probabilities, so backward induction over expectations gives the
provably optimal expected return. On slippery FrozenLake (gymnasium
1.4.0, horizon 100):

| Map | Baseline | Baseline E[return] | Optimal E[return] | Gap |
|---|---|---:|---:|---:|
| 4x4 | intended script (open loop) | 0.001 | 0.744 | 0.743 |
| 4x4 | intended policy (greedy BFS route) | 0.046 | 0.744 | 0.699 |
| 8x8 | intended script | 0.000 | 0.641 | 0.641 |
| 8x8 | intended policy | 0.348 | 0.641 | 0.293 |

Monte Carlo cross-check on the real env (20,000 seeded episodes per
policy): optimal 0.742 vs exact 0.744; intended 0.044 vs exact 0.046.
The DP is right.

The interpretation is the problem. FrozenLake's reward is clean;
nobody would call it exploitable. Yet every naive baseline loses by
a wide margin, because on ice the optimal policy is aggressively
reactive (it hugs walls to cancel slide directions) and a script or
greedy policy cannot be. **In stochastic envs, the expected gap
measures baseline competence as much as reward design.** Applying
the v0.1 flag rule to these numbers would convict a clean env. The
machinery extends trivially; the semantics do not.

Two consequences for any v0.2:

- The intended input must become a closed-loop policy, not an action
  script. An open-loop script in a stochastic env is close to a
  strawman (0.001).
- Even against a policy baseline, a gap is advisory, not proof. A
  stochastic flag needs behavioural evidence from the optimal
  policy's own rollouts (farming, looping, refusal to terminate
  across many seeded episodes), not a return difference alone.

## Track 2: black-box sampled search — quality is not the blocker

Where no transition table exists, can seeded-rollout search find the
good policies? Test bed: the same slippery FrozenLake, but the search
sees only a simulator, so Track 1's exact optimum (0.744) is ground
truth. Method: coordinate-ascent hill climbing over the policy
table from the intended policy, candidates scored on a common seed
set (paired comparisons), budget 100k episodes.

Result: **PASS** against the pre-registered criterion (>= 90% of the
exact optimum; kill below 80%). The search spent 56,400 episodes,
and its policy scores 0.732 on 20,000 fresh seeds: 98.4% of the
exact optimum. The paired-seed estimate (0.760) ran about 0.03
optimistic over the fresh-seed value; selection bias exists and any
shipped version must verify on held-out seeds, as this spike did.

## Track 2b: sampled search on a real stochastic env — no false
## exploit manufactured

OpenEnv wildfire at default humidity (0.25, 4x4 configuration,
published reward and spread model, 300 paired episode seeds per
policy). The policy space is a family of 12 reactive heuristics
(three ways to pick which burning cell to water, four idle
behaviours) plus a wait-only baseline:

- Intended rule (water first burning cell, else wait): mean 1.717
- Best alternative family members: 1.704 to 1.708, within noise
- Wait-only: -0.392 (spread burns the grid when nobody fights it)

Nothing beat the intended rule. The idle-behaviour axis collapsed to
identical scores because, as the earlier exploit hunt found, the
episode ends when nothing burns, so idle choices barely execute.
Search over this family manufactures no exploit where the
deterministic hunt also found none; the two methods agree on the
same real env.

## Verdict

Build v0.2, but not as "v0.1 with stochastic allowed." The spike
supports a specific shape:

1. **Expected-DP adapter** for stochastic envs with a known
   transition table: exact, cheap, Monte-Carlo-verifiable. Intended
   input upgrades to a closed-loop policy (a script stays accepted
   for deterministic envs only).
2. **Sampled-search discovery** (hill climbing as here) for small
   model-free envs, with held-out-seed verification and confidence
   intervals in the report. Verdicts in this mode are probabilistic
   and must be labelled as such, unlike v0.1's exact claims.
3. **A different flag rule for stochastic mode.** Return gap alone
   never flags. Require degenerate behaviour in the optimal policy's
   rollout distribution (repeated farming actions, non-termination
   mass, state revisitation far above the intended policy's) in
   addition to a margin-clearing expected gap, and print the
   baseline-competence caveat in every stochastic report.

Kill criteria, honestly scored: Track 2's pre-registered bar passed
(98.4% recovery). Track 1 produced the spike's most valuable output,
a semantics warning that would have shipped as false positives if
v0.2 had been built naively. Nothing in this spike supports auditing
stochastic envs with v0.1's "verified exploit" language, and the
spike recommends against it in writing.

## Reproduce

```bash
cd ~/workspace/reward-contract-fuzzer/v02_spike
/tmp/gym-venv/bin/python track1_expected_dp.py
/tmp/gym-venv/bin/python track2_sampled_search.py
/tmp/gym-venv/bin/python track2b_wildfire.py   # needs the headtohead vendor clone
```
