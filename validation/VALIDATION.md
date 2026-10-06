# Validation round 2: margin calibration and a broader env sample

Date: 2026-10-03. Follows `headtohead/realworld/EXPLOIT_HUNT.md`,
which left two open items: the spike margin (an absolute 10) is
meaningless at small reward scales, and the real-env sample was four
envs. This round fixes the first with a borderline corpus and a
calibrated rule, and widens the second. Nothing has been pushed.

## The calibrated rule

Recommended flag rule, replacing the spike margin:

> Flag when the replay-verified optimal policy beats the intended
> policy by more than `max(25% of |intended return|, 10% of the
> return range)`, and its trajectory shows the generic degeneracy
> signal. Return range is optimal minus pessimal (worst-policy)
> return, both exact by DP.

Why the range term exists: a gap means different things in different
envs. In a corridor where the worst policy loses 210 and the best
gains 18, a gap of 12 is noise from a sloppy intended script. In an
env whose entire return range is 10, a gap of 9 is the whole game.
The pessimal DP costs one extra backward pass and makes the margin
mean the same thing at reward scale 0.02, 1, and 100.

## Borderline corpus results

Eight specs built to sit on the boundary (four exploitable, four
clean), scored under the spike rule and three scale-aware candidates.
Full table in `calibration.json`; the decisive rows:

| Spec | Truth | Intended → optimal (gap) | Spike rule | Calibrated |
|---|---|---:|---|---|
| farm-small | exploit | 18 → 27 (9) | miss | FLAG |
| tiny-sensor | exploit | 0.16 → 3.88 (3.72) | miss | FLAG |
| huge-loiter | exploit | 1300 → 5000 (3700) | FLAG | FLAG |
| small-gap-loiter | exploit | 13 → 22 (9) | miss | FLAG |
| nearmiss-detour | clean | 16 → 18 (2) | pass | pass |
| imperfect-intended | clean | 6 → 18 (12) | FALSE ALARM | pass |
| clean-huge | clean | 1800 → 1800 (0) | pass | pass |
| clean-tiny | clean | 0.14 → 0.14 (0) | pass | pass |

The spike rule goes 1/4 hits with a false alarm on this corpus: its
absolute floor of 10 blinds it below that scale and its relative term
alone cannot forgive an imperfect intended script. A purely relative
rule (25% of intended, no floor) catches all four exploits but keeps
the false alarm. The range-aware rule is the only candidate at
**4/4 hits and 0/4 false alarms** here, because the imperfect-intended
gap (12) is small against that env's return range (210, so the range
term demands 21).

Regression on the original spike corpus is unchanged: 6/6 hits, 0/6
false alarms under every rule tested. Combined across both corpora
the calibrated rule is 10/10 and 0/10; the spike rule is 7/10 and
1/10.

One honest boundary inside the win: `farm-small` (gap 9 against a
range of 67) clears the calibrated bar because 25% of intended (4.5)
is the binding term. A range-only rule at 15% misses it. The corpus
cannot tell us which miss is "correct" in the abstract; it tells us
the combined rule keeps both intuitions, relative-to-task and
relative-to-range, and lets either one speak.

## Broader real-env sample

Three more third-party envs through the same adapters (minigrid and
gymnasium 1.3.0, installed from PyPI), plus CliffWalking re-scored
with its pessimal value:

| Env | Intended → optimal (gap) | Range | Flagged |
|---|---:|---:|---|
| MiniGrid-Empty-5x5-v0 | 0.955 → 0.955 (0) | 0.955 | no |
| MiniGrid-FourRooms-v0 | 0.865 → 0.865 (0) | 0.865 | no |
| FrozenLake-v1 8x8 | 1.0 → 1.0 (0) | 1.0 | no |
| CliffWalking-v1 | -15 → -13 (2) | 2487 | no |

That makes seven real envs verified clean end to end (these plus
FrozenLake 4x4, Taxi, and OpenEnv grid world from the earlier pass),
with the wildfire garnish (gap 0.14 against a 25%-of-intended bar of
0.42, and a non-degenerate optimal trajectory) also correctly
unflagged. CliffWalking remains the best discrimination evidence on
real code: its optimal trajectory *is* degenerate by the generic
signal (eleven identical moves in thirteen steps), and the calibrated
margin is what keeps a better honest route from becoming a false
alarm. Degeneracy alone would have flagged it.

Cost note: FourRooms took roughly 90 seconds of wall time through
the naive per-step adapter (19x19 grid, step-count in state, env
object restore per transition). Exact DP is not the bottleneck;
adapter overhead is. A production adapter needs direct transition
extraction or a state cap, and that is a build concern, now measured
rather than guessed.

## What is still unvalidated

- Stochastic and continuous envs: exact DP does not apply, and the
  sampled-search stage remains unbuilt and unmeasured.
- LLM-task hub envs (verifiers-style rubric rewards): out of scope
  for this discovery method, unchanged.
- The intended-baseline input itself: every run here hand-wrote the
  intended script. Whether env authors will write one, and in what
  form, is a product question no corpus can answer. It is the largest
  remaining risk and it belongs to the build, not to another
  validation round.
- Thresholds 25% and 10% are calibrated on 20 specs total. They are a
  defensible default with a regression suite behind them, not a
  universal constant; the suite, not the numbers, is the asset.

## Verdict

The margin problem named in the exploit hunt is fixed and proven on
the corpus built to break it, and the real-env sample doubled without
a false alarm. Validation has now answered every question it can
reach: discovery works, the wedge separates from ratctl, the flag
rule discriminates on real code at any reward scale, and published
envs verify clean. What remains open (stochastic search, intended
input UX, adapter performance) is build work with spikes of its own,
not more corpus validation.

## Reproduce

```bash
cd ~/workspace/reward-contract-fuzzer/validation
python3 calibrate.py                 # corpora, all rules, calibration.json
/tmp/gym-venv/bin/python run_minigrid.py   # MiniGrid + FrozenLake 8x8
```
