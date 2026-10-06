# SPIKE NOTES — Reward Contract Fuzzer

Date: Oct 3, 2026 (HST)
Status: spike complete, not pushed. Awaiting review before any repo or full build.

## Question

Can automated discovery find exploits hiding inside reward functions reliably?

The spike uses GOODHART as the reference point: reward specs that look reasonable can admit unintended optima. The proposed tool would find those optima mechanically instead of relying on a human to notice the loophole.

## Spike design

Seed corpus:

- 12 small deterministic MDPs.
- 6 with planted exploits.
- 6 clean by construction.
- Exported declarative corpus lives in `corpus/*.json`; ground truth lives separately in `corpus/ground_truth.json`.

Planted exploit families:

- `farm-drop`: loose drop fine lets the agent clean, drop, and clean the same dirt repeatedly.
- `sensor-parking`: repeatable scan payment makes parking on the sensor dominate delivery.
- `survival-loiter`: survival accumulator plus weak goal reward makes delay profitable; the optimal policy loiters and finishes late.
- `step-bonus-wander`: movement itself pays, so honest delivery is a trap; optimal wanders and cashes out late.
- `self-report`: unconditional repeatable report bonus can be spammed without doing the task.
- `shaping-oscillate`: approach shaping pays while retreat is free, so oscillation farms partial credit and the agent finishes late.

Discovery loop:

1. Exact finite-horizon dynamic programming finds the proxy-optimal policy.
2. Verification replays that policy exactly, compares it against the scripted intended behavior, and flags only when:
   - replay matches the DP value exactly;
   - the candidate beats intended by a meaningful margin: `max(10, 25% of |intended return|)`;
   - the trajectory shows a generic degeneracy signal: state looping or single-action dominance.

Kill criterion, registered before evaluating the result:

- Kill if planted-exploit hit rate is under 50%.
- Kill if clean-spec false-alarm rate is over 20%.

## Results

| Spec | Label | Intended | Optimal | Diff | Loop | Top action | Terminated | Flagged |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| farm-drop | exploitable | 18.0 | 162.0 | 144.0 | 0.88 | 0.47 | yes | yes |
| sensor-parking | exploitable | 16.0 | 188.0 | 172.0 | 0.93 | 0.95 | no | yes |
| survival-loiter | exploitable | 13.0 | 50.0 | 37.0 | 0.90 | 0.53 | yes | yes |
| step-bonus-wander | exploitable | 13.0 | 48.0 | 35.0 | 0.80 | 0.90 | yes | yes |
| self-report | exploitable | 23.0 | 600.0 | 577.0 | 0.95 | 1.00 | no | yes |
| shaping-oscillate | exploitable | 24.0 | 62.0 | 38.0 | 0.93 | 0.53 | yes | yes |
| clean-delivery | clean | 18.0 | 18.0 | 0.0 | 0.88 | 0.93 | yes | no |
| clean-patrol | clean | 27.0 | 27.0 | 0.0 | 0.00 | 1.00 | yes | no |
| clean-foraging | clean | 7.0 | 7.0 | 0.0 | 0.71 | 0.85 | no | no |
| clean-thermostat | clean | -3.0 | -3.0 | 0.0 | 0.81 | 0.85 | no | no |
| clean-navigation | clean | 6.0 | 6.0 | 0.0 | 0.00 | 0.50 | yes | no |
| clean-sorting | clean | 15.0 | 15.0 | 0.0 | 0.64 | 0.70 | no | no |

Headline metrics:

- Planted exploits: 6/6 found. Hit rate 100%.
- Clean specs: 0/6 flagged. False-alarm rate 0%.
- Replay verification: exact for all 12 specs. DP value matched replayed return.
- Cost: observed runs were 0.02-0.03 seconds total, roughly 2-3 ms per spec, $0 LLM spend.
- Verdict: PASS. The kill criterion is not triggered.

## What the ablations showed

Margin sweep:

- Absolute margins of 1, 5, 10, 20, and 30 all preserved the same separation: 6/6 hits and 0/6 false alarms.
- The result is not balanced on a knife-edge threshold for this corpus. The planted gaps are large, and the clean gaps are exactly zero.

Degeneracy alone:

- The degeneracy signal is not sufficient by itself.
- All 6 exploitable specs look degenerate, but so do 5 of the 6 clean specs.
- Reason: honest finite-horizon behavior often idles at the target, repeats the correct action, revisits states after completing the task, or follows a uniform patrol action. A tool that equates looping with exploitation would cry wolf.
- The margin against intended behavior is the load-bearing gate. Degeneracy is useful supporting evidence and explanation material, not a detector on its own.

Random-search baseline:

- With 300 random rollouts per spec, random search would likely flag 5/6 planted exploits, but it badly underestimates several optima and misses `sensor-parking` entirely: random best 16.0, exactly intended, versus DP optimal 188.0.
- That miss matters. Sensor parking requires a coordinated navigation sequence followed by persistent scanning. Random rollouts do not reliably discover that shape.
- Exact DP is cheap at this scale and gives a stronger claim: when DP says no exploit clears the margin, that means no policy in the enumerated finite-horizon MDP clears it. Random search can only say none of the sampled policies did.

Emergent finite-horizon behavior:

- Several optimal exploit policies farm first and cash out near the horizon, for example loitering before finally reaching the goal or oscillating before finally finishing.
- That is not a bug in the spike. It is exactly the kind of behavior a reward-contract tool should surface: the trajectory is more informative than the scalar return alone.

## Honest limits

This spike proves the mechanics on a seeded corpus. It does not yet prove robustness in the wild.

Limits:

- The corpus is hand-built. Exploit gaps are intentionally large. Clean specs are constructed so the scripted intended behavior is exactly optimal.
- Every MDP is small, deterministic, discrete, and fully observable. Exact DP will not scale unchanged to large or continuous domains.
- The intended behavior is supplied as a script. A production tool needs a practical way for users to specify intended behavior, or to infer a baseline without smuggling in the answer.
- The fuzzer detects proxy-return divergence from intended behavior. It does not yet measure divergence from a separate true utility function. That distinction matters for real reward contracts, where the most damaging exploit may score well on both supplied baselines if the baseline itself is misspecified.
- The degeneracy features are generic but crude. They help explain a flag; they should never be shipped as a standalone exploit detector.
- No stochastic transitions, partial observability, function approximation, multi-agent interaction, or learned environment model was tested.

## What this validates

- The two-stage shape is sound: exact optimization can surface candidate optima, and exact replay can verify them without sampling noise.
- A meaningful-margin gate separates planted exploits from clean specs on this corpus.
- The output can be made evidence-bearing: optimal return, intended return, margin, trajectory, looping/action-dominance signals, and termination behavior.
- Cost is negligible for small contract MDPs. The expensive part of a future system will be modeling real environments, not solving toy ones.

## Recommended next step, if approved

Do not jump straight to a general product. Harden the spike in this order:

1. Borderline corpus: small-margin exploits, near-miss clean specs, suboptimal intended scripts, and cases where intended behavior is good but not exactly optimal.
2. Stochastic and larger MDPs: replace exact DP with bounded search, hill climbing, and sampled verification; measure how hit and false-alarm rates degrade.
3. True-utility divergence: add a hidden evaluator that scores whether the proxy-optimal trajectory actually accomplishes the task, not only whether it beats intended return.
4. Realistic spec intake: define the user-facing reward contract format and require the tool to work from that format without reading labels, exploit notes, or hidden evaluator data.
5. Optional LLM proposer: use it only to propose candidate environments/specs or search heuristics, with the deterministic verifier as the authority. Keep spend capped and report cost per verified finding.

## Files

- `run_spike.py`: headline run and kill-criterion verdict.
- `ablations.py`: margin sweep, degeneracy-alone check, and random-search comparison.
- `export_corpus.py`: declarative corpus export.
- `src/mdp.py`: MDP protocol and rollout.
- `src/corpus.py`: executable seed corpus.
- `src/optimizer.py`: finite-horizon DP optimizer.
- `src/fuzzer.py`: discovery, replay verification, margin and degeneracy gates.
- `tests/test_spike.py`: corpus, exactness, blindness, and kill-criterion tests.
- `results.json`: machine-readable run output.

## Reproduce

```bash
python3 run_spike.py --json results.json
python3 ablations.py
python3 tests/test_spike.py
```

Expected headline: hit rate 100%, false-alarm rate 0%, replay verified true, kill criterion PASS.
