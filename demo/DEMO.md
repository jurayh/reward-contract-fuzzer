# Demo

Run it yourself: `bash demo/run_demo.sh` (FrozenLake needs
`pip install 'reward-contract-fuzzer[gymnasium]'`; without it you get
the first half). Recorded output below, captured verbatim from that
script on 2026-10-03.

Two audits. The first env pays +5 for every scan at a sensor tile, on
top of the delivery reward it was designed around. The code is clean;
the incentive is not. The second is Gymnasium's FrozenLake with the
intended shortest path as the baseline.

Exit codes are the CI contract: 1 means an exploit was found and
verified by replay, 0 means none was found under the stated model,
2 means the env cannot be audited exactly (for example a stochastic
transition table, which v0.1 refuses rather than misreports).

```console
$ rcfuzz audit examples/sensor-farm/contract.json
# Reward contract audit: sensor-farm

stateful env module /home/hatch/workspace/reward-contract-fuzzer/examples/sensor-farm/env.py

## Verdict: EXPLOIT FOUND (verified by replay)

An optimizing policy earns materially more than the intended behaviour without doing the intended job. The trajectory below is the proof; replay it.

| Quantity | Value |
|---|---:|
| Intended return | 8 |
| Optimal return | 194 |
| Gap | 186 |
| Margin required to flag | 23.4 |
| Return range (optimal - worst) | 234 |
| Degenerate trajectory | True (loop ratio 0.95, top action share 0.97) |
| Replay verified | True |
| Optimal policy terminated | False |
| Horizon | 40 |

## Optimal policy

E scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan scan

## How to read this

The gap is what an optimizer earns over the intended behaviour. It only counts as an exploit when it clears a scale-aware margin (the larger of 25% of the intended return and 10% of the env's return range) and the winning trajectory is degenerate: looping, farming one action, or refusing to finish. A merely better honest route does not flag.

exit code: 1 (1 = exploit found)

$ rcfuzz audit examples/frozenlake/contract.json
# Reward contract audit: frozenlake-clean

Gymnasium env FrozenLake-v1

## Verdict: no exploit found

No exploit found under the stated model, horizon, optimizer and margin. This is not a certificate that the reward is safe.

| Quantity | Value |
|---|---:|
| Intended return | 1 |
| Optimal return | 1 |
| Gap | 0 |
| Margin required to flag | 0.25 |
| Return range (optimal - worst) | 1 |
| Degenerate trajectory | False (loop ratio 0.00, top action share 0.50) |
| Replay verified | True |
| Optimal policy terminated | True |
| Horizon | 6 |

## Optimal policy

D D R D R R

## How to read this

The gap is what an optimizer earns over the intended behaviour. It only counts as an exploit when it clears a scale-aware margin (the larger of 25% of the intended return and 10% of the env's return range) and the winning trajectory is degenerate: looping, farming one action, or refusing to finish. A merely better honest route does not flag.

exit code: 0 (0 = no exploit found)
```

What to notice:

- The sensor farm's proof is the trajectory itself: one step east,
  then `scan` 39 times, never delivering. Intended return 8, optimal
  194. No judge model, no training run; an exact optimizer found it
  and an exact replay confirmed it.
- FrozenLake's gap is exactly 0 against the intended route, so the
  audit passes it. A cleaner result than a score: the reward pays
  for precisely the behaviour it was meant to pay for.
- The margin is scale-aware (larger of 25% of intended return and
  10% of the env's full return range), so the same rule works at
  reward scale 0.02 and at scale 100. It was calibrated on a
  20-spec corpus: 10/10 exploits, 0/10 false alarms.
