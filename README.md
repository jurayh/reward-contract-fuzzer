# Reward Contract Fuzzer

**Find the policy your reward function actually incentivizes, before an agent does.**

Give it an environment, its reward function, and the behaviour you intended. It computes the policy the reward actually pays for, verifies the result by exact replay, and reports the precise trajectory and return gap when the two disagree. Run it in CI so a reward change cannot quietly reopen a loophole.

Static auditors ask whether your verifier can be cheated. This asks the other half of the question: whether honest optimisation of your reward produces the wrong behaviour, with the exploiting policy as proof.

## Demo

One env pays +5 per scan next to a +10 delivery reward. An exact optimizer parks on the sensor and scans 39 times instead of delivering: intended return 8, optimal 194, proven by replay.

```console
$ rcfuzz audit examples/sensor-farm/contract.json
## Verdict: EXPLOIT FOUND (verified by replay)
| Intended return | 8 |
| Optimal return | 194 |
| Gap | 186 |
## Optimal policy
E scan scan scan scan scan ... (39 scans, never delivers)
```

Full recorded demo, including a clean FrozenLake pass: [demo/DEMO.md](demo/DEMO.md). Run it yourself with `bash demo/run_demo.sh`.

## Install and use

```bash
pip install reward-contract-fuzzer            # core, no dependencies
pip install 'reward-contract-fuzzer[gymnasium]'   # + Gymnasium envs
```

A contract is one JSON file: the env, and the intended behaviour as an action script.

```json
{
  "name": "sensor-farm",
  "env": {"type": "stateful", "module": "env.py", "horizon": 40},
  "intended_actions": ["E", "E", "deposit"]
}
```

```bash
rcfuzz audit contract.json              # markdown report to stdout
rcfuzz audit contract.json --out report.md --json report.json
```

Exit codes are the CI contract: **0** no exploit found, **1** exploit found and verified, **2** the env cannot be audited exactly. Gate reward changes on it:

```bash
rcfuzz audit contract.json || echo "reward regression: exploit found"
```

Two env adapters ship in v0.1:

- `gymnasium`: any env exposing a deterministic transition table `P` (FrozenLake, Taxi, CliffWalking). Stochastic tables are refused with a clear error rather than misreported.
- `stateful`: your own env module with `reset` / `step` / `get_state` / `set_state` and a module-level `ACTIONS` list. See `examples/sensor-farm/env.py`.

## How it decides

An exploit is flagged only when all three hold:

1. **Replay verified.** The candidate policy's return matches the optimizer's value exactly. No sampling noise, no judge model.
2. **Scale-aware margin.** The gap beats the larger of 25% of the intended return and 10% of the env's return range (best minus worst achievable return, both computed exactly). The same rule behaves at reward scale 0.02 and at scale 100, and it forgives a merely imperfect intended script.
3. **Degenerate winner.** The winning trajectory loops, farms a single action, or refuses to finish. A better honest route to the same goal does not flag: on Gymnasium's CliffWalking, the optimal cliff-edge route beats the safe intended route and is correctly left alone.

The bounded claim matters: a clean report says *no exploit found under the stated model, horizon, optimizer and margin*. It is not a certificate that a reward is safe.

## Evidence

This tool was built spike-first, and the receipts are in the repo:

- Seeded corpus: 6/6 planted exploits found, 0/6 clean false alarms, exact replay, milliseconds per spec, $0 LLM spend ([SPIKE_NOTES.md](SPIKE_NOTES.md)).
- Calibration corpus built to break the margin: the scale-aware rule goes 10/10 hits and 0/10 false alarms where the naive rule fails ([validation/VALIDATION.md](validation/VALIDATION.md)).
- Head-to-head against the closest open-source auditor on shared envs: complementary, not competing; it catches grader tampering, this catches incentive design ([headtohead/HEADTOHEAD.md](headtohead/HEADTOHEAD.md)).
- Real third-party envs (Gymnasium classics, MiniGrid, OpenEnv): verified clean where clean, with CliffWalking's better honest route correctly unflagged ([headtohead/realworld/REALWORLD.md](headtohead/realworld/REALWORLD.md)).

## Scope and limits

v0.1 audits enumerable, deterministic, discrete envs with a scripted intended baseline. It does not cover stochastic or continuous control, and it does not audit LLM-task environments whose reward is a rubric or judge over text. Those need a sampled-search discovery stage, which is deliberately not in this version: exact proof first, broader search second.

## Development

```bash
python3 tests/test_v01.py     # package tests (corpora + adapters)
python3 tests/test_spike.py   # original spike regression suite
```

Project history: the spike, competition memo, head-to-head, and validation rounds all live in this repo (`SPIKE_NOTES.md`, `COMPETITION.md`, `headtohead/`, `validation/`). Inspired by [GOODHART](https://github.com/jurayh/goodhart).

License: MIT.
