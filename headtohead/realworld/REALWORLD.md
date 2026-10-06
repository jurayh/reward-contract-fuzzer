# Real-world validation: third-party hub environments

Date: 2026-10-03. Follows the ratctl head-to-head in `HEADTOHEAD.md`.
Question: does either tool behave sensibly on environments neither was
built or tuned on? Nothing here has been pushed.

## What was run

Real third-party code, cloned or installed 2026-10-03:

- Gymnasium 1.3.0 (PyPI): `FrozenLake-v1` (is_slippery=False),
  `Taxi-v4`, `CliffWalking-v1`. Adapter reads each env's published
  transition table `P`; no env logic reimplemented.
- Meta OpenEnv (github.com/meta-pytorch/OpenEnv @ 49aa302):
  `envs/grid_world_env` run through the fuzzer by executing the repo's
  own `grid_world_environment.py` step/reward code (only the FastAPI
  server plumbing was stubbed; transition and reward code untouched).
- ratctl 0.2.0 static audits of six real hub env dirs: verifiers
  (github.com/PrimeIntellect-ai/verifiers @ 484e6de) `gsm8k`, `wordle`,
  `reverse_text`; OpenEnv `grid_world_env`, `connect4_env`, `maze_env`.

Intended baselines, defined as each env's own documentation describes
the task: FrozenLake shortest hole-avoiding path; Taxi an optimal
pickup-and-deliver script from a fixed start state; CliffWalking the
documented SAFE route away from the cliff; grid world the direct
8-step route to the goal.

## Results

Reward Contract Fuzzer on real envs (all replay-verified):

| Env | Intended | Optimal | Gap | Flagged |
|---|---:|---:|---:|:---:|
| FrozenLake-v1 | 1.0 | 1.0 | 0.0 | no |
| Taxi-v4 | 11.0 | 11.0 | 0.0 | no |
| CliffWalking-v1 | -15.0 (safe route) | -13.0 (cliff-edge route) | 2.0 | no |
| OpenEnv grid_world_env | 0.3 | 0.3 | 0.0 | no |

CliffWalking is the informative row. The return-optimal route really
does beat the intended safe route on a real, widely used env, and the
fuzzer correctly does *not* call it an exploit: the gap (2.0) sits
under the margin `max(10, 25%)`, the optimal trajectory terminates
promptly, and it shows no looping or action farming. "A better honest
path exists" and "the reward pays for degenerate behaviour" stay
separate on third-party code, not just on our fixtures. That was the
discrimination the whole flag rule exists for, and it held.

Caveat on Taxi: its intended script was derived from the optimizer's
own policy, so a gap of 0 is partly by construction. The independent
checks are FrozenLake and grid world, whose intended scripts were
written from the task description (BFS path, direct route) and also
landed at exactly 0.

ratctl on real hub envs:

| Env | Format detected | Gameability | Findings |
|---|---|---:|---|
| verifiers gsm8k | verifiers_spec | 0 | 0 |
| verifiers wordle | verifiers_spec | 0 | 0 |
| verifiers reverse_text | verifiers_spec | 0 | 0 |
| OpenEnv maze_env | openenv | 0 | 0 |
| OpenEnv connect4_env | openenv | 4 | 1 (medium) |
| OpenEnv grid_world_env | openenv | 6 | 1 (high: "Hardcoded maximum reward") |

Same signature as the fixture round: production hub envs are mostly
clean under static audit, and the one high finding fires on grid
world's ordinary +1.0 goal reward, the same non-discriminating pattern
that scored an exploitable fixture, a clean fixture, and now a real
env identically (gameability 6 in all three).

## The applicability boundary, stated plainly

The fuzzer applied cleanly to 4 of 4 enumerable control-style envs
(three Gymnasium classics plus OpenEnv grid world). It does not apply,
today, to most of the hub catalogue: the verifiers environments
(gsm8k, wordle, wiki_search, and siblings) and most of OpenEnv's 41
envs (chat, coding, browser, tool-use) are LLM-task environments whose
state is text and whose reward is a rubric, parser, or judge over a
completion. There is no finite state space to run DP over, and no
scripted intended policy in the MDP sense. Claiming coverage there
would require a different discovery stage (learned search, sampled
rollouts, or an LLM proposer with replay as authority), which is the
bounded next step already named in the spike notes, not a result.

So the real-world picture after this pass:

- ratctl's home turf (grader and harness code in LLM-task envs) is
  where most hub volume is, and there it mostly finds production code
  clean. Its tamper detection is real but rarely triggered by shipped
  envs in this sample.
- The fuzzer's home turf (enumerable reward contracts: grid worlds,
  control tasks, simulators with discrete state) is a minority of hub
  volume but a real one, and on it the tool now has third-party
  evidence: no false alarms on four real envs, correct handling of a
  genuinely-better honest path, and (from the fixture round)
  quantified proof on the exploit class static audit cannot separate
  from clean code.
- Positioning consequence: sell the fuzzer first to people shipping
  enumerable RL envs and reward-bearing simulators, and as the
  incentive-integrity complement wherever ratctl-style static audit
  already runs. Do not sell it as a general LLM-env auditor; that
  claim would be false today.

## Limits of this pass

- Four fuzzer envs, all small and deterministic; no stochastic or
  continuous envs, so nothing here speaks to sampled-search discovery.
- ratctl dynamic LLM mode still not run (needs Ollama or an API key).
- No real design-exploitable third-party env was found in the sample,
  so real-world hit rate on exploits is still unproven; the fixture
  round is the only hit-rate evidence. Finding or planting an exploit
  in a real published env (with disclosure to its author) is the
  strongest next evidence and the obvious follow-up.

## Reproduce

```bash
cd ~/workspace/reward-contract-fuzzer/headtohead/realworld
/tmp/gym-venv/bin/python run_gym_fuzzer.py       # writes gym_results.json
/tmp/gym-venv/bin/python run_openenv_fuzzer.py   # writes openenv_results.json
/tmp/ratctl-venv/bin/ratctl audit ../vendor/OpenEnv/envs/grid_world_env \
    --format json -o ratctl_out/OpenEnv_envs_grid_world_env.json
```
