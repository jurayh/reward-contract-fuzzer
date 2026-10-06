# Head-to-head: Reward Contract Fuzzer vs ratctl

Date: 2026-10-03. Status: evidence for the positioning decision, not a
product launch. Nothing here has been pushed.

## The question

The spike proved automated discovery works on a seeded corpus. The open
question was whether the project is *unique* — ratctl already markets
itself as "Fuzz your verifier before an RL agent does" — and whether it
delivers tangible value to a specific person. So both tools were run on
the same environments, in the formats ratctl actually consumes
(Gymnasium-style and OpenEnv-style directories), and scored against
ground truth neither tool could see.

## Method

Test set: 8 environments in `headtohead/envs/`, ground truth in
`headtohead/ground_truth.json`.

- 2 tamper-vulnerable: grader/harness code an agent can abuse
  (`tamper_openenv`: test deletion, frame introspection, git-history
  leak, premature `sys.exit(0)`; `tamper_gym_grader`: a checker that
  returns success from a bare `except`).
- 4 design-exploitable: clean code, no tamper surface, but faithful
  optimisation of the reward beats the intended behaviour
  (`design_farm` loose drop fine, `design_sensor` repeatable scan
  bonus, `design_loiter` survival accumulator, `design_shaping`
  approach shaping with free retreat).
- 2 clean controls (`clean_delivery`, `clean_navigation`).

Tools:

- ratctl 0.2.0 (commit ea2e463), static audit:
  `ratctl audit <env> --format json`. Its dynamic LLM fuzzing mode was
  not run; it needs a local Ollama model or a paid frontier API key.
- Reward Contract Fuzzer: the spike's exact DP optimizer plus replay
  verification, reached through an MDP adapter (`get_state`/`set_state`)
  so the same env code drives both tools. Flag rule unchanged: verified
  replay, return gap above `max(10, 25% of |intended|)`, plus a generic
  degeneracy signal.

A ratctl env counts as flagged when gameability >= 30 (ratctl's own CI
gate is `gameability>0.3`) or any finding is high/critical severity.
Tamper envs have no MDP, so the fuzzer cell there is honestly recorded
as out-of-model: grader tampering is not an action inside the reward
MDP, and pretending to score it would be fabrication.

## Results

| Env | Truth | ratctl | Reward Contract Fuzzer |
|---|---|---|---|
| tamper_openenv | tamper | gameability 39, 4 critical findings, FLAG | out of model (no MDP) |
| tamper_gym_grader | tamper | gameability 20, 4 findings, FLAG | out of model (no MDP) |
| design_farm | design exploit | gameability 6, 1 high finding, FLAG* | intended 18 vs optimal 162, FLAG, verified |
| design_sensor | design exploit | gameability 6, 1 high finding, FLAG* | intended 8 vs optimal 194, FLAG, verified |
| design_loiter | design exploit | gameability 0, 0 findings, pass | intended 13 vs optimal 50, FLAG, verified |
| design_shaping | design exploit | gameability 0, 0 findings, pass | intended 24 vs optimal 62, FLAG, verified |
| clean_delivery | clean | gameability 6, 1 high finding, FLAG* | intended 18 vs optimal 18, pass, verified |
| clean_navigation | clean | gameability 0, 0 findings, pass | intended 7 vs optimal 7, pass, verified |

\* The asterisk is the finding that matters. ratctl's "Hardcoded maximum
reward" high finding fires identically on `design_farm`,
`design_sensor`, and the clean control `clean_delivery` — same
gameability 6 in all three. It does not discriminate an exploitable
reward from a clean one:

- At ratctl's own `gameability>0.3` CI gate, it flags 0 of 4 design
  exploits.
- At any-finding sensitivity, it flags 2 of 4 design exploits and
  false-alarms 1 of 2 clean controls, with identical scores for a real
  exploit and a clean env.
- `design_loiter` and `design_shaping` are total misses: zero findings.
  Nothing in the code looks wrong; the exploit only exists in what an
  optimizer does with the reward over time.

The fuzzer's side is the mirror image. It flags all 4 design exploits
with quantified, replay-verified gaps (the sensor env's optimal policy
is `scan` 39 times and never deliver: 194 vs an intended 8), passes
both clean controls with a gap of exactly 0, and is blind to the tamper
axis by construction. One nuance worth keeping: `clean_delivery`'s
optimal policy *looks* degenerate (it idles with zero-reward cleans
before depositing), but the return gap is 0, so the margin gate — not
the degeneracy signal — is what keeps it unflagged. That matches the
spike ablation: degeneracy alone is not a detector.

Cost: fuzzer 0.5–4.2 ms per env, $0. ratctl static audit ran the whole
set in about a second. Both are cheap enough for CI; cost is not the
differentiator.

## What this proves, and what it does not

Proven on this test set:

- The two tools answer different questions. ratctl asks "can the agent
  cheat the grader?" and answers it well: 2/2 tamper envs with precise,
  actionable findings. The fuzzer asks "does honest optimisation of
  this reward produce the wrong behaviour?" and answers it 4/4 with
  proof, where the static auditor has no discriminating signal.
- The wedge from the competition memo survives contact with the direct
  competitor on its own formats. "Reward hacking detection" as a
  generic claim is ratctl's. "The policy your reward actually
  incentivizes, proven by replay, before you train" is not occupied.

Not proven:

- The test set is ours and hand-built, 8 small deterministic envs. The
  design envs are ports of spike patterns into Gymnasium clothing, not
  third-party hub environments.
- ratctl's dynamic LLM mode might catch some design exploits by
  reasoning about the reward code; it was not run (no local model or
  API key in this pass), so the comparison is static-vs-optimizer only.
- The fuzzer still needs an enumerable MDP adapter and a scripted
  intended policy. Real hub envs will stress both.

## Positioning

Category: reward contract testing — the incentive-integrity half of
reward auditing.

One-liner: Reward Contract Fuzzer finds the policy your reward function
actually incentivizes, before an agent does.

The pairing that keeps both tools honest:

> ratctl tells you your verifier can be cheated. Reward Contract Fuzzer
> shows you the behaviour your reward pays for under faithful
> execution — with the trajectory and the return gap.

Run both. Static audit for the tamper surface, optimizer proof for the
incentive surface. Neither substitutes for the other, and saying so is
part of the credibility.

## Tangible value, per person

- **RL environment authors** (publishing to OpenEnv / Prime Intellect /
  verifiers hubs): a pre-publish check that returns an executable
  exploit trajectory and a return gap they can paste into an issue or
  PR. The sensor env's report is one line an author can act on:
  optimal policy scans 39 times, never delivers, 194 vs intended 8.
  Fix the reward term before the env is public and someone trains on
  it.
- **Benchmark maintainers**: severity in numbers, not vibes. Intended
  vs optimal return ranks which reward to fix first, and a CI gate on
  reward changes (`flag if verified gap > margin`) stops a reward tweak
  from silently reopening a loophole.
- **Post-training teams** (GRPO, RLHF, RLAIF, verifiable-reward RL):
  catch farmable rewards before burning training compute on an agent
  that learns to park at a sensor. The deliverable is a replayable
  trajectory an engineer can run, not a score from a judge model.
- **Eval infrastructure teams**: the complementary half of the audit
  they already want from ratctl-style tooling — same CI slot, different
  failure class.

What the user of the tool walks away with, concretely: a yes/no flag, the
exact exploiting trajectory, intended vs optimal returns, and a replay
that anyone can re-run to confirm. No LLM judge, no trace corpus, no
training run required first.

## Recommended next validation

1. Run the fuzzer adapter against at least one real third-party
   Gymnasium env with enumerable state, and ratctl against real hub
   envs, so the comparison is not only on our own fixtures.
2. If an Ollama model or API budget is approved, run ratctl's dynamic
   mode on the 4 design envs to complete the comparison.
3. Borderline corpus from the spike notes (small-margin exploits,
   near-miss cleans) before any public claim about hit rates.

## Reproduce

```bash
cd ~/workspace/reward-contract-fuzzer/headtohead
python3 build_envs.py
/tmp/ratctl-venv/bin/ratctl audit envs/<env> --format json -o ratctl_out/<env>.json
python3 run_fuzzer.py     # writes fuzzer_out.json
python3 compare.py        # writes comparison.json, prints the matrix
```

Raw outputs: `ratctl_out/*.json`, `fuzzer_out.json`, `comparison.json`.
