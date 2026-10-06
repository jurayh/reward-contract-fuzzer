# Reward Contract Fuzzer — review brief (5 min)

Status: local spike PASS, not pushed. Waiting on your review before any repo/full build.
Read order: this brief, then SPIKE_NOTES.md if you want detail. COMPETITION.md is the positioning memo.

## Result
- 6/6 planted exploits found, 0/6 clean specs flagged, every hit exact-replay verified.
- Kill criterion was <50% hits or >20% false alarms. Cleared comfortably.
- Cost: $0 LLM, 0.02-0.03s total on the seed corpus.
- Margin is load-bearing: degeneracy alone would false-alarm 5/6 clean specs.
- Random 300-rollout search missed sensor-parking (16 vs DP optimum 188).

## Caveat to hold onto
Seeded corpus only, large gaps, clean intended scripts exactly optimal, small deterministic MDPs.
This proves mechanics, not wild robustness. Borderline, stochastic, larger cases untested.

## Positioning (from COMPETITION.md)
Do not use: "detects reward hacking" — ratctl, BenchShield, Applied Compute Ari already contest that.
Use: "Finds the policy your reward function actually incentivizes, before an agent does."
Pairing: Evalwarden = trust the measurement. Fuzzer = trust what the measurement incentivizes.

## Your decision — one of three
A) Approve next hardening pass in order: (1) borderline corpus, (2) stochastic/larger search, (3) true-utility divergence, (4) real Gymnasium/verifiers/OpenEnv intake, (5) optional LLM proposer with deterministic verifier as authority.
B) Approve only the head-to-head vs ratctl on real env formats, skip borderline corpus for now.
C) Park it. Keep local spike, no further spend.

Recommended: A, with (4) pulled earlier if you want external proof sooner. Do not push a repo until positioning in COMPETITION.md is accepted as-is.

## Files
- ~/workspace/reward-contract-fuzzer/SPIKE_NOTES.md
- ~/workspace/reward-contract-fuzzer/COMPETITION.md
- Reproduce: cd ~/workspace/reward-contract-fuzzer && python3 run_spike.py --json results.json && python3 ablations.py && python3 tests/test_spike.py
