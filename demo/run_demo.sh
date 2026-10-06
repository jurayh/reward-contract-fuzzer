#!/usr/bin/env bash
# Demo: one env whose reward pays for the wrong behaviour, one clean env.
# The audit exits 1 on the farm and 0 on the clean env, which is the
# whole product in two commands (and exactly how it gates CI).
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

echo '$ rcfuzz audit examples/sensor-farm/contract.json'
$PY -m rcfuzz.cli audit examples/sensor-farm/contract.json
echo "exit code: $? (1 = exploit found)"
echo
if $PY -c "import gymnasium" 2>/dev/null; then
  echo '$ rcfuzz audit examples/frozenlake/contract.json'
  $PY -m rcfuzz.cli audit examples/frozenlake/contract.json
  echo "exit code: $? (0 = no exploit found)"
else
  echo "FrozenLake demo skipped: gymnasium is not installed"
  echo "  pip install 'reward-contract-fuzzer[gymnasium]'"
fi
