"""Contract loading: a contract is the whole audit input in one file.

JSON format:

  {
    "name": "sensor-farm",
    "env": {"type": "stateful", "module": "env.py", "horizon": 40},
    "intended_actions": ["E", "E", "deposit"]
  }

  {
    "name": "frozenlake",
    "env": {"type": "gymnasium", "id": "FrozenLake-v1",
            "kwargs": {"is_slippery": false},
            "action_names": ["L", "D", "R", "U"], "horizon": 16},
    "intended_actions": ["D", "D", "R", "R", "D", "R"]
  }

Relative module paths resolve against the contract file's directory.
"""

from __future__ import annotations

import json
import os

from . import adapters


def load_contract(path):
    with open(path) as fh:
        contract = json.load(fh)
    base = os.path.dirname(os.path.abspath(path))
    env_cfg = contract["env"]
    intended = contract.get("intended_actions")
    env_type = env_cfg.get("type")
    if env_type == "gymnasium":
        spec = adapters.gymnasium_spec(
            env_cfg["id"],
            intended_actions=intended or [],
            horizon=env_cfg.get("horizon"),
            kwargs=env_cfg.get("kwargs"),
            action_names=env_cfg.get("action_names"),
        )
    elif env_type == "stateful":
        module_path = env_cfg["module"]
        if not os.path.isabs(module_path):
            module_path = os.path.join(base, module_path)
        spec = adapters.stateful_spec(
            module_path,
            intended_actions=intended,
            horizon=env_cfg.get("horizon"),
        )
    else:
        raise adapters.AuditNotApplicable(
            f"unknown env type {env_type!r}; expected 'gymnasium' or "
            "'stateful'"
        )
    name = contract.get("name") or spec.id
    spec.id = name
    return spec, contract
