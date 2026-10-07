"""Contract loading: a contract is the whole audit input in one file.

JSON format (v0.1, deterministic):

  {
    "name": "sensor-farm",
    "env": {"type": "stateful", "module": "env.py", "horizon": 40},
    "intended_actions": ["E", "E", "deposit"]
  }

v0.2 additions:

  * "mode": "exact" | "expected" | "sampled" at the top level or in
    env (default "auto": deterministic tables audit exactly,
    stochastic tables audit in expectation).
  * "intended_policy": a closed-loop baseline, required for
    stochastic audits. Two forms:
      {"table": {"0": "D", "1": "R"}, "default": "L"}
      {"module": "policy.py"}   # defines act(state, actions) or POLICY
    Table keys are states as strings; they are matched against the
    env's state values (integers when the env's states are ints).

Sampled mode (model-free) uses a live Gymnasium env and also
requires an intended_policy:

  {
    "name": "frozenlake-sampled",
    "mode": "sampled",
    "env": {"type": "gymnasium", "id": "FrozenLake-v1",
            "kwargs": {"is_slippery": true},
            "action_names": ["L", "D", "R", "U"], "horizon": 100},
    "intended_policy": {"table": {...}, "default": "D"},
    "search": {"episodes_per_candidate": 200, "budget": 60000,
               "verify_episodes": 2000}
  }

Relative module paths resolve against the contract file's directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from . import adapters


def _policy_fn_from_cfg(cfg, base):
    """Return fn(state, action_names) -> action name."""
    if cfg is None:
        return None
    if "module" in cfg:
        module_path = cfg["module"]
        if not os.path.isabs(module_path):
            module_path = os.path.join(base, module_path)
        module = adapters._load_module(module_path)
        act = getattr(module, "act", None)
        if act is not None:
            return lambda state, names: act(state, list(names))
        table = getattr(module, "POLICY", None)
        if table is None:
            raise adapters.AuditNotApplicable(
                "policy module must define act(state, actions) or "
                "POLICY")
        cfg = {"table": table, "default": cfg.get("default")}
    table = {}
    for key, value in (cfg.get("table") or {}).items():
        table[key] = value
        try:
            table[int(key)] = value
        except (TypeError, ValueError):
            pass
    default = cfg.get("default")

    def fn(state, names):
        if state in table:
            return table[state]
        if str(state) in table:
            return table[str(state)]
        if default is not None:
            return default
        raise adapters.AuditNotApplicable(
            f"intended_policy has no action for state {state!r} and "
            "no default")

    return fn


@dataclass
class SampledTarget:
    simulator: object
    intended_fn: object  # fn(obs) -> action name
    search_cfg: dict


def load_contract(path):
    with open(path) as fh:
        contract = json.load(fh)
    base = os.path.dirname(os.path.abspath(path))
    env_cfg = contract["env"]
    intended = contract.get("intended_actions")
    env_type = env_cfg.get("type")
    mode = contract.get("mode") or env_cfg.get("mode") or "auto"
    policy_fn = _policy_fn_from_cfg(contract.get("intended_policy"),
                                    base)

    if mode == "sampled":
        if env_type != "gymnasium":
            raise adapters.AuditNotApplicable(
                "sampled mode currently supports gymnasium envs")
        if policy_fn is None:
            raise adapters.AuditNotApplicable(
                "sampled mode requires a closed-loop intended_policy")
        sim = adapters.gymnasium_simulator(
            env_cfg["id"],
            action_names=env_cfg.get("action_names"),
            kwargs=env_cfg.get("kwargs"),
            horizon=env_cfg.get("horizon"))
        name = contract.get("name") or sim.id
        sim.id = name
        target = SampledTarget(
            simulator=sim,
            intended_fn=lambda obs: policy_fn(obs, sim.actions),
            search_cfg=contract.get("search") or {})
        return target, contract

    if env_type == "gymnasium":
        spec = adapters.gymnasium_spec(
            env_cfg["id"],
            intended_actions=intended or [],
            horizon=env_cfg.get("horizon"),
            kwargs=env_cfg.get("kwargs"),
            action_names=env_cfg.get("action_names"),
            intended_policy_fn=policy_fn,
            mode=mode,
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
        if policy_fn is not None:
            names = list(spec.actions(spec.initial_state))
            spec.intended_policy_fn = (
                lambda state, t, _f=policy_fn, _n=names: _f(state, _n))
    else:
        raise adapters.AuditNotApplicable(
            f"unknown env type {env_type!r}; expected 'gymnasium' or "
            "'stateful'"
        )
    name = contract.get("name") or spec.id
    spec.id = name
    return spec, contract
