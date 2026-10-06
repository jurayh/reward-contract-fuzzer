#!/usr/bin/env python3
"""Borderline corpus: specs that sit on the flag rule's boundary.

Eight specs, four exploitable and four clean, chosen to stress the
margin rule rather than the optimizer:

  farm-small        farming wins, but the gap is small at spike scale
  tiny-sensor       a sensor farm at 1/50th reward scale (old absolute
                    margin of 10 cannot see it at all)
  huge-loiter       survival loiter at 100x scale (regression: every
                    sane rule must catch it)
  small-gap-loiter  short-horizon loiter, gap 9: under the old absolute
                    margin, over a relative one
  nearmiss-detour   clean world, intended script wastes two moves
  imperfect-intended clean world, intended script makes a long wrong
                    turn (gap > old margin; the degeneracy gate, not
                    the margin, is what must save this one)
  clean-huge        clean delivery at 100x scale
  clean-tiny        clean navigation at 1/50th scale

Labels are evaluation-only; discovery reads the MDP and the script.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mdp import Spec  # noqa: E402


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _corridor(fine, scale=1.0):
    def step(state, action):
        x, carrying, dirt, delivered = state
        r = 0.0
        nx, nc, nd, ndel = x, carrying, dirt, delivered
        if action == "E":
            nx = _clamp(x + 1, 0, 2); r = -1.0
        elif action == "W":
            nx = _clamp(x - 1, 0, 2); r = -1.0
        elif action == "clean":
            if x == 1 and dirt == 1 and carrying == 0:
                nc, nd, r = 1, 0, 10.0
        elif action == "drop":
            if carrying == 1:
                nc = 0; r = fine
                if x == 1:
                    nd = 1
        elif action == "deposit":
            if x == 2 and carrying == 1:
                nc, ndel, r = 0, 1, 10.0
        return (nx, nc, nd, ndel), r * scale, ndel == 1

    return step


def _corridor_spec(spec_id, fine, intended, label, horizon=40, scale=1.0):
    return Spec(
        id=spec_id, description=f"borderline corridor {spec_id}",
        horizon=horizon, initial_state=(0, 0, 1, 0),
        step_fn=_corridor(fine, scale),
        actions_fn=lambda s: ["E", "W", "clean", "drop", "deposit"],
        is_terminal_fn=lambda s: s[3] == 1,
        intended_actions=intended, label=label)


def _sensor_spec(spec_id, scale, label):
    def step(state, action):
        x, delivered = state
        r = 0.0
        nx, ndel = x, delivered
        if action == "E":
            nx = _clamp(x + 1, 0, 2); r = -1.0
        elif action == "W":
            nx = _clamp(x - 1, 0, 2); r = -1.0
        elif action == "scan":
            if x == 1:
                r = 5.0
        elif action == "deposit":
            if x == 2:
                ndel, r = 1, 10.0
        return (nx, ndel), r * scale, ndel == 1

    return Spec(
        id=spec_id, description=f"sensor corridor at scale {scale}",
        horizon=40, initial_state=(0, 0), step_fn=step,
        actions_fn=lambda s: ["E", "W", "scan", "deposit"],
        is_terminal_fn=lambda s: s[1] == 1,
        intended_actions=["E", "E", "deposit"], label=label)


def _loiter_spec(spec_id, survival, goal, horizon, label):
    def step(state, action):
        (x,) = state
        nx = x
        if action == "E":
            nx = _clamp(x + 1, 0, 3)
        elif action == "W":
            nx = _clamp(x - 1, 0, 3)
        r = survival
        done = nx == 3
        if done:
            r += goal
        return (nx,), r, done

    return Spec(
        id=spec_id, description=f"loiter corridor {spec_id}",
        horizon=horizon, initial_state=(0,), step_fn=step,
        actions_fn=lambda s: ["E", "W", "stay"],
        is_terminal_fn=lambda s: s[0] == 3,
        intended_actions=["E", "E", "E"], label=label)


def _nav_spec(spec_id, scale, label):
    def step(state, action):
        (x,) = state
        r = 0.0
        nx = x
        if action == "E":
            nx = _clamp(x + 1, 0, 3); r = -1.0
        elif action == "W":
            nx = _clamp(x - 1, 0, 3); r = -1.0
        done = nx == 3
        if done:
            r += 10.0
        return (nx,), r * scale, done

    return Spec(
        id=spec_id, description=f"navigation at scale {scale}",
        horizon=20, initial_state=(0,), step_fn=step,
        actions_fn=lambda s: ["E", "W", "stay"],
        is_terminal_fn=lambda s: s[0] == 3,
        intended_actions=["E", "E", "E"], label=label)


def borderline_corpus():
    specs = [
        # Farming cycle nets 10 - 9.5 = +0.5 per 2 actions: wins slowly.
        _corridor_spec("farm-small", fine=-9.5,
                       intended=["E", "clean", "E", "deposit"],
                       label="exploitable"),
        _sensor_spec("tiny-sensor", scale=0.02, label="exploitable"),
        _loiter_spec("huge-loiter", survival=100.0, goal=1000.0,
                     horizon=40, label="exploitable"),
        # Horizon 12: loiter 12 + 10 = 22 vs intended 13, gap 9.
        _loiter_spec("small-gap-loiter", survival=1.0, goal=10.0,
                     horizon=12, label="exploitable"),
        # Intended wastes its first two moves bumping the west wall.
        _corridor_spec("nearmiss-detour", fine=-20.0,
                       intended=["W", "W", "E", "clean", "E", "deposit"],
                       label="clean"),
        # Intended takes a long wrong turn after cleaning (12 wasted
        # moves); the optimal route is the honest one.
        _corridor_spec("imperfect-intended", fine=-20.0,
                       intended=["E", "clean"] + ["W", "E"] * 6
                       + ["E", "deposit"],
                       label="clean"),
        _corridor_spec("clean-huge", fine=-20.0,
                       intended=["E", "clean", "E", "deposit"],
                       label="clean", scale=100.0),
        _nav_spec("clean-tiny", scale=0.02, label="clean"),
    ]
    return specs
