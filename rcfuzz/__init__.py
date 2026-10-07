"""Reward Contract Fuzzer: the optimizer as auditor.

Finds the policy a reward function actually incentivizes, before an
agent does, and proves it by replay.
"""

from .audit import AuditResult, audit_spec
from .mdp import Spec
from .stochastic import ProbSpec, audit_stochastic

__version__ = "0.2.0"

__all__ = ["AuditResult", "audit_spec", "Spec", "ProbSpec",
           "audit_stochastic", "__version__"]
