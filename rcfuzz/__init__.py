"""Reward Contract Fuzzer: the optimizer as auditor.

Finds the policy a reward function actually incentivizes, before an
agent does, and proves it by replay.
"""

from .audit import AuditResult, audit_spec
from .mdp import Spec

__version__ = "0.1.0"

__all__ = ["AuditResult", "audit_spec", "Spec", "__version__"]
