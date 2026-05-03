"""
Neutrosophic Core Engine for Multi-Agent Decision Making.

This package implements a neutrosophic logic-based framework for representing
and processing uncertain, incomplete, and contradictory information in multi-agent
systems.
"""

from .triplet import NeutrosophicTriplet
from .operators import NeutrosophicOperators
from .scorer import NeutrosophicScorer
from .consensus import NeutrosophicConsensus
from .decision import NeutrosophicDecisionMaker

__all__ = [
    "NeutrosophicTriplet",
    "NeutrosophicOperators", 
    "NeutrosophicScorer",
    "NeutrosophicConsensus",
    "NeutrosophicDecisionMaker"
]