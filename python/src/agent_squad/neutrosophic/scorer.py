"""
Neutrosophic Scorer implementation.

Provides deterministic scoring functions for neutrosophic triplets to enable
ranking and decision making based on neutrosophic representations.
"""

from typing import Callable
from .triplet import NeutrosophicTriplet


class NeutrosophicScorer:
    """
    Deterministic scorer for neutrosophic triplets.
    
    Provides various scoring functions to convert neutrosophic representations
    to crisp values for decision making.
    """
    
    @staticmethod
    def score_accuracy(triplet: NeutrosophicTriplet) -> float:
        """
        Accuracy score: S = T - F
        Higher values indicate more truth than falsity.
        
        Args:
            triplet: Neutrosophic triplet to score
            
        Returns:
            Accuracy score in range [-1, 1]
        """
        return triplet.t - triplet.f

    @staticmethod
    def score_certainty(triplet: NeutrosophicTriplet) -> float:
        """
        Certainty score: S = T + F
        Higher values indicate less indeterminacy.
        
        Args:
            triplet: Neutrosophic triplet to score
            
        Returns:
            Certainty score in range [0, 2]
        """
        return triplet.t + triplet.f

    @staticmethod
    def score_truth_indicator(triplet: NeutrosophicTriplet) -> float:
        """
        Truth indicator score: S = T / (T + F + ε) where ε prevents division by zero
        Measures the proportion of truth among definite values.
        
        Args:
            triplet: Neutrosophic triplet to score
            
        Returns:
            Truth indicator score in range [0, 1]
        """
        denominator = triplet.t + triplet.f
        if denominator == 0.0:
            return 0.0
        return triplet.t / (denominator + 1e-10)

    @staticmethod
    def score_belief_plausibility(triplet: NeutrosophicTriplet) -> tuple[float, float]:
        """
        Belief-Plausibility scoring:
        - Belief: T (direct evidence for truth)
        - Plausibility: T + I (evidence that doesn't contradict truth)
        
        Args:
            triplet: Neutrosophic triplet to score
            
        Returns:
            Tuple of (belief, plausibility) scores
        """
        belief = triplet.t
        plausibility = triplet.t + triplet.i
        return belief, plausibility

    @staticmethod
    def create_weighted_scorer(weights: tuple[float, float, float]) -> Callable[[NeutrosophicTriplet], float]:
        """
        Create a weighted linear scorer: S = w1*T + w2*I + w3*F
        
        Args:
            weights: Tuple of (w_T, w_I, w_F) weights for T, I, F components
            
        Returns:
            Function that takes a NeutrosophicTriplet and returns a score
        """
        w_t, w_i, w_f = weights
        
        def scorer(triplet: NeutrosophicTriplet) -> float:
            return w_t * triplet.t + w_i * triplet.i + w_f * triplet.f
            
        return scorer

    @staticmethod
    def score_default(triplet: NeutrosophicTriplet) -> float:
        """
        Default scoring function: Accuracy score (T - F)
        This provides a deterministic, interpretable ranking.
        
        Args:
            triplet: Neutrosophic triplet to score
            
        Returns:
            Default score (accuracy) in range [-1, 1]
        """
        return NeutrosophicScorer.score_accuracy(triplet)