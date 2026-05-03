"""
Neutrosophic Consensus implementation.

Implements consensus mechanisms for multi-agent systems using neutrosophic logic
to handle uncertainty, disagreement, and incomplete information in group decisions.
"""

from typing import List, Tuple
from .triplet import NeutrosophicTriplet
from .operators import NeutrosophicOperators
from .scorer import NeutrosophicScorer


class NeutrosophicConsensus:
    """
    Consensus mechanisms for neutrosophic multi-agent systems.
    
    Provides methods to compute group consensus from individual neutrosophic
    judgments while properly handling uncertainty and conflict.
    """
    
    @staticmethod
    def compute_consensus(triplets: List[NeutrosophicTriplet]) -> NeutrosophicTriplet:
        """
        Compute consensus using averaging approach.
        
        Args:
            triplets: List of individual agent judgments as neutrosophic triplets
            
        Returns:
            Consensus neutrosophic triplet
        """
        if not triplets:
            raise ValueError("Cannot compute consensus of empty list")
            
        # Compute average of each component
        avg_t = sum(t.t for t in triplets) / len(triplets)
        avg_i = sum(t.i for t in triplets) / len(triplets)
        avg_f = sum(t.f for t in triplets) / len(triplets)
        
        return NeutrosophicTriplet(t=avg_t, i=avg_i, f=avg_f)

    @staticmethod
    def compute_weighted_consensus(
        triplets: List[NeutrosophicTriplet], 
        weights: List[float]
    ) -> NeutrosophicTriplet:
        """
        Compute weighted consensus where agents have different importance.
        
        Args:
            triplets: List of individual agent judgments
            weights: List of weights for each agent (must sum to 1.0)
            
        Returns:
            Weighted consensus neutrosophic triplet
        """
        if not triplets:
            raise ValueError("Cannot compute consensus of empty list")
            
        if len(triplets) != len(weights):
            raise ValueError("Number of triplets must match number of weights")
            
        if abs(sum(weights) - 1.0) > 1e-10:
            raise ValueError("Weights must sum to 1.0")
            
        # Compute weighted average of each component
        weighted_t = sum(t.t * w for t, w in zip(triplets, weights))
        weighted_i = sum(t.i * w for t, w in zip(triplets, weights))
        weighted_f = sum(t.f * w for t, w in zip(triplets, weights))
        
        return NeutrosophicTriplet(t=weighted_t, i=weighted_i, f=weighted_f)

    @staticmethod
    def compute_median_consensus(triplets: List[NeutrosophicTriplet]) -> NeutrosophicTriplet:
        """
        Compute consensus using median approach (more robust to outliers).
        
        Args:
            triplets: List of individual agent judgments
            
        Returns:
            Median consensus neutrosophic triplet
        """
        if not triplets:
            raise ValueError("Cannot compute consensus of empty list")
            
        # Sort by each component and take median
        t_values = sorted([t.t for t in triplets])
        i_values = sorted([t.i for t in triplets])
        f_values = sorted([t.f for t in triplets])
        
        n = len(triplets)
        if n % 2 == 1:
            # Odd number - take middle value
            median_t = t_values[n // 2]
            median_i = i_values[n // 2]
            median_f = f_values[n // 2]
        else:
            # Even number - average middle two values
            median_t = (t_values[n // 2 - 1] + t_values[n // 2]) / 2
            median_i = (i_values[n // 2 - 1] + i_values[n // 2]) / 2
            median_f = (f_values[n // 2 - 1] + f_values[n // 2]) / 2
            
        return NeutrosophicTriplet(t=median_t, i=median_i, f=median_f)

    @staticmethod
    def measure_agreement(triplets: List[NeutrosophicTriplet]) -> float:
        """
        Measure level of agreement among agents using variance.
        Returns agreement score in [0, 1] where 1 is perfect agreement.
        
        Args:
            triplets: List of individual agent judgments
            
        Returns:
            Agreement score (1 - normalized variance)
        """
        if len(triplets) <= 1:
            return 1.0
            
        # Compute variance for each component
        t_values = [t.t for t in triplets]
        i_values = [t.i for t in triplets]
        f_values = [t.f for t in triplets]
        
        # Mean values
        mean_t = sum(t_values) / len(t_values)
        mean_i = sum(i_values) / len(i_values)
        mean_f = sum(f_values) / len(f_values)
        
        # Variances
        var_t = sum((t - mean_t) ** 2 for t in t_values) / len(t_values)
        var_i = sum((i - mean_i) ** 2 for i in i_values) / len(i_values)
        var_f = sum((f - mean_f) ** 2 for f in f_values) / len(f_values)
        
        # Average variance (maximum possible variance is 0.25 for [0,1] range)
        avg_variance = (var_t + var_i + var_f) / 3
        max_variance = 0.25
        
        # Agreement is inverse of normalized variance
        agreement = max(0.0, 1.0 - (avg_variance / max_variance))
        return agreement

    @staticmethod
    def detect_conflict(triplets: List[NeutrosophicTriplet], threshold: float = 0.5) -> bool:
        """
        Detect if there is significant conflict among agents.
        
        Args:
            triplets: List of individual agent judgments
            threshold: Agreement threshold below which conflict is detected
            
        Returns:
            True if significant conflict detected, False otherwise
        """
        agreement = NeutrosophicConsensus.measure_agreement(triplets)
        return agreement < threshold

    @staticmethod
    def consensus_with_confidence(
        triplets: List[NeutrosophicTriplet]
    ) -> Tuple[NeutrosophicTriplet, float]:
        """
        Compute consensus along with confidence measure.
        
        Args:
            triplets: List of individual agent judgments
            
        Returns:
            Tuple of (consensus_triplet, confidence_score)
        """
        consensus = NeutrosophicConsensus.compute_consensus(triplets)
        confidence = NeutrosophicConsensus.measure_agreement(triplets)
        return consensus, confidence