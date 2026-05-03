"""
Neutrosophic Decision Maker implementation.

Provides decision-making capabilities based on neutrosophic representations,
enabling agents to make choices under uncertainty, ambiguity, and conflict.
"""

from typing import List, Tuple, Optional, Dict, Any
from .triplet import NeutrosophicTriplet
from .operators import NeutrosophicOperators
from .scorer import NeutrosophicScorer
from .consensus import NeutrosophicConsensus


class NeutrosophicDecisionMaker:
    """
    Decision maker for neutrosophic multi-agent systems.
    
    Provides methods to make decisions based on neutrosophic evaluations
    of options, incorporating individual judgments and group consensus.
    """
    
    def __init__(self, scoring_function=None):
        """
        Initialize decision maker.
        
        Args:
            scoring_function: Function to score neutrosophic triplets.
                            If None, uses default accuracy scoring.
        """
        self.scoring_function = scoring_function or NeutrosophicScorer.score_default

    def evaluate_option(
        self, 
        option: Any, 
        judgments: List[NeutrosophicTriplet]
    ) -> Dict[str, Any]:
        """
        Evaluate a single option based on multiple agent judgments.
        
        Args:
            option: The option being evaluated (can be any object)
            judgments: List of neutrosophic judgments from agents
            
        Returns:
            Dictionary with evaluation results
        """
        if not judgments:
            raise ValueError("Cannot evaluate option with empty judgments")
            
        # Compute consensus judgment
        consensus_judgment, confidence = NeutrosophicConsensus.consensus_with_confidence(judgments)
        
        # Score the consensus judgment
        score = self.scoring_function(consensus_judgment)
        
        # Measure agreement/disagreement
        agreement = NeutrosophicConsensus.measure_agreement(judgments)
        has_conflict = NeutrosophicConsensus.detect_conflict(judgments)
        
        return {
            "option": option,
            "consensus_judgment": consensus_judgment,
            "score": score,
            "confidence": confidence,
            "agreement": agreement,
            "has_conflict": has_conflict,
            "individual_judgments": judgments.copy()
        }

    def rank_options(
        self, 
        options_with_judgments: List[Tuple[Any, List[NeutrosophicTriplet]]]
    ) -> List[Dict[str, Any]]:
        """
        Rank multiple options based on their evaluations.
        
        Args:
            options_with_judgments: List of (option, judgments) tuples
            
        Returns:
            List of evaluations sorted by score (descending)
        """
        evaluations = []
        
        for option, judgments in options_with_judgments:
            evaluation = self.evaluate_option(option, judgments)
            evaluations.append(evaluation)
            
        # Sort by score descending (highest score first)
        evaluations.sort(key=lambda x: x["score"], reverse=True)
        
        return evaluations

    def select_best_option(
        self, 
        options_with_judgments: List[Tuple[Any, List[NeutrosophicTriplet]]]
    ) -> Dict[str, Any]:
        """
        Select the best option based on evaluation.
        
        Args:
            options_with_judgments: List of (option, judgments) tuples
            
        Returns:
            Evaluation of the best option
        """
        rankings = self.rank_options(options_with_judgments)
        if not rankings:
            raise ValueError("No options to evaluate")
            
        return rankings[0]

    def make_majority_decision(
        self, 
        options_with_judgments: List[Tuple[Any, List[NeutrosophicTriplet]]]
    ) -> Dict[str, Any]:
        """
        Make decision based on majority rule using scored judgments.
        
        Args:
            options_with_judgments: List of (option, judgments) tuples
            
        Returns:
            Decision result with selected option and reasoning
        """
        rankings = self.rank_options(options_with_judgments)
        if not rankings:
            raise ValueError("No options to evaluate")
            
        best = rankings[0]
        
        # Check if there's a clear winner (score gap > 0.1)
        if len(rankings) > 1:
            score_gap = best["score"] - rankings[1]["score"]
            clear_winner = score_gap > 0.1
        else:
            clear_winner = True
            
        return {
            "decision": "majority_rule",
            "selected_option": best["option"],
            "evaluation": best,
            "alternatives": rankings[1:] if len(rankings) > 1 else [],
            "clear_winner": clear_winner,
            "reasoning": f"Selected option with score {best['score']:.3f} "
                        f"(confidence: {best['confidence']:.3f}, "
                        f"agreement: {best['agreement']:.3f})"
        }

    def make_consensus_decision(
        self, 
        options_with_judgments: List[Tuple[Any, List[NeutrosophicTriplet]]],
        min_agreement: float = 0.6
    ) -> Dict[str, Any]:
        """
        Make decision requiring minimum consensus agreement.
        
        Args:
            options_with_judgments: List of (option, judgments) tuples
            min_agreement: Minimum agreement level required (0-1)
            
        Returns:
            Decision result or indication that consensus wasn't reached
        """
        rankings = self.rank_options(options_with_judgments)
        if not rankings:
            raise ValueError("No options to evaluate")
            
        best = rankings[0]
        
        # Check if best option meets agreement threshold
        consensus_reached = best["agreement"] >= min_agreement
        
        if consensus_reached:
            return {
                "decision": "consensus",
                "selected_option": best["option"],
                "evaluation": best,
                "alternatives": rankings[1:] if len(rankings) > 1 else [],
                "consensus_achieved": True,
                "reasoning": f"Consensus reached with agreement {best['agreement']:.3f} "
                            f">= {min_agreement}"
            }
        else:
            return {
                "decision": "no_consensus",
                "selected_option": None,
                "evaluation": best,
                "alternatives": rankings[1:] if len(rankings) > 1 else [],
                "consensus_achieved": False,
                "reasoning": f"Insufficient agreement: {best['agreement']:.3f} "
                            f"< {min_agreement} required"
            }

    def compute_option_scores(
        self, 
        options_with_judgments: List[Tuple[Any, List[NeutrosophicTriplet]]]
    ) -> List[Tuple[Any, float]]:
        """
        Compute simplified scores for options (option, score) pairs.
        
        Args:
            options_with_judgments: List of (option, judgments) tuples
            
        Returns:
            List of (option, score) tuples sorted by score descending
        """
        rankings = self.rank_options(options_with_judgments)
        return [(eval_result["option"], eval_result["score"]) for eval_result in rankings]