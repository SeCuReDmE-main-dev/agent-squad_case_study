"""
Tests for NeutrosophicScorer class.
"""

import pytest
from agent_squad.neutrosophic.triplet import NeutrosophicTriplet
from agent_squad.neutrosophic.scorer import NeutrosophicScorer


def test_score_accuracy():
    """Test accuracy scoring function."""
    # Test basic accuracy: T - F
    t = NeutrosophicTriplet(0.8, 0.1, 0.1)
    score = NeutrosophicScorer.score_accuracy(t)
    assert abs(score - 0.7) < 1e-10  # 0.8 - 0.1
    
    # Test pure truth
    t_true = NeutrosophicTriplet.boolean_true()
    score_true = NeutrosophicScorer.score_accuracy(t_true)
    assert abs(score_true - 1.0) < 1e-10  # 1.0 - 0.0
    
    # Test pure falsity
    t_false = NeutrosophicTriplet.boolean_false()
    score_false = NeutrosophicScorer.score_accuracy(t_false)
    assert abs(score_false - (-1.0)) < 1e-10  # 0.0 - 1.0
    
    # Test unknown
    t_unknown = NeutrosophicTriplet.unknown()
    score_unknown = NeutrosophicScorer.score_accuracy(t_unknown)
    assert abs(score_unknown - 0.0) < 1e-10  # 0.0 - 0.0
    
    # Test contradiction
    t_contr = NeutrosophicTriplet.contradiction()
    score_contr = NeutrosophicScorer.score_accuracy(t_contr)
    assert abs(score_contr - 0.0) < 1e-10  # 1.0 - 1.0


def test_score_certainty():
    """Test certainty scoring function."""
    # Test certainty: T + F
    t = NeutrosophicTriplet(0.8, 0.1, 0.1)
    score = NeutrosophicScorer.score_certainty(t)
    assert abs(score - 0.9) < 1e-10  # 0.8 + 0.1
    
    # Test pure truth
    t_true = NeutrosophicTriplet.boolean_true()
    score_true = NeutrosophicScorer.score_certainty(t_true)
    assert abs(score_true - 1.0) < 1e-10  # 1.0 + 0.0
    
    # Test pure falsity
    t_false = NeutrosophicTriplet.boolean_false()
    score_false = NeutrosophicScorer.score_certainty(t_false)
    assert abs(score_false - 1.0) < 1e-10  # 0.0 + 1.0
    
    # Test unknown (minimum certainty)
    t_unknown = NeutrosophicTriplet.unknown()
    score_unknown = NeutrosophicScorer.score_certainty(t_unknown)
    assert abs(score_unknown - 0.0) < 1e-10  # 0.0 + 0.0
    
    # Test contradiction (maximum certainty)
    t_contr = NeutrosophicTriplet.contradiction()
    score_contr = NeutrosophicScorer.score_certainty(t_contr)
    assert abs(score_contr - 2.0) < 1e-10  # 1.0 + 1.0


def test_score_truth_indicator():
    """Test truth indicator scoring function."""
    # Test truth indicator: T / (T + F + ε)
    t = NeutrosophicTriplet(0.8, 0.1, 0.2)
    score = NeutrosophicScorer.score_truth_indicator(t)
    expected = 0.8 / (0.8 + 0.2)  # 0.8 / 1.0 = 0.8
    assert abs(score - expected) < 1e-10
    
    # Test pure truth
    t_true = NeutrosophicTriplet.boolean_true()
    score_true = NeutrosophicScorer.score_truth_indicator(t_true)
    # Due to epsilon in denominator, score is slightly less than 1.0
    assert abs(score_true - 1.0) < 1e-9  # 1.0 / (1.0 + 0.0 + 1e-10)
    
    # Test pure falsity
    t_false = NeutrosophicTriplet.boolean_false()
    score_false = NeutrosophicScorer.score_truth_indicator(t_false)
    assert abs(score_false - 0.0) < 1e-10  # 0.0 / (0.0 + 1.0)
    
    # Test unknown (should be 0 when T+F=0)
    t_unknown = NeutrosophicTriplet.unknown()
    score_unknown = NeutrosophicScorer.score_truth_indicator(t_unknown)
    assert abs(score_unknown - 0.0) < 1e-10
    
    # Test contradiction
    t_contr = NeutrosophicTriplet.contradiction()
    score_contr = NeutrosophicScorer.score_truth_indicator(t_contr)
    assert abs(score_contr - 0.5) < 1e-10  # 1.0 / (1.0 + 1.0)


def test_score_belief_plausibility():
    """Test belief-plausibility scoring."""
    t = NeutrosophicTriplet(0.6, 0.3, 0.1)
    belief, plausibility = NeutrosophicScorer.score_belief_plausibility(t)
    
    # Belief = T
    assert abs(belief - 0.6) < 1e-10
    
    # Plausibility = T + I
    assert abs(plausibility - 0.9) < 1e-10  # 0.6 + 0.3
    
    # Test edge cases
    t_true = NeutrosophicTriplet.boolean_true()
    b, p = NeutrosophicScorer.score_belief_plausibility(t_true)
    assert abs(b - 1.0) < 1e-10
    assert abs(p - 1.0) < 1e-10
    
    t_false = NeutrosophicTriplet.boolean_false()
    b, p = NeutrosophicScorer.score_belief_plausibility(t_false)
    assert abs(b - 0.0) < 1e-10
    assert abs(p - 0.0) < 1e-10  # 0.0 + 0.0
    
    t_unknown = NeutrosophicTriplet.unknown()
    b, p = NeutrosophicScorer.score_belief_plausibility(t_unknown)
    assert abs(b - 0.0) < 1e-10
    assert abs(p - 1.0) < 1e-10  # 0.0 + 1.0


def test_create_weighted_scorer():
    """Test weighted scorer creation."""
    # Create scorer with weights (0.5, 0.3, 0.2)
    weights = (0.5, 0.3, 0.2)
    scorer_func = NeutrosophicScorer.create_weighted_scorer(weights)
    
    # Test with triplet (0.8, 0.1, 0.1)
    t = NeutrosophicTriplet(0.8, 0.1, 0.1)
    score = scorer_func(t)
    expected = 0.5*0.8 + 0.3*0.1 + 0.2*0.1  # 0.4 + 0.03 + 0.02 = 0.45
    assert abs(score - expected) < 1e-10
    
    # Test with boolean true
    t_true = NeutrosophicTriplet.boolean_true()
    score_true = scorer_func(t_true)
    expected_true = 0.5*1.0 + 0.3*0.0 + 0.2*0.0  # 0.5
    assert abs(score_true - expected_true) < 1e-10


def test_score_default():
    """Test default scoring function."""
    # Should be same as accuracy score
    t = NeutrosophicTriplet(0.7, 0.2, 0.3)
    default_score = NeutrosophicScorer.score_default(t)
    accuracy_score = NeutrosophicScorer.score_accuracy(t)
    assert abs(default_score - accuracy_score) < 1e-10
    
    # Test with various values
    test_cases = [
        (1.0, 0.0, 0.0),  # Pure truth
        (0.0, 0.0, 1.0),  # Pure falsity
        (0.0, 1.0, 0.0),  # Pure unknown
        (0.5, 0.0, 0.5),  # Equal truth/falsity
        (0.2, 0.6, 0.2),  # High indeterminacy
    ]
    
    for t_val, i_val, f_val in test_cases:
        t = NeutrosophicTriplet(t_val, i_val, f_val)
        default_score = NeutrosophicScorer.score_default(t)
        accuracy_score = NeutrosophicScorer.score_accuracy(t)
        assert abs(default_score - accuracy_score) < 1e-10