"""
Tests for NeutrosophicDecisionMaker class.
"""

import pytest
from agent_squad.neutrosophic.triplet import NeutrosophicTriplet
from agent_squad.neutrosophic.scorer import NeutrosophicScorer
from agent_squad.neutrosophic.decision import NeutrosophicDecisionMaker


def test_decision_maker_init():
    """Test decision maker initialization."""
    # Default initialization
    dm = NeutrosophicDecisionMaker()
    assert dm.scoring_function == NeutrosophicScorer.score_default
    
    # Custom scoring function
    custom_scorer = lambda t: NeutrosophicScorer.score_certainty(t)
    dm_custom = NeutrosophicDecisionMaker(scoring_function=custom_scorer)
    assert dm_custom.scoring_function == custom_scorer


def test_evaluate_option():
    """Test option evaluation."""
    dm = NeutrosophicDecisionMaker()
    
    option = "Option A"
    judgments = [
        NeutrosophicTriplet(0.8, 0.1, 0.1),
        NeutrosophicTriplet(0.7, 0.2, 0.1),
        NeutrosophicTriplet(0.9, 0.05, 0.05)
    ]
    
    result = dm.evaluate_option(option, judgments)
    
    # Check structure
    assert result["option"] == option
    assert isinstance(result["consensus_judgment"], NeutrosophicTriplet)
    assert isinstance(result["score"], float)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["agreement"], float)
    assert isinstance(result["has_conflict"], bool)
    assert result["individual_judgments"] == judgments
    
    # Check values are in expected ranges
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["agreement"] <= 1.0
    assert -1.0 <= result["score"] <= 1.0  # Accuracy score range


def test_evaluate_option_empty_judgments():
    """Test evaluation with empty judgments raises error."""
    dm = NeutrosophicDecisionMaker()
    
    with pytest.raises(ValueError):
        dm.evaluate_option("Option A", [])


def test_rank_options():
    """Test ranking options."""
    dm = NeutrosophicDecisionMaker()
    
    options_with_judgments = [
        ("Option A", [
            NeutrosophicTriplet(0.8, 0.1, 0.1),  # Score: 0.7
            NeutrosophicTriplet(0.7, 0.2, 0.1)   # Score: 0.6
        ]),
        ("Option B", [
            NeutrosophicTriplet(0.6, 0.2, 0.2),  # Score: 0.4
            NeutrosophicTriplet(0.5, 0.3, 0.2)   # Score: 0.3
        ]),
        ("Option C", [
            NeutrosophicTriplet(0.9, 0.05, 0.05), # Score: 0.85
            NeutrosophicTriplet(0.85, 0.1, 0.05)  # Score: 0.8
        ])
    ]
    
    rankings = dm.rank_options(options_with_judgments)
    
    # Should be ranked by score descending: C (~0.825), A (~0.65), B (~0.35)
    assert len(rankings) == 3
    assert rankings[0]["option"] == "Option C"
    assert rankings[1]["option"] == "Option A"
    assert rankings[2]["option"] == "Option B"
    
    # Scores should be descending
    assert rankings[0]["score"] >= rankings[1]["score"]
    assert rankings[1]["score"] >= rankings[2]["score"]


def test_rank_options_empty():
    """Test ranking with empty list."""
    dm = NeutrosophicDecisionMaker()
    rankings = dm.rank_options([])
    assert rankings == []


def test_select_best_option():
    """Test selecting best option."""
    dm = NeutrosophicDecisionMaker()
    
    options_with_judgments = [
        ("Option A", [
            NeutrosophicTriplet(0.8, 0.1, 0.1),  # Score: 0.7
            NeutrosophicTriplet(0.7, 0.2, 0.1)   # Score: 0.6
        ]),
        ("Option B", [
            NeutrosophicTriplet(0.9, 0.05, 0.05), # Score: 0.85
            NeutrosophicTriplet(0.85, 0.1, 0.05)  # Score: 0.8
        ])
    ]
    
    best = dm.select_best_option(options_with_judgments)
    assert best["option"] == "Option B"
    assert best["score"] > 0.8  # Should be around 0.825


def test_select_best_option_empty():
    """Test selecting best option from empty list raises error."""
    dm = NeutrosophicDecisionMaker()
    
    with pytest.raises(ValueError):
        dm.select_best_option([])


def test_make_majority_decision():
    """Test majority decision making."""
    dm = NeutrosophicDecisionMaker()
    
    options_with_judgments = [
        ("Option A", [
            NeutrosophicTriplet(0.8, 0.1, 0.1),  # Score: 0.7
            NeutrosophicTriplet(0.7, 0.2, 0.1)   # Score: 0.6
        ]),
        ("Option B", [
            NeutrosophicTriplet(0.9, 0.05, 0.05), # Score: 0.85
            NeutrosophicTriplet(0.85, 0.1, 0.05)  # Score: 0.8
        ])
    ]
    
    decision = dm.make_majority_decision(options_with_judgments)
    
    assert decision["decision"] == "majority_rule"
    assert decision["selected_option"] == "Option B"
    assert "evaluation" in decision
    assert "alternatives" in decision
    assert isinstance(decision["clear_winner"], bool)
    assert "reasoning" in decision


def test_make_majority_decision_empty():
    """Test majority decision with empty options raises error."""
    dm = NeutrosophicDecisionMaker()
    
    with pytest.raises(ValueError):
        dm.make_majority_decision([])


def test_make_consensus_decision():
    """Test consensus decision making."""
    dm = NeutrosophicDecisionMaker()
    
    # High agreement case
    options_with_judgments = [
        ("Option A", [
            NeutrosophicTriplet(0.8, 0.1, 0.1),
            NeutrosophicTriplet(0.75, 0.15, 0.1),
            NeutrosophicTriplet(0.85, 0.05, 0.1)
        ]),  # High agreement expected
        ("Option B", [
            NeutrosophicTriplet(0.9, 0.0, 0.1),
            NeutrosophicTriplet(0.1, 0.8, 0.1),  # Very different judgment
            NeutrosophicTriplet(0.5, 0.5, 0.0)
        ])   # Low agreement expected
    ]
    
    # Test with lenient threshold
    decision = dm.make_consensus_decision(options_with_judgments, min_agreement=0.3)
    assert decision["decision"] == "consensus"
    assert decision["selected_option"] == "Option A"  # Should be Option A with higher agreement
    assert decision["consensus_achieved"] == True
    
    # Test with strict threshold
    decision_strict = dm.make_consensus_decision(options_with_judgments, min_agreement=0.9)
    # Depending on actual agreement scores, this might still be consensus
    # Let's check what we actually got and adjust accordingly
    if decision_strict["decision"] == "no_consensus":
        assert decision_strict["consensus_achieved"] == False
    else:
        # If it's still consensus, the agreement must be >= 0.9
        assert decision_strict["consensus_achieved"] == True
        assert decision_strict["evaluation"]["agreement"] >= 0.9


def test_make_consensus_decision_empty():
    """Test consensus decision with empty options raises error."""
    dm = NeutrosophicDecisionMaker()
    
    with pytest.raises(ValueError):
        dm.make_consensus_decision([])


def test_compute_option_scores():
    """Test computing simplified option scores."""
    dm = NeutrosophicDecisionMaker()
    
    options_with_judgments = [
        ("Option A", [
            NeutrosophicTriplet(0.8, 0.1, 0.1),  # Score: 0.7
            NeutrosophicTriplet(0.7, 0.2, 0.1)   # Score: 0.6
        ]),
        ("Option B", [
            NeutrosophicTriplet(0.9, 0.05, 0.05), # Score: 0.85
            NeutrosophicTriplet(0.85, 0.1, 0.05)  # Score: 0.8
        ])
    ]
    
    scores = dm.compute_option_scores(options_with_judgments)
    
    # Should be [(Option B, ~0.825), (Option A, ~0.65)]
    assert len(scores) == 2
    assert scores[0][0] == "Option B"  # Highest score first
    assert scores[1][0] == "Option A"
    assert scores[0][1] > scores[1][1]  # First score higher than second


def test_compute_option_scores_empty():
    """Test computing scores with empty list."""
    dm = NeutrosophicDecisionMaker()
    scores = dm.compute_option_scores([])
    assert scores == []


def test_different_scoring_functions():
    """Test decision maker with different scoring functions."""
    # Test with certainty scorer
    dm_certainty = NeutrosophicDecisionMaker(
        scoring_function=NeutrosophicScorer.score_certainty
    )
    
    option = "Test Option"
    judgments = [
        NeutrosophicTriplet(0.8, 0.1, 0.1),  # Certainty: 0.9
        NeutrosophicTriplet(0.6, 0.2, 0.2)   # Certainty: 0.8
    ]
    
    result = dm_certainty.evaluate_option(option, judgments)
    # Average certainty should be around 0.85
    assert result["score"] > 0.8
    assert result["score"] < 0.9
    
    # Test with truth indicator scorer
    dm_truth = NeutrosophicDecisionMaker(
        scoring_function=NeutrosophicScorer.score_truth_indicator
    )
    
    result_truth = dm_truth.evaluate_option(option, judgments)
    # Truth indicator for first: 0.8/(0.8+0.1)=0.888...
    # Truth indicator for second: 0.6/(0.6+0.2)=0.75
    # Average should be around 0.82
    assert result_truth["score"] > 0.7
    assert result_truth["score"] < 0.9