"""
Tests for NeutrosophicOperators class.
"""

import pytest
from agent_squad.neutrosophic.triplet import NeutrosophicTriplet
from agent_squad.neutrosophic.operators import NeutrosophicOperators


def test_negate():
    """Test negation operator."""
    # Test basic negation
    t = NeutrosophicTriplet(0.7, 0.2, 0.1)
    negated = NeutrosophicOperators.negate(t)
    expected = NeutrosophicTriplet(0.1, 0.2, 0.7)  # F, I, T
    assert negated == expected
    
    # Test negation of extremes
    true_t = NeutrosophicTriplet.boolean_true()
    negated_true = NeutrosophicOperators.negate(true_t)
    expected_false = NeutrosophicTriplet.boolean_false()
    assert negated_true == expected_false
    
    false_t = NeutrosophicTriplet.boolean_false()
    negated_false = NeutrosophicOperators.negate(false_t)
    expected_true = NeutrosophicTriplet.boolean_true()
    assert negated_false == expected_true


def test_conjunction():
    """Test conjunction (AND) operator."""
    # Test basic conjunction
    t1 = NeutrosophicTriplet(0.8, 0.1, 0.1)
    t2 = NeutrosophicTriplet(0.6, 0.2, 0.2)
    conj = NeutrosophicOperators.conjunction(t1, t2)
    expected = NeutrosophicTriplet(
        t=min(0.8, 0.6),  # 0.6
        i=max(0.1, 0.2),  # 0.2
        f=max(0.1, 0.2)   # 0.2
    )
    assert conj == expected
    
    # Test with boolean values
    true_t = NeutrosophicTriplet.boolean_true()
    false_t = NeutrosophicTriplet.boolean_false()
    
    # TRUE AND FALSE = FALSE
    conj_tf = NeutrosophicOperators.conjunction(true_t, false_t)
    assert conj_tf == false_t
    
    # TRUE AND TRUE = TRUE
    conj_tt = NeutrosophicOperators.conjunction(true_t, true_t)
    assert conj_tt == true_t
    
    # FALSE AND FALSE = FALSE
    conj_ff = NeutrosophicOperators.conjunction(false_t, false_t)
    assert conj_ff == false_t


def test_disjunction():
    """Test disjunction (OR) operator."""
    # Test basic disjunction
    t1 = NeutrosophicTriplet(0.8, 0.1, 0.1)
    t2 = NeutrosophicTriplet(0.6, 0.2, 0.2)
    disj = NeutrosophicOperators.disjunction(t1, t2)
    expected = NeutrosophicTriplet(
        t=max(0.8, 0.6),  # 0.8
        i=min(0.1, 0.2),  # 0.1
        f=min(0.1, 0.2)   # 0.1
    )
    assert disj == expected
    
    # Test with boolean values
    true_t = NeutrosophicTriplet.boolean_true()
    false_t = NeutrosophicTriplet.boolean_false()
    
    # TRUE OR FALSE = TRUE
    disj_tf = NeutrosophicOperators.disjunction(true_t, false_t)
    assert disj_tf == true_t
    
    # FALSE OR FALSE = FALSE
    disj_ff = NeutrosophicOperators.disjunction(false_t, false_t)
    assert disj_ff == false_t
    
    # TRUE OR TRUE = TRUE
    disj_tt = NeutrosophicOperators.disjunction(true_t, true_t)
    assert disj_tt == true_t


def test_implication():
    """Test implication operator."""
    # A → B = ¬A ∨ B
    t1 = NeutrosophicTriplet(0.8, 0.1, 0.1)  # Mostly true
    t2 = NeutrosophicTriplet(0.3, 0.2, 0.5)  # Mostly false
    
    impl = NeutrosophicOperators.implication(t1, t2)
    
    # Manual calculation: ¬A ∨ B
    not_t1 = NeutrosophicOperators.negate(t1)  # (0.1, 0.1, 0.8)
    expected = NeutrosophicOperators.disjunction(not_t1, t2)
    assert impl == expected
    
    # Test with boolean logic
    true_t = NeutrosophicTriplet.boolean_true()
    false_t = NeutrosophicTriplet.boolean_false()
    
    # TRUE → TRUE = TRUE
    impl_tt = NeutrosophicOperators.implication(true_t, true_t)
    assert impl_tt == true_t
    
    # TRUE → FALSE = FALSE
    impl_tf = NeutrosophicOperators.implication(true_t, false_t)
    assert impl_tf == false_t
    
    # FALSE → TRUE = TRUE
    impl_ft = NeutrosophicOperators.implication(false_t, true_t)
    assert impl_ft == true_t
    
    # FALSE → FALSE = TRUE
    impl_ff = NeutrosophicOperators.implication(false_t, false_t)
    assert impl_ff == true_t


def test_equivalence():
    """Test equivalence operator."""
    # A ↔ B = (A → B) ∧ (B → A)
    t1 = NeutrosophicTriplet(0.8, 0.1, 0.1)
    t2 = NeutrosophicTriplet(0.8, 0.1, 0.1)  # Same as t1
    
    equiv = NeutrosophicOperators.equivalence(t1, t2)
    # Should be high truth since they're equivalent
    assert equiv.t > 0.5
    
    # Test with boolean logic
    true_t = NeutrosophicTriplet.boolean_true()
    false_t = NeutrosophicTriplet.boolean_false()
    
    # TRUE ↔ TRUE = TRUE
    equiv_tt = NeutrosophicOperators.equivalence(true_t, true_t)
    assert equiv_tt == true_t
    
    # FALSE ↔ FALSE = TRUE
    equiv_ff = NeutrosophicOperators.equivalence(false_t, false_t)
    assert equiv_ff == true_t
    
    # TRUE ↔ FALSE = FALSE
    equiv_tf = NeutrosophicOperators.equivalence(true_t, false_t)
    assert equiv_tf == false_t
    
    # FALSE ↔ TRUE = FALSE
    equiv_ft = NeutrosophicOperators.equivalence(false_t, true_t)
    assert equiv_ft == false_t


def test_complement():
    """Test complement operator."""
    t = NeutrosophicTriplet(0.3, 0.4, 0.3)
    complemented = NeutrosophicOperators.complement(t)
    expected = NeutrosophicTriplet(0.7, 0.6, 0.7)  # 1-T, 1-I, 1-F
    assert complemented == expected
    
    # Test boundary cases
    zero_t = NeutrosophicTriplet(0.0, 0.0, 0.0)
    comp_zero = NeutrosophicOperators.complement(zero_t)
    expected_one = NeutrosophicTriplet(1.0, 1.0, 1.0)
    assert comp_zero == expected_one
    
    one_t = NeutrosophicTriplet(1.0, 1.0, 1.0)
    comp_one = NeutrosophicOperators.complement(one_t)
    expected_zero = NeutrosophicTriplet(0.0, 0.0, 0.0)
    assert comp_one == expected_zero