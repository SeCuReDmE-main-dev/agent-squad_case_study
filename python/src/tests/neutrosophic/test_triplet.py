"""
Tests for NeutrosophicTriplet class.
"""

import pytest
from agent_squad.neutrosophic.triplet import NeutrosophicTriplet


def test_triplet_creation_valid():
    """Test creating valid neutrosophic triplets."""
    # Test basic creation
    t = NeutrosophicTriplet(0.5, 0.3, 0.2)
    assert t.t == 0.5
    assert t.i == 0.3
    assert t.f == 0.2
    
    # Test boundary values
    t_min = NeutrosophicTriplet(0.0, 0.0, 0.0)
    assert t_min.t == 0.0
    assert t_min.i == 0.0
    assert t_min.f == 0.0
    
    t_max = NeutrosophicTriplet(1.0, 1.0, 1.0)
    assert t_max.t == 1.0
    assert t_max.i == 1.0
    assert t_max.f == 1.0


def test_triplet_creation_invalid():
    """Test that invalid values raise appropriate errors."""
    # Test values outside [0,1] range
    with pytest.raises(ValueError):
        NeutrosophicTriplet(-0.1, 0.5, 0.5)
        
    with pytest.raises(ValueError):
        NeutrosophicTriplet(0.5, 1.1, 0.5)
        
    with pytest.raises(ValueError):
        NeutrosophicTriplet(0.5, 0.5, -0.1)
        
    with pytest.raises(ValueError):
        NeutrosophicTriplet(0.5, 0.5, 1.1)


def test_triplet_equality():
    """Test triplet equality comparison."""
    t1 = NeutrosophicTriplet(0.5, 0.3, 0.2)
    t2 = NeutrosophicTriplet(0.5, 0.3, 0.2)
    t3 = NeutrosophicTriplet(0.5, 0.3, 0.3)
    
    assert t1 == t2
    assert t1 != t3
    assert t1 != "not a triplet"


def test_triplet_special_values():
    """Test special value constructors."""
    # Boolean true
    t_true = NeutrosophicTriplet.boolean_true()
    assert t_true.t == 1.0
    assert t_true.i == 0.0
    assert t_true.f == 0.0
    
    # Boolean false
    t_false = NeutrosophicTriplet.boolean_false()
    assert t_false.t == 0.0
    assert t_false.i == 0.0
    assert t_false.f == 1.0
    
    # Unknown
    t_unknown = NeutrosophicTriplet.unknown()
    assert t_unknown.t == 0.0
    assert t_unknown.i == 1.0
    assert t_unknown.f == 0.0
    
    # Contradiction
    t_contr = NeutrosophicTriplet.contradiction()
    assert t_contr.t == 1.0
    assert t_contr.i == 0.0
    assert t_contr.f == 1.0


def test_triplet_to_from_tuple():
    """Test conversion to and from tuple."""
    original = NeutrosophicTriplet(0.5, 0.3, 0.2)
    tuple_form = original.to_tuple()
    assert tuple_form == (0.5, 0.3, 0.2)
    
    restored = NeutrosophicTriplet.from_tuple(tuple_form)
    assert restored == original


def test_triplet_immutable():
    """Test that triplet is immutable (frozen dataclass)."""
    t = NeutrosophicTriplet(0.5, 0.3, 0.2)
    
    # Attempting to modify should raise FrozenInstanceError
    with pytest.raises(AttributeError):
        t.t = 0.6
        
    with pytest.raises(AttributeError):
        t.i = 0.4
        
    with pytest.raises(AttributeError):
        t.f = 0.1


def test_triplet_string_representation():
    """Test string representations."""
    t = NeutrosophicTriplet(0.5, 0.3, 0.2)
    str_repr = str(t)
    assert "NeutrosophicTriplet" in str_repr
    assert "T=0.500" in str_repr
    assert "I=0.300" in str_repr
    assert "F=0.200" in str_repr
    
    repr_str = repr(t)
    assert repr_str == str_repr