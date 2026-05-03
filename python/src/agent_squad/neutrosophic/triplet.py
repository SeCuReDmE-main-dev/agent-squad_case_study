"""
Neutrosophic Triplet representation.

Represents a neutrosophic value as (T, I, F) where:
- T: Truth membership degree [0, 1]
- I: Indeterminacy membership degree [0, 1] 
- F: Falsity membership degree [0, 1]
"""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class NeutrosophicTriplet:
    """
    Immutable neutrosophic triplet (T, I, F) representing truth, indeterminacy, and falsity.
    
    All components are floats in the range [0, 1].
    """
    t: float  # Truth membership
    i: float  # Indeterminacy membership  
    f: float  # Falsity membership

    def __post_init__(self):
        """Validate that all components are in [0, 1] range."""
        if not (0.0 <= self.t <= 1.0):
            raise ValueError(f"Truth value must be in [0, 1], got {self.t}")
        if not (0.0 <= self.i <= 1.0):
            raise ValueError(f"Indeterminacy value must be in [0, 1], got {self.i}")
        if not (0.0 <= self.f <= 1.0):
            raise ValueError(f"Falsity value must be in [0, 1], got {self.f}")

    def __str__(self) -> str:
        return f"NeutrosophicTriplet(T={self.t:.3f}, I={self.i:.3f}, F={self.f:.3f})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, NeutrosophicTriplet):
            return False
        return (abs(self.t - other.t) < 1e-10 and 
                abs(self.i - other.i) < 1e-10 and 
                abs(self.f - other.f) < 1e-10)

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert to tuple representation."""
        return (self.t, self.i, self.f)

    @classmethod
    def from_tuple(cls, values: tuple[float, float, float]) -> 'NeutrosophicTriplet':
        """Create from tuple representation."""
        return cls(t=values[0], i=values[1], f=values[2])

    @classmethod
    def boolean_true(cls) -> 'NeutrosophicTriplet':
        """Create a triplet representing classical boolean true."""
        return cls(t=1.0, i=0.0, f=0.0)

    @classmethod
    def boolean_false(cls) -> 'NeutrosophicTriplet':
        """Create a triplet representing classical boolean false."""  
        return cls(t=0.0, i=0.0, f=1.0)

    @classmethod
    def unknown(cls) -> 'NeutrosophicTriplet':
        """Create a triplet representing completely unknown state."""
        return cls(t=0.0, i=1.0, f=0.0)

    @classmethod
    def contradiction(cls) -> 'NeutrosophicTriplet':
        """Create a triplet representing contradiction."""
        return cls(t=1.0, i=0.0, f=1.0)