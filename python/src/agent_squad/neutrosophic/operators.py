"""
Neutrosophic Operators implementation.

Implements basic operators for neutrosophic logic including:
- Negation (NOT)
- Conjunction (AND) 
- Disjunction (OR)
- Implication (IMPLIES)
- Equivalence (IFF)
"""

from .triplet import NeutrosophicTriplet


class NeutrosophicOperators:
    """Collection of static methods for neutrosophic logic operations."""
    
    @staticmethod
    def negate(triplet: NeutrosophicTriplet) -> NeutrosophicTriplet:
        """
        Negation operator: NOT(x) = (F, I, T)
        
        Args:
            triplet: Input neutrosophic triplet
            
        Returns:
            Negated neutrosophic triplet
        """
        return NeutrosophicTriplet(t=triplet.f, i=triplet.i, f=triplet.t)

    @staticmethod
    def conjunction(a: NeutrosophicTriplet, b: NeutrosophicTriplet) -> NeutrosophicTriplet:
        """
        Conjunction operator (AND): 
        T = min(Ta, Tb), I = max(Ia, Ib), F = max(Fa, Fb)
        
        Args:
            a: First neutrosophic triplet
            b: Second neutrosophic triplet
            
        Returns:
            Result of AND operation
        """
        t = min(a.t, b.t)
        i = max(a.i, b.i)
        f = max(a.f, b.f)
        return NeutrosophicTriplet(t=t, i=i, f=f)

    @staticmethod
    def disjunction(a: NeutrosophicTriplet, b: NeutrosophicTriplet) -> NeutrosophicTriplet:
        """
        Disjunction operator (OR):
        T = max(Ta, Tb), I = min(Ia, Ib), F = min(Fa, Fb)
        
        Args:
            a: First neutrosophic triplet
            b: Second neutrosophic triplet
            
        Returns:
            Result of OR operation
        """
        t = max(a.t, b.t)
        i = min(a.i, b.i)
        f = min(a.f, b.f)
        return NeutrosophicTriplet(t=t, i=i, f=f)

    @staticmethod
    def implication(a: NeutrosophicTriplet, b: NeutrosophicTriplet) -> NeutrosophicTriplet:
        """
        Implication operator (IMPLIES): 
        A → B = ¬A ∨ B
        
        Args:
            a: Antecedent neutrosophic triplet
            b: Consequent neutrosophic triplet
            
        Returns:
            Result of implication operation
        """
        not_a = NeutrosophicOperators.negate(a)
        return NeutrosophicOperators.disjunction(not_a, b)

    @staticmethod
    def equivalence(a: NeutrosophicTriplet, b: NeutrosophicTriplet) -> NeutrosophicTriplet:
        """
        Equivalence operator (IFF):
        A ↔ B = (A → B) ∧ (B → A)
        
        Args:
            a: First neutrosophic triplet
            b: Second neutrosophic triplet
            
        Returns:
            Result of equivalence operation
        """
        ab_implication = NeutrosophicOperators.implication(a, b)
        ba_implication = NeutrosophicOperators.implication(b, a)
        return NeutrosophicOperators.conjunction(ab_implication, ba_implication)

    @staticmethod
    def complement(triplet: NeutrosophicTriplet) -> NeutrosophicTriplet:
        """
        Complement operator (standard neutrosophic complement):
        C(x) = (1-T, 1-I, 1-F)
        
        Args:
            triplet: Input neutrosophic triplet
            
        Returns:
            Complemented neutrosophic triplet
        """
        return NeutrosophicTriplet(
            t=1.0 - triplet.t,
            i=1.0 - triplet.i,
            f=1.0 - triplet.f
        )