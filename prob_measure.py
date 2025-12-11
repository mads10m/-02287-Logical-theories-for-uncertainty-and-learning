from __future__ import annotations
from dataclasses import dataclass
from itertools import chain, combinations
from typing import Dict, Iterable, FrozenSet, Hashable


World = Hashable          # any hashable object: int, str, tuples, ...
Event = FrozenSet[World]  # immutable set of worlds


def powerset(iterable) -> Iterable[Event]:
    """All subsets of a finite iterable as frozensets."""
    s = list(iterable)
    for r in range(len(s) + 1):
        for comb in combinations(s, r):
            yield frozenset(comb)


@dataclass(frozen=True)
class ProbabilitySpace:
    """
    Finite probability space (W, F, μ) where F = P(W) and μ is
    determined by probabilities of singletons {w}.

    We store μ({w}) in `world_probs` and extend μ to all events by
    finite additivity: μ(U) = sum_{w∈U} μ({w}).
    """
    world_probs: Dict[World, float]

    def __post_init__(self):
        # Basic sanity checks: non-negative and sums to 1 (up to rounding)
        if any(p < 0 for p in self.world_probs.values()):
            raise ValueError("All probabilities must be non-negative.")
        total = sum(self.world_probs.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Sum of world probabilities must be 1, got {total}.")

    @property
    def W(self) -> Event:
        """The set W of all possible worlds."""
        return frozenset(self.world_probs.keys())

    # ---- The probability measure μ ----
    def mu(self, U: Iterable[World]) -> float:
        """
        μ(U): probability of event U.

        U can be any iterable of worlds (list, set, frozenset, ...).
        """
        event = frozenset(U)
        return sum(self.world_probs[w] for w in event)

    # ---- Conditional probability measure μ|U ----
    def conditional_measure(self, U: Iterable[World]) -> "ProbabilitySpace":
        """
        Return the conditional probability measure μ|U as a new
        ProbabilitySpace object, in the sense used in the course:

            (μ|U)(V) = μ(V ∩ U) / μ(U),   provided μ(U) ≠ 0.

        We keep the same underlying set W, but effectively
        zero-out everything outside U and renormalise.
        """
        Uset = frozenset(U)
        mu_U = self.mu(Uset)
        if mu_U == 0:
            raise ZeroDivisionError("μ(U) = 0, conditional measure μ|U is undefined.")

        new_world_probs = {}
        for w, p in self.world_probs.items():
            if w in Uset:
                new_world_probs[w] = p / mu_U
            else:
                new_world_probs[w] = 0.0

        return ProbabilitySpace(new_world_probs)

    # Convenience: notation-ish wrapper for μ|U(V)
    def mu_given(self, V: Iterable[World], U: Iterable[World]) -> float:
        """
        Compute (μ|U)(V) directly, i.e. the conditional probability measure
        evaluated at V:

            μ|U(V) = μ(V ∩ U) / μ(U)
        """
        Uset = frozenset(U)
        Vset = frozenset(V)
        mu_U = self.mu(Uset)
        if mu_U == 0:
            raise ZeroDivisionError("μ(U) = 0, conditional probability undefined.")
        return self.mu(Uset & Vset) / mu_U

    # ---- Optional: check finite additivity explicitly on P(W) ----
    def check_axioms(self, verbose: bool = False) -> bool:
        """
        Brute-force check of:
          P1: μ(W) = 1  (already enforced)
          P2: μ(U ∪ V) = μ(U) + μ(V) for all disjoint U, V ⊆ W.

        This is exponential in |W|, so only use for small examples.
        """
        ok = True
        if abs(self.mu(self.W) - 1.0) > 1e-9:
            if verbose:
                print("P1 failed: μ(W) != 1")
            return False

        all_events = list(powerset(self.W))
        for U in all_events:
            for V in all_events:
                if U.isdisjoint(V):
                    left = self.mu(U | V)
                    right = self.mu(U) + self.mu(V)
                    if abs(left - right) > 1e-9:
                        ok = False
                        if verbose:
                            print(f"P2 failed for U={U}, V={V}: "
                                  f"μ(U∪V)={left}, μ(U)+μ(V)={right}")
                        return False
        if verbose:
            print("P1 and P2 hold for all events.")
        return ok

if __name__ == "__main__":
    # Example: fair six-sided die
    world_probs = {i: 1/6 for i in range(1, 7)}
    space = ProbabilitySpace(world_probs)

    # Events
    even = {2, 4, 6}
    gt_3 = {4, 5, 6}

    print("μ(even) =", space.mu(even))          # → 0.5
    print("μ(>3)   =", space.mu(gt_3))          # → 0.5

    # Conditional probability measure μ|_{>3}
    mu_given_gt3 = space.conditional_measure(gt_3)

    print("μ|_{>3}(even) =", mu_given_gt3.mu(even))  # same as μ(even ∩ >3) / μ(>3)

    # Or directly as μ|U(V)
    print("μ|_{>3}(even) =", space.mu_given(even, gt_3))