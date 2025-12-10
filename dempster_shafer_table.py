#!/usr/bin/env python3
"""
Dempster–Shafer table solver.

- Specify the domain, e.g. {'R','Y','G'}.
- Specify rows of the partially filled table as:
    (subset, m, Bel, Plaus)
  where subset is any iterable of domain elements
  and unknown entries are written as None.

The script solves for all unknown masses and then
computes Bel_m and Plaus_m for every subset.
"""

from itertools import combinations
import numpy as np


# ---------- utilities ----------

def powerset(iterable):
    """Generate all subsets (as frozensets)."""
    s = list(iterable)
    for r in range(len(s) + 1):
        for comb in combinations(s, r):
            yield frozenset(comb)


def subset_str(S):
    if not S:
        return "∅"
    return "{" + ",".join(sorted(S)) + "}"


# ---------- core solver ----------

def solve_ds(domain, rows, tol=1e-9):
    """
    domain: iterable of atomic states, e.g. {'R','Y','G'}
    rows: iterable of (subset, m, Bel, Plaus) with None for unknowns

    returns: (mass_dict, belief_dict, plaus_dict)
    """
    W = frozenset(domain)
    all_subsets = list(powerset(W))

    # Collect known masses / beliefs / plausibilities
    known_m = {}
    known_bel = {}
    known_plaus = {}

    for subset, m, bel, plaus in rows:
        S = frozenset(subset)
        if m is not None:
            known_m[S] = float(m)
        if bel is not None:
            known_bel[S] = float(bel)
        if plaus is not None:
            known_plaus[S] = float(plaus)

    # m(∅) = 0 by definition
    if frozenset() not in known_m:
        known_m[frozenset()] = 0.0

    # Unknown mass variables
    unknown_subsets = [S for S in all_subsets if S not in known_m]
    n = len(unknown_subsets)

    # Build linear system A x = b
    A = []
    b = []

    # (1) Normalisation: sum_U m(U) = 1
    row = [1.0] * n
    rhs = 1.0 - sum(known_m.get(S, 0.0) for S in all_subsets)
    A.append(row)
    b.append(rhs)

    # (2) Belief constraints: Bel_m(U) = sum_{S⊆U} m(S)
    for U, val in known_bel.items():
        U = frozenset(U)
        row = [0.0] * n
        rhs = val
        for S in all_subsets:
            if S.issubset(U):
                if S in known_m:
                    rhs -= known_m[S]
                else:
                    idx = unknown_subsets.index(S)
                    row[idx] += 1.0
        A.append(row)
        b.append(rhs)

    # (3) Plausibility constraints: Plaus_m(U) = sum_{S∩U≠∅} m(S)
    for U, val in known_plaus.items():
        U = frozenset(U)
        row = [0.0] * n
        rhs = val
        for S in all_subsets:
            if S and not S.isdisjoint(U):  # ignore ∅
                if S in known_m:
                    rhs -= known_m[S]
                else:
                    idx = unknown_subsets.index(S)
                    row[idx] += 1.0
        A.append(row)
        b.append(rhs)

    A = np.array(A, float)
    b = np.array(b, float)

    # Solve system
    if A.shape[0] == A.shape[1]:
        x = np.linalg.solve(A, b)
    else:
        # Least-squares in case of redundant equations
        x, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)
        if resid.size > 0 and max(abs(resid)) > tol:
            raise ValueError(f"Inconsistent equations, residuals: {resid}")

    # Combine known + solved masses
    m = dict(known_m)
    for S, val in zip(unknown_subsets, x):
        if abs(val) < tol:
            val = 0.0
        m[S] = float(val)

    # Compute Bel and Plaus everywhere
    bel = {}
    plaus = {}
    for U in all_subsets:
        bel_U = 0.0
        plaus_U = 0.0
        for S in all_subsets:
            if S.issubset(U):
                bel_U += m[S]
            if S and not S.isdisjoint(U):
                plaus_U += m[S]
        bel[U] = bel_U
        plaus[U] = plaus_U

    return m, bel, plaus


# ---------- example: your table ----------

if __name__ == "__main__":
    # Domain {R, Y, G}
    domain = {"R", "Y", "G"}

    # Table rows: (subset, m(X), Bel_m(X), Plaus_m(X))
    # Unknown entries are written as None
    rows = [
        (frozenset(), None, None, None),
        ({"R"},      0.1, None, None),
        ({"Y"},      0.1, None, None),
        ({"G"},      0.1, None, None),
        ({"R", "Y"}, None, 0.5, None),
        ({"R", "G"}, 0.2, None, None),
        ({"Y", "G"}, 0.2, None, 0.9),
        ({"R", "Y", "G"}, None, None, None),
    ]

    m, bel, plaus = solve_ds(domain, rows)

    print("Mass function m:")
    for S in sorted(m, key=lambda x: (len(x), sorted(x))):
        print(f"  m({subset_str(S)}) = {m[S]:.3f}")

    print("\nBelief Bel_m:")
    for S in sorted(bel, key=lambda x: (len(x), sorted(x))):
        print(f"  Bel({subset_str(S)}) = {bel[S]:.3f}")

    print("\nPlausibility Plaus_m:")
    for S in sorted(plaus, key=lambda x: (len(x), sorted(x))):
        print(f"  Plaus({subset_str(S)}) = {plaus[S]:.3f}")
