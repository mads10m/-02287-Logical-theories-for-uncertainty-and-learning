"""
Examples for the syntactic proof checker.
Run this file to see the examples in action.
"""

from syntactic_proof import (
    ProofChecker, ProofLine, find_valid_answer,
    is_propositional_tautology, matches_distribution
)
from epistemic_checker import P, K_


# Define propositions
p, q = P("p"), P("q")


def run_examples():
    # ========================================================================
    # Example 1: Checking K axiom instance
    # ========================================================================
    print("=== Example 1: Checking K axiom instance ===")
    checker = ProofChecker("K")
    
    k_axiom = K_(1, p >> q) >> (K_(1, p) >> K_(1, q))
    result = checker.is_axiom(k_axiom)
    print(f"Formula: {k_axiom}")
    print(f"Is axiom: {result}")
    print()

    # ========================================================================
    # Example 2: A simple proof
    # ========================================================================
    print("=== Example 2: Simple proof ===")
    proof = [
        ProofLine(1, p >> (q >> p), "A1"),  # Tautology
        ProofLine(2, K_(1, p >> (q >> p)), "Nec 1"),  # Necessitation
    ]
    
    results = checker.check_proof(proof)
    for num, valid, expl in results:
        status = "✓" if valid else "✗"
        print(f"  Line {num}: {status} - {expl}")
    print()

    # ========================================================================
    # Example 3: Multiple choice question
    # ========================================================================
    print()
    print("=== Example 3: Multiple choice ===")
    print("Which formula should go on line 3?")
    print()

    # K_1 = K_i
    # shorthands
    p1 = K_(1, p)
    p2 = K_(1, p | q)
    p3 = K_(1, ~p >> q)
    
    # Note: write "Premise" for the first non-axiom lines
    proof_with_blank = [
        ProofLine(1, p1 >> p2, "Premise"),
        ProofLine(2, p2 >> p3, "Premise"),  
        ProofLine(3, None, "A1"),  # BLANK
        ProofLine(4, (p2 >> p3) >> (p1 >> p3), "MP 1,3"),
        ProofLine(5, p1 >> p3, "MP 2,4"),
    ]

    choices = [
        (p1 >> p2) >> ((p2 >> p3) >> (p1 >> p3)),  # Correct (hypothetical syllogism)
        ((p1 & p2) >> p3) >> (p2 >> (p1 >> p3)),   # Wrong
        (p1 >> p2) >> ((p1 >> p3) >> (p2 >> p3)),  # Wrong
    ]
    
    print("Choices:")
    for i, c in enumerate(choices, 1):
        print(f"  {chr(64+i)}) {c}")
    print()
    
    results = find_valid_answer(
        proof_with_blank, 
        blank_line_number=3, 
        choices=choices,
        premises={K_(1, p) >> K_(1, p | q), K_(1, p | q) >> K_(1, ~p >> q)},
    )
    
    print("Results:")
    for i, (formula, valid, expl) in enumerate(results, 1):
        status = "✓ VALID" if valid else "✗ invalid"
        print(f"  {chr(64+i)}) {status} - {expl}")

    # ========================================================================
    # Examples based on the textbook proof: K_i p → K_i(p ∨ q)
    # ========================================================================
    # The full proof:
    # 1. p → (p ∨ q)                                                    (A1)
    # 2. K_i(p → (p ∨ q))                                               (R2/Nec) from 1
    # 3. (K_i p ∧ K_i(p → (p ∨ q))) → K_i(p ∨ q)                        (A2)
    # 4. ((K_i p ∧ K_i(p → (p ∨ q))) → K_i(p ∨ q)) → 
    #       (K_i(p → (p ∨ q)) → (K_i p → K_i(p ∨ q)))                   (A1)
    # 5. K_i(p → (p ∨ q)) → (K_i p → K_i(p ∨ q))                        (R1/MP) 3,4
    # 6. K_i p → K_i(p ∨ q)                                             (R1/MP) 2,5
    # ========================================================================

    print()
    print("=" * 70)
    print("EXAMPLES FROM TEXTBOOK PROOF: K_i p → K_i(p ∨ q)")
    print("=" * 70)

    # Building blocks for this proof
    i = 1  # agent
    psi1 = p >> (p | q)                                              # p → (p ∨ q)
    psi2 = K_(i, p >> (p | q))                                       # K_i(p → (p ∨ q))
    psi3 = (K_(i, p) & K_(i, p >> (p | q))) >> K_(i, p | q)          # (K_i p ∧ K_i(p→(p∨q))) → K_i(p∨q)
    psi4 = psi3 >> (psi2 >> (K_(i, p) >> K_(i, p | q)))              # line 3 → (line 2 → (K_i p → K_i(p∨q)))
    psi5 = psi2 >> (K_(i, p) >> K_(i, p | q))                        # K_i(p→(p∨q)) → (K_i p → K_i(p∨q))
    psi6 = K_(i, p) >> K_(i, p | q)                                  # K_i p → K_i(p ∨ q)

    # ----- Example 6: Line 1 is blank -----
    print()
    print("=== Example 6: Which formula goes on line 1? ===")
    print("Justification: A1 (propositional tautology)")
    print()

    proof_ex6 = [
        ProofLine(1, None, "A1"),  # BLANK
        ProofLine(2, psi2, "Nec 1"),
        ProofLine(3, psi3, "A2"),
        ProofLine(4, psi4, "A1"),
        ProofLine(5, psi5, "MP 3,4"),
        ProofLine(6, psi6, "MP 2,5"),
    ]

    choices_ex6 = [
        p >> (p | q),                # A) Correct - tautology
        p >> (p & q),                # B) Wrong - not a tautology
        (p | q) >> p,                # C) Wrong - not a tautology
        K_(i, p) >> K_(i, p | q),    # D) Wrong - this is the conclusion
    ]

    print("Choices:")
    for j, c in enumerate(choices_ex6, 1):
        print(f"  {chr(64+j)}) {c}")
    print()

    results_ex6 = find_valid_answer(proof_ex6, 1, choices_ex6)
    print("Results:")
    for j, (formula, valid, expl) in enumerate(results_ex6, 1):
        status = "✓ VALID" if valid else "✗ invalid"
        print(f"  {chr(64+j)}) {status} - {expl}")

    # ----- Example 7: Line 2 is blank -----
    print()
    print("=== Example 7: Which formula goes on line 2? ===")
    print("Justification: Nec/R2 (Necessitation from line 1)")
    print()

    proof_ex7 = [
        ProofLine(1, psi1, "A1"),
        ProofLine(2, None, "Nec 1"),  # BLANK
        ProofLine(3, psi3, "A2"),
        ProofLine(4, psi4, "A1"),
        ProofLine(5, psi5, "MP 3,4"),
        ProofLine(6, psi6, "MP 2,5"),
    ]

    choices_ex7 = [
        K_(i, p >> (p | q)),          # A) Correct - K_i(p → (p ∨ q))
        K_(i, p) >> K_(i, p | q),     # B) Wrong - this is the conclusion
        p >> (p | q),                 # C) Wrong - this is line 1, not necessitated
        K_(i, p >> (p & q)),          # D) Wrong - wrong formula inside K
    ]

    print("Choices:")
    for j, c in enumerate(choices_ex7, 1):
        print(f"  {chr(64+j)}) {c}")
    print()

    results_ex7 = find_valid_answer(proof_ex7, 2, choices_ex7)
    print("Results:")
    for j, (formula, valid, expl) in enumerate(results_ex7, 1):
        status = "✓ VALID" if valid else "✗ invalid"
        print(f"  {chr(64+j)}) {status} - {expl}")

    # ----- Example 8: Line 3 is blank -----
    print()
    print("=== Example 8: Which formula goes on line 3? ===")
    print("Justification: A2 (Distribution axiom)")
    print()

    proof_ex8 = [
        ProofLine(1, psi1, "A1"),
        ProofLine(2, psi2, "Nec 1"),
        ProofLine(3, None, "A2"),  # BLANK
        ProofLine(4, psi4, "A1"),
        ProofLine(5, psi5, "MP 3,4"),
        ProofLine(6, psi6, "MP 2,5"),
    ]

    choices_ex8 = [
        (K_(i, p) & K_(i, p >> (p | q))) >> K_(i, p | q),  # A) Correct - A2 instance
        K_(i, p >> (p | q)) >> (K_(i, p) >> K_(i, p | q)), # B) Wrong for this proof
        K_(i, p) >> K_(i, p | q),                          # C) Wrong - conclusion
        (K_(i, p) >> K_(i, p >> (p | q))) >> K_(i, p | q), # D) Wrong structure
    ]

    print("Choices:")
    for j, c in enumerate(choices_ex8, 1):
        print(f"  {chr(64+j)}) {c}")
    print()

    results_ex8 = find_valid_answer(proof_ex8, 3, choices_ex8)
    print("Results:")
    for j, (formula, valid, expl) in enumerate(results_ex8, 1):
        status = "✓ VALID" if valid else "✗ invalid"
        print(f"  {chr(64+j)}) {status} - {expl}")

    # ----- Example 9: Line 5 is blank -----
    print()
    print("=== Example 9: Which formula goes on line 5? ===")
    print("Justification: MP/R1 from lines 3 and 4")
    print()

    proof_ex9 = [
        ProofLine(1, psi1, "A1"),
        ProofLine(2, psi2, "Nec 1"),
        ProofLine(3, psi3, "A2"),
        ProofLine(4, psi4, "A1"),
        ProofLine(5, None, "MP 3,4"),  # BLANK
        ProofLine(6, psi6, "MP 2,5"),
    ]

    choices_ex9 = [
        psi2 >> (K_(i, p) >> K_(i, p | q)),   # A) Correct
        K_(i, p) >> K_(i, p | q),             # B) Wrong - this is line 6
        psi3,                                 # C) Wrong - this is line 3
        (K_(i, p) >> K_(i, p | q)) >> psi2,   # D) Wrong - reversed
    ]

    print("Choices:")
    for j, c in enumerate(choices_ex9, 1):
        print(f"  {chr(64+j)}) {c}")
    print()

    results_ex9 = find_valid_answer(proof_ex9, 5, choices_ex9)
    print("Results:")
    for j, (formula, valid, expl) in enumerate(results_ex9, 1):
        status = "✓ VALID" if valid else "✗ invalid"
        print(f"  {chr(64+j)}) {status} - {expl}")

    # ----- Example 10: Line 6 is blank -----
    print()
    print("=== Example 10: Which formula goes on line 6? ===")
    print("Justification: MP/R1 from lines 2 and 5")
    print()

    proof_ex10 = [
        ProofLine(1, psi1, "A1"),
        ProofLine(2, psi2, "Nec 1"),
        ProofLine(3, psi3, "A2"),
        ProofLine(4, psi4, "A1"),
        ProofLine(5, psi5, "MP 3,4"),
        ProofLine(6, None, "MP 2,5"),  # BLANK
    ]

    choices_ex10 = [
        K_(i, p) >> K_(i, p | q),              # A) Correct - the conclusion
        K_(i, p | q) >> K_(i, p),              # B) Wrong - reversed
        K_(i, p) >> K_(i, p & q),              # C) Wrong - wrong connective
        psi5,                                  # D) Wrong - this is line 5
    ]

    print("Choices:")
    for j, c in enumerate(choices_ex10, 1):
        print(f"  {chr(64+j)}) {c}")
    print()

    results_ex10 = find_valid_answer(proof_ex10, 6, choices_ex10)
    print("Results:")
    for j, (formula, valid, expl) in enumerate(results_ex10, 1):
        status = "✓ VALID" if valid else "✗ invalid"
        print(f"  {chr(64+j)}) {status} - {expl}")


if __name__ == "__main__":
    run_examples()




    # # ========================================================================
    # # Example 4: Distribution axiom proof
    # # ========================================================================
    # print()
    # print("=" * 50)
    # print("=== Example 4: Distribution axiom proof ===")
    # print("Goal: Prove K_1(p → q) → (K_1 p → K_1 q)")
    # print("Which formula should go on line 2?")
    # print()

    # premise = K_(1, p >> q)
    # conclusion = K_(1, p) >> K_(1, q)
    # distribution_instance = premise >> conclusion

    # proof_ex4 = [
    #     ProofLine(1, premise, "Premise"),
    #     ProofLine(2, None, "A2"),  # BLANK
    #     ProofLine(3, conclusion, "MP 1,2"),
    # ]

    # choices_ex4 = [
    #     K_(1, p) >> K_(1, q),                    # A) Wrong - this is the conclusion
    #     distribution_instance,                    # B) Correct - K_1(p→q) → (K_1 p → K_1 q)
    #     K_(1, p >> q) >> K_(1, p),               # C) Wrong
    #     (K_(1, p) & K_(1, p >> q)) >> K_(1, q),  # D) Wrong - not A2 form
    # ]

    # print("Choices:")
    # for i, c in enumerate(choices_ex4, 1):
    #     print(f"  {chr(64+i)}) {c}")
    # print()

    # results_ex4 = find_valid_answer(
    #     proof_ex4,
    #     blank_line_number=2,
    #     choices=choices_ex4,
    #     premises={premise},
    # )

    # print("Results:")
    # for i, (formula, valid, expl) in enumerate(results_ex4, 1):
    #     status = "✓ VALID" if valid else "✗ invalid"
    #     print(f"  {chr(64+i)}) {status} - {expl}")

    # # ========================================================================
    # # Example 5: S5 proof with Truth axiom
    # # ========================================================================
    # print()
    # print("=" * 50)
    # print("=== Example 5: S5 proof with Truth axiom ===")
    # print("Goal: From K_1 p, derive p (using axiom T)")
    # print("Which formula should go on line 2?")
    # print()

    # proof_ex5 = [
    #     ProofLine(1, K_(1, p), "Premise"),
    #     ProofLine(2, None, "T"),  # BLANK
    #     ProofLine(3, p, "MP 1,2"),
    # ]

    # choices_ex5 = [
    #     K_(1, p) >> K_(1, K_(1, p)),    # A) Wrong - axiom 4
    #     p >> K_(1, p),                   # B) Wrong - converse
    #     K_(1, p) >> p,                   # C) Correct - Truth axiom
    #     ~K_(1, p) >> K_(1, ~K_(1, p)),  # D) Wrong - axiom 5
    # ]

    # print("Choices:")
    # for i, c in enumerate(choices_ex5, 1):
    #     print(f"  {chr(64+i)}) {c}")
    # print()

    # results_ex5 = find_valid_answer(
    #     proof_ex5,
    #     blank_line_number=2,
    #     choices=choices_ex5,
    #     premises={K_(1, p)},
    #     axiom_system="S5",
    # )

    # print("Results:")
    # for i, (formula, valid, expl) in enumerate(results_ex5, 1):
    #     status = "✓ VALID" if valid else "✗ invalid"
    #     print(f"  {chr(64+i)}) {status} - {expl}")
