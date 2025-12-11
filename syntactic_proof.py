"""
Syntactic proof checker for epistemic logic (K_n axiom system).

Axiom schemas:
  (A1) All propositional tautologies
  (A2) K_i(φ → ψ) → (K_i φ → K_i ψ)   [Distribution / K axiom]
  
Inference rules:
  (MP)  From φ and φ → ψ, infer ψ           [Modus Ponens]
  (Nec) From ⊢ φ, infer ⊢ K_i φ             [Necessitation]

For S5 (knowledge), add:
  (T)   K_i φ → φ                            [Truth / Reflexivity]
  (4)   K_i φ → K_i K_i φ                    [Positive introspection]
  (5)   ¬K_i φ → K_i ¬K_i φ                  [Negative introspection]
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
from itertools import product
from epistemic_checker import (
    Formula, Prop, Not, And, Or, Imp, BiImp, K, E, C, D,
    P, K_, E_, C_, D_
)


# ---------- Tautology checker (for A1) ----------

def get_props(phi: Formula) -> Set[str]:
    """Extract all proposition names from a formula."""
    if isinstance(phi, Prop):
        return {phi.name}
    if isinstance(phi, Not):
        return get_props(phi.phi)
    if isinstance(phi, (And, Or, Imp, BiImp)):
        return get_props(phi.left) | get_props(phi.right)
    if isinstance(phi, K):
        return get_props(phi.phi)
    return set()


def eval_propositional(phi: Formula, assignment: Dict[str, bool]) -> bool:
    """Evaluate formula under a propositional assignment (ignoring K operators)."""
    if isinstance(phi, Prop):
        return assignment.get(phi.name, False)
    if isinstance(phi, Not):
        return not eval_propositional(phi.phi, assignment)
    if isinstance(phi, And):
        return eval_propositional(phi.left, assignment) and eval_propositional(phi.right, assignment)
    if isinstance(phi, Or):
        return eval_propositional(phi.left, assignment) or eval_propositional(phi.right, assignment)
    if isinstance(phi, Imp):
        return (not eval_propositional(phi.left, assignment)) or eval_propositional(phi.right, assignment)
    if isinstance(phi, BiImp):
        return eval_propositional(phi.left, assignment) == eval_propositional(phi.right, assignment)
    if isinstance(phi, K):
        # Treat K_i φ as a fresh proposition for tautology checking
        return assignment.get(f"__K_{phi.agent}_{phi.phi}", False)
    return False


def is_propositional_tautology(phi: Formula) -> bool:
    """Check if phi is a propositional tautology (A1)."""
    # Collect all "atoms" including K-formulas treated as atoms
    props = get_props(phi)
    k_atoms = collect_k_atoms(phi)
    all_atoms = list(props) + [f"__K_{a}_{p}" for a, p in k_atoms]
    
    if not all_atoms:
        return eval_propositional(phi, {})
    
    for values in product([False, True], repeat=len(all_atoms)):
        assignment = dict(zip(all_atoms, values))
        if not eval_propositional(phi, assignment):
            return False
    return True


def collect_k_atoms(phi: Formula) -> Set[Tuple[int, Formula]]:
    """Collect all K_i(ψ) subformulas."""
    if isinstance(phi, Prop):
        return set()
    if isinstance(phi, Not):
        return collect_k_atoms(phi.phi)
    if isinstance(phi, (And, Or, Imp, BiImp)):
        return collect_k_atoms(phi.left) | collect_k_atoms(phi.right)
    if isinstance(phi, K):
        return {(phi.agent, phi.phi)} | collect_k_atoms(phi.phi)
    return set()


# ---------- Pattern matching for axiom schemas ----------

def matches_distribution(phi: Formula) -> bool:
    """
    Check if phi matches K_i(ψ → χ) → (K_i ψ → K_i χ) [A2/K axiom].
    Also accepts the equivalent conjunctive form: (K_i ψ ∧ K_i(ψ → χ)) → K_i χ
    """
    if not isinstance(phi, Imp):
        return False
    
    antecedent, consequent = phi.left, phi.right
    
    # Form 1: K_i(ψ → χ) → (K_i ψ → K_i χ)
    if isinstance(antecedent, K):
        agent = antecedent.agent
        inner = antecedent.phi
        if isinstance(inner, Imp):
            psi, chi = inner.left, inner.right
            # consequent should be (K_i ψ → K_i χ)
            if isinstance(consequent, Imp):
                if isinstance(consequent.left, K) and isinstance(consequent.right, K):
                    if consequent.left.agent == agent and consequent.right.agent == agent:
                        if consequent.left.phi == psi and consequent.right.phi == chi:
                            return True
    
    # Form 2: (K_i ψ ∧ K_i(ψ → χ)) → K_i χ
    if isinstance(antecedent, And):
        left, right = antecedent.left, antecedent.right
        
        # Try both orderings: (K_i ψ ∧ K_i(ψ→χ)) or (K_i(ψ→χ) ∧ K_i ψ)
        for k_psi, k_imp in [(left, right), (right, left)]:
            if isinstance(k_psi, K) and isinstance(k_imp, K):
                agent = k_psi.agent
                if k_imp.agent != agent:
                    continue
                psi = k_psi.phi
                if isinstance(k_imp.phi, Imp):
                    if k_imp.phi.left == psi:
                        chi = k_imp.phi.right
                        # consequent should be K_i χ
                        if isinstance(consequent, K):
                            if consequent.agent == agent and consequent.phi == chi:
                                return True
    
    return False


def matches_truth(phi: Formula) -> bool:
    """Check if phi matches K_i φ → φ [T axiom]."""
    if not isinstance(phi, Imp):
        return False
    if not isinstance(phi.left, K):
        return False
    return phi.left.phi == phi.right


def matches_positive_introspection(phi: Formula) -> bool:
    """Check if phi matches K_i φ → K_i K_i φ [4 axiom]."""
    if not isinstance(phi, Imp):
        return False
    if not isinstance(phi.left, K):
        return False
    agent = phi.left.agent
    inner = phi.left.phi
    
    if not isinstance(phi.right, K):
        return False
    if phi.right.agent != agent:
        return False
    if not isinstance(phi.right.phi, K):
        return False
    if phi.right.phi.agent != agent:
        return False
    return phi.right.phi.phi == inner


def matches_negative_introspection(phi: Formula) -> bool:
    """Check if phi matches ¬K_i φ → K_i ¬K_i φ [5 axiom]."""
    if not isinstance(phi, Imp):
        return False
    # antecedent: ¬K_i φ
    if not isinstance(phi.left, Not):
        return False
    if not isinstance(phi.left.phi, K):
        return False
    agent = phi.left.phi.agent
    inner = phi.left.phi.phi
    
    # consequent: K_i ¬K_i φ
    if not isinstance(phi.right, K):
        return False
    if phi.right.agent != agent:
        return False
    if not isinstance(phi.right.phi, Not):
        return False
    if not isinstance(phi.right.phi.phi, K):
        return False
    if phi.right.phi.phi.agent != agent:
        return False
    return phi.right.phi.phi.phi == inner


# ---------- Proof line representation ----------

@dataclass
class ProofLine:
    number: int
    formula: Formula
    justification: str  # e.g., "A1", "A2", "T", "4", "5", "MP 1,2", "Nec 3", "Premise"
    
    def __str__(self):
        return f"{self.number}. {self.formula}  [{self.justification}]"


# ---------- Proof checker ----------

class ProofChecker:
    def __init__(self, axiom_system: str = "K"):
        """
        axiom_system: "K" for basic modal logic, "S5" for S5 (adds T, 4, 5)
        """
        self.axiom_system = axiom_system.upper()
    
    def is_axiom(self, phi: Formula) -> Optional[str]:
        """Check if phi is an axiom instance. Returns axiom name or None."""
        if is_propositional_tautology(phi):
            return "A1 (tautology)"
        if matches_distribution(phi):
            return "A2 (distribution/K)"
        if self.axiom_system == "S5":
            if matches_truth(phi):
                return "T (truth)"
            if matches_positive_introspection(phi):
                return "4 (positive introspection)"
            if matches_negative_introspection(phi):
                return "5 (negative introspection)"
        return None
    
    def check_modus_ponens(self, conclusion: Formula, 
                           premises: List[Formula]) -> Optional[Tuple[int, int]]:
        """
        Check if conclusion follows from any pair in premises via MP.
        Returns (index of φ, index of φ→ψ) or None.
        """
        for i, p1 in enumerate(premises):
            for j, p2 in enumerate(premises):
                if isinstance(p2, Imp) and p2.left == p1 and p2.right == conclusion:
                    return (i, j)
        return None
    
    def check_necessitation(self, conclusion: Formula,
                           premises: List[Formula]) -> Optional[int]:
        """
        Check if conclusion = K_i φ follows from φ in premises via Nec.
        Returns index of φ or None.
        """
        if not isinstance(conclusion, K):
            return None
        inner = conclusion.phi
        for i, p in enumerate(premises):
            if p == inner:
                return i
        return None
    
    def verify_line(self, line: ProofLine, 
                    previous_formulas: List[Formula],
                    premises: Set[Formula]) -> Tuple[bool, str]:
        """
        Verify that a proof line is valid given previous lines.
        Returns (is_valid, explanation).
        """
        phi = line.formula
        just = line.justification.strip().upper()
        
        # Check if it's a premise
        if just == "PREMISE" or just.startswith("PREM"):
            if phi in premises:
                return True, "Valid premise"
            return False, f"Not in given premises"
        
        # Check if it's an axiom
        if just.startswith("A1") or just == "TAUT" or just == "TAUTOLOGY":
            if is_propositional_tautology(phi):
                return True, "Valid propositional tautology"
            return False, "Not a propositional tautology"
        
        if just.startswith("A2") or just == "K" or just == "DIST":
            if matches_distribution(phi):
                return True, "Valid distribution axiom instance"
            return False, "Does not match K_i(φ→ψ) → (K_i φ → K_i ψ)"
        
        if just == "T" or just == "TRUTH":
            if matches_truth(phi):
                return True, "Valid truth axiom instance"
            return False, "Does not match K_i φ → φ"
        
        if just == "4" or just == "POS" or just == "PI":
            if matches_positive_introspection(phi):
                return True, "Valid positive introspection axiom"
            return False, "Does not match K_i φ → K_i K_i φ"
        
        if just == "5" or just == "NEG" or just == "NI":
            if matches_negative_introspection(phi):
                return True, "Valid negative introspection axiom"
            return False, "Does not match ¬K_i φ → K_i ¬K_i φ"
        
        # Check Modus Ponens: "MP i,j" or "MP(i,j)" or "MP i, j"
        if just.startswith("MP"):
            # Try to parse specific line numbers from justification
            import re
            match = re.search(r'MP\s*\(?(\d+)\s*[,\s]\s*(\d+)\)?', just)
            if match:
                # Specific lines given - check only those lines
                line1 = int(match.group(1)) - 1  # Convert to 0-indexed
                line2 = int(match.group(2)) - 1
                if line1 < len(previous_formulas) and line2 < len(previous_formulas):
                    p1 = previous_formulas[line1]
                    p2 = previous_formulas[line2]
                    # Check both orderings: p1 and p1→phi, or p2 and p2→phi
                    if isinstance(p2, Imp) and p2.left == p1 and p2.right == phi:
                        return True, f"Valid MP from lines {line1+1} and {line2+1}"
                    if isinstance(p1, Imp) and p1.left == p2 and p1.right == phi:
                        return True, f"Valid MP from lines {line2+1} and {line1+1}"
                return False, f"Cannot derive via MP from lines {line1+1} and {line2+1}"
            else:
                # No specific lines - check all pairs
                result = self.check_modus_ponens(phi, previous_formulas)
                if result:
                    return True, f"Valid MP from lines {result[0]+1} and {result[1]+1}"
            return False, "Cannot derive via modus ponens from previous lines"
        
        # Check Necessitation: "NEC i" or "N i" or "Nec 1"
        if just.startswith("NEC") or (just.startswith("N") and len(just) > 1 and just[1:].strip()[0].isdigit()):
            import re
            match = re.search(r'N(?:EC)?\s*(\d+)', just)
            if match:
                # Specific line given
                line_num = int(match.group(1)) - 1  # Convert to 0-indexed
                if not isinstance(phi, K):
                    return False, "Necessitation requires K_i formula"
                if line_num < len(previous_formulas):
                    if previous_formulas[line_num] == phi.phi:
                        return True, f"Valid necessitation from line {line_num+1}"
                return False, f"Cannot derive via necessitation from line {line_num+1}"
            else:
                result = self.check_necessitation(phi, previous_formulas)
                if result is not None:
                    return True, f"Valid necessitation from line {result+1}"
            return False, "Cannot derive via necessitation from previous lines"
        
        # Auto-detect justification
        axiom = self.is_axiom(phi)
        if axiom:
            return True, f"Auto-detected: {axiom}"
        
        mp = self.check_modus_ponens(phi, previous_formulas)
        if mp:
            return True, f"Auto-detected: MP from lines {mp[0]+1}, {mp[1]+1}"
        
        nec = self.check_necessitation(phi, previous_formulas)
        if nec is not None:
            return True, f"Auto-detected: Necessitation from line {nec+1}"
        
        return False, "Cannot justify this line"
    
    def check_proof(self, lines: List[ProofLine], 
                    premises: Set[Formula] = None) -> List[Tuple[int, bool, str]]:
        """
        Check entire proof. Returns list of (line_number, is_valid, explanation).
        """
        if premises is None:
            premises = set()
        
        results = []
        previous_formulas = []
        
        for line in lines:
            valid, explanation = self.verify_line(line, previous_formulas, premises)
            results.append((line.number, valid, explanation))
            previous_formulas.append(line.formula)
        
        return results


def find_valid_answer(proof_lines: List[ProofLine],
                      blank_line_number: int,
                      choices: List[Formula],
                      premises: Set[Formula] = None,
                      axiom_system: str = "K") -> List[Tuple[Formula, bool, str]]:
    """
    Given a proof with a blank line, test each choice to see which one(s) work.
    
    Returns list of (formula, is_valid, explanation) for each choice.
    """
    checker = ProofChecker(axiom_system)
    if premises is None:
        premises = set()
    
    results = []
    
    for choice in choices:
        # Build proof with this choice filled in
        test_lines = []
        for line in proof_lines:
            if line.number == blank_line_number:
                test_lines.append(ProofLine(line.number, choice, line.justification))
            else:
                test_lines.append(line)
        
        # Check the proof
        check_results = checker.check_proof(test_lines, premises)
        
        # See if all lines pass
        all_valid = all(valid for _, valid, _ in check_results)
        
        # Get explanation for the blank line specifically
        blank_result = next((r for r in check_results if r[0] == blank_line_number), None)
        if blank_result:
            _, valid, expl = blank_result
            results.append((choice, all_valid, expl))
        else:
            results.append((choice, all_valid, "Line not found"))
    
    return results


# ---------- Example usage ----------

if __name__ == "__main__":
    # Run the examples from the separate file
    print("Run 'uv run python syntactic_proof_examples.py' to see examples.")
    print()
    print("Proof Results:")
    
    # ========================================================================
    # Example: Multiple choice question
    # ========================================================================
    print()
    print("=== Example: Multiple choice ===")
    print("Which formula should go on blank line?")
    print()

    # Define propositions
    p, q = P("p"), P("q")

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
    
    # OBS: remember to change accordingly the line number of the blank line and the premises
    results = find_valid_answer(
        proof_with_blank, 
        blank_line_number=3, 
        choices=choices,
        premises={p1 >> p2, p2 >> p3},
    )
    
    print("Results:")
    for i, (formula, valid, expl) in enumerate(results, 1):
        status = "✓ VALID" if valid else "✗ invalid"
        print(f"  {chr(64+i)}) {status} - {expl}")


    p, q = P("p"), P("q")

    # Define the proof with a blank line (formula = None)
    proof = [
        ProofLine(1, K_(1, p >> q), "Premise"),
        ProofLine(2, None, "A2"),  # BLANK LINE
        ProofLine(3, K_(1, p) >> K_(1, q), "MP 1,2"),
    ]

    # Define the multiple choice options
    choices = [
        K_(1, p) >> K_(1, q),                              # A
        K_(1, p >> q) >> (K_(1, p) >> K_(1, q)),           # B (correct)
        K_(1, p >> q) >> K_(1, p),                         # C
    ]

    # Find which answer is correct
    results = find_valid_answer(
        proof,
        blank_line_number=2,
        choices=choices,
        premises={K_(1, p >> q)},
        axiom_system="K",  # or "S5" for S5 axioms, if not specified defaults to "K"
    )
    
    print()
    for i, (formula, valid, explanation) in enumerate(results, 1):
        status = "✓ VALID" if valid else "✗ invalid"
        print(f"{chr(64+i)}) {status} - {explanation}")

