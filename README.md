

# Run

## Epistemic Checker
```bash
uv run epistemic_checker.py
```

### Char to logical
- `%`: BICONDITIONAL (`↔`)
- `>>`: IMPLIES (`→`)
- `XOR`: EXCLUSIVE OR (`⊕`)
- `&`: AND (`∧`)
- `|`: OR (`∨`)
- `~`: NOT (`¬`)
- `(` and `)`: Parentheses for grouping

Epistemic operators:
- `K_(n, phi)`: Knowledge operator for agent n and formula phi
- `E_({1,2,..}, phi)`: Epistemic operator for agents 1,2,.. and formula phi
- `E_pow({1,2,..}, k, phi)`: Bounded epistemic operator for agents 1,2,.., bound k and formula phi
- `C_({1,2,..}, phi)`: Common knowledge operator for agents 1,2,.. and formula phi
- `D_({1,2,..}, phi)`: Distributed knowledge operator for agents 1,2,.. and formula phi

public announcement operators:
- `[phi] psi`: Public announcement of phi followed by formula psi
- `<phi> psi`: Possibility of public announcement of phi followed by formula psi

## Dempster-Shafer Table
```bash
uv run dempster_shafer_table.py
```

## Truth Table
```bash
uv run truth_table.py "p | ~p"
uv run truth_table.py "(p -> q) & (q -> r) -> (p -> r)"
```
### Char to logical Tools
- `<->` or `<=>`: BICONDITIONAL (`↔`)
- `->`: IMPLIES (`→`)
- `XOR`: EXCLUSIVE OR (`⊕`)
- `&`: AND (`∧`)
- `|`: OR (`∨`)
- `~` or `!`: NOT (`¬`)
- `(` and `)`: Parentheses for grouping


### Entailment Check
- `false`, `f`, `0`: Contradiction (`⊥`)
- `true`, `t`, `1`: Tautology (`⊤`)
- `A |= B`: Check if formula A entails formula B

```bash
uv run truth_table.py "false |= B"
```

## Syntactic Proof Checker (Epistemic Logic)

A tool for verifying syntactic proofs in the K_n axiom system (and S5).

```bash
uv run python syntactic_proof.py
uv run python syntactic_proof_examples.py  # See all examples
```

### Axiom System

| Axiom | Name | Schema |
|-------|------|--------|
| A1 | Tautology | All propositional tautologies |
| A2 | Distribution/K | `K_i(φ → ψ) → (K_i φ → K_i ψ)` |
| T  | Truth (S5) | `K_i φ → φ` |
| 4  | Positive Introspection (S5) | `K_i φ → K_i K_i φ` |
| 5  | Negative Introspection (S5) | `¬K_i φ → K_i ¬K_i φ` |

(Last three axioms are only for S5, and not really needed for K_n)

**Inference Rules:**
- **MP** (Modus Ponens): From `φ` and `φ → ψ`, infer `ψ`
- **Nec** (Necessitation): From `⊢ φ`, infer `⊢ K_i φ`

**Premises:**
- **Premise**: Any set of formulas assumed to be true without proof.

For "start" of proof where no rule is specified, it is assumed to be a premise. Or if it is labeled "Premise".

### Justification Labels

| Label | Meaning |
|-------|---------|
| `A1`, `Taut` | Propositional tautology |
| `A2`, `K`, `Dist` | Distribution axiom |
| `T`, `Truth` | Truth axiom (S5) |
| `4`, `PI` | Positive introspection (S5) |
| `5`, `NI` | Negative introspection (S5) |
| `MP`, `MP 1,2` | Modus ponens (optionally with line numbers) |
| `Nec 1`, `N 1` | Necessitation from line 1 |
| `Premise` | Given assumption |

### Usage Example
In ```syntactic_proof.py```, you can define a proof and check which of the multiple choice options is valid for a blank line.

```python

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

for i, (formula, valid, explanation) in enumerate(results, 1):
    status = "✓ VALID" if valid else "✗ invalid"
    print(f"{chr(64+i)}) {status} - {explanation}")
```
This will output:
```
A) ✗ invalid - Does not match K_i(φ→ψ) → (K_i φ → K_i ψ)
B) ✓ VALID - Valid distribution axiom instance
C) ✗ invalid - Does not match K_i(φ→ψ) → (K_i φ → K_i ψ)
```
File contains test exam example proof.