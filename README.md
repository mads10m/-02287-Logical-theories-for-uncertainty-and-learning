

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

- `K_(n, phi)`: Knowledge operator for agent n and formula phi
- `E_({1,2,..}, phi)`: Epistemic operator for agents 1,2,.. and formula phi
- `E_pow({1,2,..}, k, phi)`: Bounded epistemic operator for agents 1,2,.., bound k and formula phi
- `C_({1,2,..}, phi)`: Common knowledge operator for agents 1,2,.. and formula phi
- `D_({1,2,..}, phi)`: Distributed knowledge operator for agents 1,2,.. and formula phi

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