#!/usr/bin/env python3
from itertools import product

# ---------------- Tokeniser ----------------

def tokenize(s: str):
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue

        # Parentheses
        if c in '()':
            tokens.append(c)
            i += 1
            continue

        # Multi-character operators first
        if s.startswith('<->', i) or s.startswith('<=>', i):
            tokens.append('IFF')
            i += 3
            continue
        if s.startswith('->', i):
            tokens.append('IMP')
            i += 2
            continue
        if s.startswith('XOR', i):
            tokens.append('XOR')
            i += 3
            continue

        # Single-character operators
        if c in ['&', '∧', '^']:
            tokens.append('AND')
            i += 1
            continue
        if c in ['|', '∨']:
            tokens.append('OR')
            i += 1
            continue
        if c in ['⊕']:
            tokens.append('XOR')
            i += 1
            continue
        if c in ['!', '~', '¬']:
            tokens.append('NOT')
            i += 1
            continue

        # Identifiers (proposition letters like p, q, p1, p2, ...)
        if c.isalnum() or c == '_':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                j += 1
            tokens.append(s[i:j])
            i = j
            continue

        raise ValueError(f"Unexpected character: {c!r}")

    return tokens


# ---------------- Parser: Shunting-yard to RPN ----------------

PRECEDENCE = {
    'NOT': 5,
    'AND': 4,
    'XOR': 3,
    'OR':  2,
    'IMP': 1,
    'IFF': 0,
}

def to_rpn(tokens):
    output = []
    stack = []
    for tok in tokens:
        if tok == '(':
            stack.append(tok)
        elif tok == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if not stack:
                raise ValueError("Mismatched parentheses")
            stack.pop()  # remove '('
        elif tok in PRECEDENCE:
            while (stack and stack[-1] in PRECEDENCE and
                   PRECEDENCE[stack[-1]] > PRECEDENCE[tok]):
                output.append(stack.pop())
            stack.append(tok)
        else:
            # variable
            output.append(tok)

    while stack:
        if stack[-1] in ('(', ')'):
            raise ValueError("Mismatched parentheses")
        output.append(stack.pop())
    return output


# ---------------- Evaluator ----------------

def eval_rpn(rpn, valuation):
    stack = []
    for tok in rpn:
        if tok in PRECEDENCE:
            if tok == 'NOT':
                a = stack.pop()
                stack.append(not a)
            else:
                b = stack.pop()
                a = stack.pop()
                if tok == 'AND':
                    stack.append(a and b)
                elif tok == 'OR':
                    stack.append(a or b)
                elif tok == 'XOR':
                    stack.append(a != b)
                elif tok == 'IMP':
                    stack.append((not a) or b)
                elif tok == 'IFF':
                    stack.append(a == b)
        else:
            # variable
            stack.append(valuation[tok])

    if len(stack) != 1:
        raise ValueError("Invalid formula (stack not singleton at end)")
    return stack[0]


# ---------------- Truth table + classification ----------------

def truth_table(expr: str):
    tokens = tokenize(expr)
    rpn = to_rpn(tokens)

    # Propositional variables = tokens that are not operators or parentheses
    vars_ = sorted(
        {t for t in tokens if t not in PRECEDENCE and t not in ('(', ')')}
    )

    # Convert to "normal math" (Unicode) expression
    math_map = {
        'AND': '∧',
        'OR':  '∨',
        'NOT': '¬',
        'IMP': '→',
        'IFF': '↔',
        'XOR': '⊕',
    }
    math_tokens = [math_map[t] if t in math_map else t for t in tokens]
    # Join with spaces, but maybe we can be smarter about spacing later.
    # For now, simple join is fine.
    math_expr = " ".join(math_tokens)

    print("Formula:", math_expr)
    print("Variables:", ", ".join(vars_))
    print()

    header = " ".join(vars_) + " | " + math_expr
    print(header)
    print("-" * len(header))

    results = []
    for values in product([False, True], repeat=len(vars_)):
        valuation = dict(zip(vars_, values))
        res = eval_rpn(rpn, valuation)
        results.append(res)
        vals_str = " ".join('1' if valuation[v] else '0' for v in vars_)
        print(f"{vals_str} | {'1' if res else '0'}")

    print()
    if all(results):
        print("Result: VALID (TAUTOLOGY) (true in all valuations).")
    elif any(results):
        print("Result: SATISFIABLE but not a tautology.")
    else:
        print("Result: UNSATISFIABLE (contradiction).")


# ---------------- Entailment Check ----------------

def check_entailment(lhs_str, rhs_str):
    # Split lhs by comma to support multiple premises
    if lhs_str.strip():
        premise_strs = [s.strip() for s in lhs_str.split(',')]
    else:
        premise_strs = []
    
    conclusion_str = rhs_str.strip()
    
    # Parse all formulas
    premises_rpn = []
    all_vars = set()
    
    try:
        for p_str in premise_strs:
            tokens = tokenize(p_str)
            rpn = to_rpn(tokens)
            premises_rpn.append(rpn)
            # Extract vars
            vars_ = {t for t in tokens if t not in PRECEDENCE and t not in ('(', ')')}
            all_vars.update(vars_)
            
        # Parse conclusion
        tokens_c = tokenize(conclusion_str)
        rpn_c = to_rpn(tokens_c)
        vars_c = {t for t in tokens_c if t not in PRECEDENCE and t not in ('(', ')')}
        all_vars.update(vars_c)
    except Exception as e:
        print(f"Error parsing formulas: {e}")
        return

    sorted_vars = sorted(all_vars)
    
    print(f"Checking entailment: {{{', '.join(premise_strs)}}} |= {conclusion_str}")
    print(f"Variables: {', '.join(sorted_vars)}")
    print()
    
    # Header
    header_vars = " ".join(sorted_vars)
    # We'll show a simplified table: Vars | Premises | Conclusion | Valid?
    header = f"{header_vars} | Premises | Conclusion | Valid?"
    print(header)
    print("-" * len(header))
    
    is_valid = True
    counterexamples = []
    
    for values in product([False, True], repeat=len(sorted_vars)):
        valuation = dict(zip(sorted_vars, values))
        
        try:
            # Eval premises
            premise_vals = [eval_rpn(rpn, valuation) for rpn in premises_rpn]
            all_premises_true = all(premise_vals)
            
            # Eval conclusion
            conclusion_val = eval_rpn(rpn_c, valuation)
        except Exception as e:
            print(f"Error evaluating for valuation {valuation}: {e}")
            return

        # Check entailment
        # If premises are true, conclusion must be true.
        row_valid = True
        if all_premises_true and not conclusion_val:
            row_valid = False
            is_valid = False
            counterexamples.append(valuation)
        
        # Output row
        vals_str = " ".join('1' if valuation[v] else '0' for v in sorted_vars)
        prem_str = "1" if all_premises_true else "0"
        conc_str = "1" if conclusion_val else "0"
        status = "OK" if row_valid else "FAIL"
        
        # Formatting alignment
        # Vars section width is len(header_vars)
        # Premises column is 8 chars wide in header "Premises"
        # Conclusion column is 10 chars wide "Conclusion"
        # Valid? is 6 chars
        
        # Let's just use simple tabbing or fixed width
        print(f"{vals_str} | {prem_str:^8} | {conc_str:^10} | {status}")

    print()
    if is_valid:
        print("Result: Entailment HOLDS.")
    else:
        print("Result: Entailment DOES NOT HOLD.")
        # print("Counterexamples (Premises=True, Conclusion=False):")
        # for ce in counterexamples:
        #     print(ce)


# ---------------- Main ----------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        expr = " ".join(sys.argv[1:])
    else:
        expr = input("Enter a propositional formula (or entailment A |= B): ")

    if '|=' in expr:
        lhs, rhs = expr.split('|=', 1)
        check_entailment(lhs, rhs)
    else:
        truth_table(expr)
