from dataclasses import dataclass
from typing import Set, Dict, FrozenSet, Iterable

# ---------- Formula AST ----------

@dataclass(frozen=True)
class Formula:
    # allow ~phi, phi & psi, phi | psi, phi >> psi (implication)
    def __invert__(self):        # ~phi
        return Not(self)

    def __and__(self, other):    # phi & psi
        return And(self, other)

    def __or__(self, other):     # phi | psi
        return Or(self, other)

    def __rshift__(self, other): # phi >> psi  (phi -> psi)
        return Imp(self, other)


@dataclass(frozen=True)
class Prop(Formula):
    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Not(Formula):
    phi: Formula

    def __str__(self):
        return f"¬{self.phi}"


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def __str__(self):
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def __str__(self):
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Imp(Formula):
    left: Formula
    right: Formula

    def __str__(self):
        return f"({self.left} → {self.right})"


@dataclass(frozen=True)
class K(Formula):
    agent: int
    phi: Formula

    def __str__(self):
        return f"K_{self.agent} {self.phi}"


@dataclass(frozen=True)
class E(Formula):
    group: FrozenSet[int]
    phi: Formula

    def __str__(self):
        return f"E_{set(self.group)} {self.phi}"


@dataclass(frozen=True)
class D(Formula):
    group: FrozenSet[int]
    phi: Formula

    def __str__(self):
        return f"D_{set(self.group)} {self.phi}"


@dataclass(frozen=True)
class C(Formula):
    group: FrozenSet[int]
    phi: Formula

    def __str__(self):
        return f"C_{set(self.group)} {self.phi}"


# Small helpers so formulas look nicer
def P(name: str) -> Prop:
    return Prop(name)


def K_(agent: int, phi: Formula) -> K:
    return K(agent, phi)

def E_(group: Iterable[int], phi: Formula) -> E:
    return E(frozenset(group), phi)

def E_pow(group, k: int, phi: Formula) -> Formula:
    """
    Build the formula E_G^k phi using the inductive definition:
    E_G^0 phi = phi
    E_G^{k+1} phi = E_G(E_G^k phi)
    """
    result = phi
    for _ in range(k):
        result = E_(group, result)
    return result

def C_(group: Iterable[int], phi: Formula) -> C:
    return C(frozenset(group), phi)

def D_(group: Iterable[int], phi: Formula) -> D:
    return D(frozenset(group), phi)


# ---------- Kripke model ----------

@dataclass
class KripkeModel:
    worlds: Set[str]
    valuation: Dict[str, Set[str]]              # world -> set of atoms that are true
    relations: Dict[int, Dict[str, Set[str]]]   # agent -> world -> accessible worlds

    def __str__(self):
        lines = ["Kripke Model:"]
        lines.append(f"  Worlds: {sorted(list(self.worlds))}")
        lines.append("  Valuation:")
        for w in sorted(self.valuation.keys()):
            val = sorted(list(self.valuation[w]))
            lines.append(f"    {w}: {val}")
        lines.append("  Relations:")
        for agent, rels in self.relations.items():
            lines.append(f"    Agent {agent}:")
            for w in sorted(rels.keys()):
                acc = sorted(list(rels[w]))
                lines.append(f"      {w} -> {acc}")
        return "\n".join(lines)


# ---------- Semantics ----------

def eval_formula(model: KripkeModel, w: str, phi: Formula) -> bool:
    """Return whether phi is true at world w in model."""
    if isinstance(phi, Prop):
        return phi.name in model.valuation[w]

    if isinstance(phi, Not):
        return not eval_formula(model, w, phi.phi)

    if isinstance(phi, And):
        return eval_formula(model, w, phi.left) and eval_formula(model, w, phi.right)

    if isinstance(phi, Or):
        return eval_formula(model, w, phi.left) or eval_formula(model, w, phi.right)

    if isinstance(phi, Imp):
        return (not eval_formula(model, w, phi.left)) or eval_formula(model, w, phi.right)

    if isinstance(phi, K):
        # K_i phi: phi holds in all worlds accessible for agent i
        return all(eval_formula(model, v, phi.phi)
                   for v in model.relations[phi.agent][w])

    if isinstance(phi, E):
        # everybody knows: conjunction of K_i over the group
        return all(eval_formula(model, w, K(agent, phi.phi))
                   for agent in phi.group)

    if isinstance(phi, C):
        # Build union relation R_E for the group
        union_rel = {
            world: set().union(
                *[model.relations[agent][world] for agent in phi.group]
            )
            for world in model.worlds
        }

        # BFS/DFS from w along R_E
        visited = set()
        stack = [w]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            # φ must hold in every reachable world
            if not eval_formula(model, current, phi.phi):
                return False

            # add successors
            for nxt in union_rel[current]:
                if nxt not in visited:
                    stack.append(nxt)

        return True

    if isinstance(phi, D):
        # distributed knowledge: intersection of accessibility relations
        accessible_sets = [model.relations[agent][w] for agent in phi.group]
        if not accessible_sets:
            # group shouldn't be empty; if it is, just return phi itself
            return eval_formula(model, w, phi.phi)
        intersection = set(accessible_sets[0])
        for s in accessible_sets[1:]:
            intersection &= s
        return all(eval_formula(model, v, phi.phi) for v in intersection)

    raise TypeError(f"Unknown formula type: {type(phi)}")


def truth_in_all_worlds(model: KripkeModel, phi: Formula):
    """Return a dict world -> truth value."""
    return {w: eval_formula(model, w, phi) for w in sorted(model.worlds)}


def is_valid_in_model(model: KripkeModel, phi: Formula) -> bool:
    """True iff phi is true in all worlds of the model."""
    return all(truth_in_all_worlds(model, phi).values())

# ---------- Example: your model ----------

if __name__ == "__main__":
    # worlds
    worlds = {"t", "s", "u"}

    # atomic valuation: which props are true in each world
    valuation = {
        "t": {"q"},        # t: ¬p, q
        "s": {"p", "q"},   # s: p, q
        "u": {"p"},        # u: p, ¬q
    }

    # accessibility relations for agents 1 and 2
    relations = {
        1: {
            "t": {"t"},
            "s": {"s", "u"},
            "u": {"s", "u"},
        },
        2: {
            "t": {"t", "s"},
            "s": {"t", "s"},
            "u": {"u"},
        },
    }

    model = KripkeModel(worlds, valuation, relations)

    print(model)
    print()

    # shorthand for atoms
    p, q = P("p"), P("q")

    # Your four formulas:
    phi1 = K_(2, K_(1, p | q))
    phi2 = E_({1, 2}, p & D_({1, 2}, p))
    phi3 = K_(1, K_(2, p & q) | K_(2, ~(p & q)))
    phi4 = (p & ~p) >> q
    phi5 = C_({1,2}, p & q)

    formulas = [phi1, phi2, phi3, phi4, phi5]

    for i, phi in enumerate(formulas, start=1):
        truth = truth_in_all_worlds(model, phi)
        #print(f"Formula {i}: truth values {truth}, valid = {is_valid_in_model(model, phi)}")
        print(f"Formula {i}")
        print(f"  formula: {phi}")
        print(f"  Truth worlds: {truth}")
        print(f"  Valid in model: {is_valid_in_model(model, phi)}")
        print()
