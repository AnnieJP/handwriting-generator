"""
LaTeX to symbolic text conversion.

Converts LaTeX mathematical expressions into a plain text / Unicode
representation that can be fed into the handwriting generator as a
character sequence.

Two modes are supported:
  1. latex_to_text(expr)    – returns a Unicode string approximation
  2. latex_to_sympy(expr)   – returns a SymPy expression (for validation)

The conversion covers the most common LaTeX constructs found in
undergraduate-level math: fractions, exponents, subscripts, Greek
letters, operators, integrals, sums, etc.
"""

import re
from typing import Optional, Tuple

try:
    import sympy
    from sympy.parsing.latex import parse_latex as sympy_parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Greek letter mappings
# ---------------------------------------------------------------------------
GREEK = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\vartheta": "θ", r"\iota": "ι", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ",
    r"\pi": "π", r"\varpi": "π", r"\rho": "ρ", r"\varrho": "ρ",
    r"\sigma": "σ", r"\varsigma": "ς", r"\tau": "τ", r"\upsilon": "υ",
    r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ", r"\psi": "ψ",
    r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Upsilon": "Υ",
    r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
}

# ---------------------------------------------------------------------------
# Operator / symbol mappings
# ---------------------------------------------------------------------------
OPERATORS = {
    r"\cdot": "·", r"\times": "×", r"\div": "÷", r"\pm": "±",
    r"\mp": "∓", r"\leq": "≤", r"\geq": "≥", r"\neq": "≠",
    r"\approx": "≈", r"\equiv": "≡", r"\sim": "~", r"\propto": "∝",
    r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇",
    r"\in": "∈", r"\notin": "∉", r"\subset": "⊂", r"\supset": "⊃",
    r"\cup": "∪", r"\cap": "∩", r"\emptyset": "∅",
    r"\forall": "∀", r"\exists": "∃", r"\neg": "¬",
    r"\land": "∧", r"\lor": "∨", r"\oplus": "⊕", r"\otimes": "⊗",
    r"\to": "→", r"\leftarrow": "←", r"\Rightarrow": "⇒",
    r"\Leftrightarrow": "⟺", r"\mapsto": "↦",
    r"\sqrt": "√",
    r"\int": "∫", r"\iint": "∬", r"\iiint": "∭", r"\oint": "∮",
    r"\sum": "Σ", r"\prod": "Π",
    r"\lim": "lim", r"\max": "max", r"\min": "min",
    r"\log": "log", r"\ln": "ln", r"\exp": "exp",
    r"\sin": "sin", r"\cos": "cos", r"\tan": "tan",
    r"\arcsin": "arcsin", r"\arccos": "arccos", r"\arctan": "arctan",
    r"\sinh": "sinh", r"\cosh": "cosh", r"\tanh": "tanh",
    r"\det": "det", r"\tr": "tr", r"\mathrm{T}": "T",
    r"\ldots": "...", r"\cdots": "···", r"\vdots": "⋮", r"\ddots": "⋱",
    r"\langle": "⟨", r"\rangle": "⟩",
    r"\|": "‖", r"\left": "", r"\right": "",
    r"\,": " ", r"\;": " ", r"\:": " ", r"\ ": " ", r"\!": "",
    r"\quad": "  ", r"\qquad": "   ",
}

# ---------------------------------------------------------------------------
# Superscript / subscript digit maps (for compact notation)
# ---------------------------------------------------------------------------
SUPERSCRIPT_MAP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
SUBSCRIPT_MAP = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")


def _apply_superscript(text: str) -> str:
    if len(text) == 1:
        return text.translate(SUPERSCRIPT_MAP)
    return f"^({text})"


def _apply_subscript(text: str) -> str:
    if len(text) == 1:
        return text.translate(SUBSCRIPT_MAP)
    return f"_({text})"


def _strip_braces(s: str) -> str:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s[1:-1]
    return s


def _extract_braced(s: str, start: int) -> Tuple[str, int]:
    """
    Extract the content of the {...} block starting at position `start`.
    Returns (content, end_position).
    """
    if start >= len(s) or s[start] != "{":
        if start < len(s):
            return s[start], start + 1
        return "", start
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i + 1
    return s[start + 1 :], len(s)


def latex_to_text(expr: str) -> str:
    """
    Convert a LaTeX expression to a plain-text Unicode approximation.

    Args:
        expr: A LaTeX math expression (with or without $ delimiters).

    Returns:
        A Unicode string suitable as input to the handwriting generator.
    """
    expr = expr.strip()
    expr = re.sub(r"^\$+|^\\\[|\\\]$|\$+$", "", expr).strip()
    expr = re.sub(r"^\\begin\{[^}]*\}|\\end\{[^}]*\}$", "", expr).strip()

    for cmd, sym in sorted(GREEK.items(), key=lambda x: -len(x[0])):
        expr = expr.replace(cmd, sym)
    for cmd, sym in sorted(OPERATORS.items(), key=lambda x: -len(x[0])):
        expr = expr.replace(cmd, sym)

    expr = _process_fracs(expr)
    expr = _process_scripts(expr)
    expr = _process_sqrt(expr)

    expr = re.sub(r"\\[a-zA-Z]+\*?", "", expr)
    expr = re.sub(r"[{}]", "", expr)
    expr = re.sub(r"\s+", " ", expr).strip()

    return expr


def _process_fracs(expr: str) -> str:
    """Replace \frac{a}{b} with (a/b)."""
    pattern = re.compile(r"\\frac")
    result = []
    i = 0
    while i < len(expr):
        m = pattern.search(expr, i)
        if not m:
            result.append(expr[i:])
            break
        result.append(expr[i : m.start()])
        pos = m.end()
        num, pos = _extract_braced(expr, pos)
        den, pos = _extract_braced(expr, pos)
        num_text = latex_to_text(num)
        den_text = latex_to_text(den)
        result.append(f"({num_text}/{den_text})")
        i = pos
    return "".join(result)


def _process_sqrt(expr: str) -> str:
    """Replace √{x} with √(x)."""
    def _repl(m):
        inner = m.group(1)
        return f"√({latex_to_text(inner)})"
    expr = re.sub(r"√\{([^}]*)\}", _repl, expr)
    return expr


def _process_scripts(expr: str) -> str:
    """
    Handle ^ and _ for superscripts/subscripts.
    Single char or {group} following ^ or _ is converted.
    """
    result = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c in ("^", "_"):
            fn = _apply_superscript if c == "^" else _apply_subscript
            i += 1
            if i < len(expr) and expr[i] == "{":
                content, i = _extract_braced(expr, i)
                content = latex_to_text(content)
                result.append(fn(content))
            elif i < len(expr):
                result.append(fn(expr[i]))
                i += 1
        else:
            result.append(c)
            i += 1
    return "".join(result)


def latex_to_sympy(expr: str):
    """
    Parse a LaTeX expression with SymPy (requires sympy[latex]).
    Returns a SymPy expression or None on failure.
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("sympy is required: pip install sympy")
    expr = expr.strip().strip("$")
    try:
        return sympy_parse_latex(expr)
    except Exception:
        return None


def preprocess_input(text: str) -> str:
    """
    Preprocess mixed printed + LaTeX input.

    Inline LaTeX is delimited by $...$ or \\(...\\).
    Display LaTeX is delimited by $$...$$ or \\[...\\].

    Returns the text with all LaTeX replaced by Unicode equivalents.
    """
    def _replace_math(m: re.Match) -> str:
        return latex_to_text(m.group(1))

    text = re.sub(r"\$\$(.+?)\$\$", _replace_math, text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.+?)\\\]", _replace_math, text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", _replace_math, text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", _replace_math, text, flags=re.DOTALL)
    return text


if __name__ == "__main__":
    tests = [
        r"\frac{d}{dx} e^x = e^x",
        r"\sum_{i=0}^{n} x_i",
        r"\alpha + \beta = \gamma",
        r"f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}",
        r"\int_0^\infty e^{-\lambda t} dt = \frac{1}{\lambda}",
    ]
    for t in tests:
        print(f"LaTeX : {t}")
        print(f"Text  : {latex_to_text(t)}")
        print()
