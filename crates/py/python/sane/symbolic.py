#########################################################################################
##
##                              SYMBOLIC EXPRESSION ENGINE
##                                   (symbolic.py)
##
##            A small symbolic-math layer over SANE's hash-consed expression
##         DAG. Build expressions from symbols with the ordinary Python
##           operators, differentiate, simplify, evaluate, or compile to a
##                          fast tape -- sympy-like, exact.
##
#########################################################################################

# IMPORTS ===============================================================================

from . import _core

#: A symbolic context (DAG arena + symbol table). Usually created implicitly by
#: :func:`symbols`; construct one explicitly to add symbols incrementally.
Context = _core.Context

#: A handle to a node in a :class:`Context`'s expression DAG. Supports the Python
#: arithmetic operators (``+ - * / **``, unary ``-``) against other ``Expr`` and
#: against Python ints / floats, plus ``.exp() .ln() .sqrt() .sin() .cos()
#: .sinh() .cosh() .tanh() .floor()``, ``.diff(sym)``, ``.simplify()``,
#: ``.eval(**vals)`` / ``.eval_complex(**vals)`` and ``.free_symbols``.
Expr = _core.Expr

#: A compiled flat evaluator (see :func:`compile_tape`).
Tape = _core.Tape


# SYMBOL CONSTRUCTION ===================================================================

def symbols(names):
    """Create symbols in a fresh :class:`Context`.

    Parameters
    ----------
    names : str
        a space-separated list of names, e.g. ``"x y z"``

    Returns
    -------
    Expr | tuple[Expr]
        one :class:`Expr` for a single name, otherwise a tuple

    Example
    -------

    .. code-block:: python

        x, y = sane.symbols("x y")
        f = 2 * x + (y ** 2).exp()
        df = f.diff(x)                 # 2
        f.eval(x=1.0, y=0.5)           # 2 + exp(0.25)

    Notes
    -----
    Symbols from one ``symbols`` call share a context; symbols from different
    calls do not, and mixing them raises. To add a symbol compatible with an
    existing expression, use its context: ``z = f.context.sym("z")``.
    """
    return _core.symbols(names)


# CALCULUS ==============================================================================

def jacobian(residuals, wrt):
    """The symbolic Jacobian ``J[i][j] = d(residuals[i]) / d(wrt[j])``.

    Parameters
    ----------
    residuals : list[Expr]
        the expressions to differentiate
    wrt : list[Expr]
        the symbols to differentiate against (each must be a symbol)

    Returns
    -------
    list[list[Expr]]
        the Jacobian, row-major. All inputs must share one context.
    """
    return _core.jacobian(list(residuals), list(wrt))


def sparsity(jac):
    """The structural sparsity pattern of a symbolic Jacobian.

    Parameters
    ----------
    jac : list[list[Expr]]
        a Jacobian as returned by :func:`jacobian`

    Returns
    -------
    list[list[bool]]
        ``True`` where the entry is not identically zero
    """
    return _core.sparsity([list(row) for row in jac])


# COMPILATION ===========================================================================

def compile_tape(roots, inputs):
    """Compile expressions into a fast :class:`Tape` over named input symbols.

    The reachable sub-DAG is flattened into a straight-line evaluator with a
    reusable work buffer -- the right tool for evaluating the same expressions at
    many input points (sweeps, optimization loops).

    Parameters
    ----------
    roots : list[Expr]
        the expressions to evaluate
    inputs : list[Expr]
        the input symbols, in the order ``Tape.eval`` will expect their values

    Returns
    -------
    Tape
        call ``tape.eval([v0, v1, ...])`` -> ``[root0, root1, ...]``

    Example
    -------

    .. code-block:: python

        x, y = sane.symbols("x y")
        tape = sane.compile_tape([x * y, x + y], [x, y])
        tape.eval([3.0, 4.0])          # [12.0, 7.0]
    """
    return _core.compile(list(roots), list(inputs))
