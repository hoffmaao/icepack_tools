r"""Geometry helpers that are safe to use inside a momentum residual.

Ported from ``ismip7/icepack2_tools/geometry.py``.
"""

import weakref

from firedrake import (
    Function, FunctionSpace, TestFunction, assemble, dx, grad,
)

_LUMPED_MASS = weakref.WeakKeyDictionary()


def _lumped_mass(mesh, Q_cg):
    """Lumped CG1 mass vector, assembled once per mesh.

    Keyed on the *mesh*, which outlives the run: the CG1 space
    :func:`cg1_lift` builds is transient, so keying on that would evict the
    entry on every call.  The cached value is the plain array rather than
    the assembled Cofunction, which would hold a strong reference back to
    its own key and pin the mesh forever.
    """
    cached = _LUMPED_MASS.get(mesh)
    if cached is None:
        cached = assemble(TestFunction(Q_cg) * dx).dat.data_ro.copy()
        _LUMPED_MASS[mesh] = cached
    return cached


def cg1_lift(f):
    r"""Lumped-mass CG1 reconstruction of a cell-wise (DG0) field.

    Each CG1 node takes the area-weighted mean of the adjacent cells.
    Volume-preserving and a convex combination, so it cannot overshoot --
    unlike an L2 projection, which can (projecting a DG0 fluidity to CG1
    has produced negative fluidities against a strictly positive DG0
    range).

    It is **not** unbiased on the domain boundary, where the stencil is
    one-sided and pulls boundary nodes toward interior values.  On an
    unbuffered mesh that boundary is the calving front, where the terminus
    traction goes as :math:`h^2`, so lifting the thickness there inflates
    the front and multiplies the outflux.  Keep it out of momentum
    residuals and out of every flux: it is for diagnostics, fixed
    reference scalings and parameterisations.
    """
    mesh = f.function_space().mesh()
    Q_cg = FunctionSpace(mesh, "CG", 1)
    lumped = _lumped_mass(mesh, Q_cg)
    rhs = assemble(TestFunction(Q_cg) * f * dx)
    out = Function(Q_cg)
    out.dat.data[:] = rhs.dat.data_ro / lumped
    return out


def surface_slope(s):
    r"""``grad(s)`` valid for a CG1 *or* DG0 surface.

    A DG0 surface has an identically zero cell gradient -- UFL folds it
    away -- because its slope lives entirely in the inter-cell jumps,
    which a momentum balance picks up weakly through its ``jump(s, nu)``
    facet term.  Callers that need a *pointwise* slope (a friction anchor,
    a melt parameterisation) get one from a CG1 reconstruction instead.
    """
    if s.function_space().ufl_element().degree() == 0:
        return grad(cg1_lift(s))
    return grad(s)
