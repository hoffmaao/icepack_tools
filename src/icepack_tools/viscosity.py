r"""Composite membrane rheology for the icepack2 dual form.

The dual viscous block has the same rank-deficiency problem as the basal
one: the Hessian of :math:`\int 2h\frac{A}{n+1}|M|^{n+1}` scales like
:math:`|M|^{n-1}` and vanishes at :math:`M = 0`, so the first Newton step
of any :math:`n > 1` solve has nothing to descend on.  It also vanishes
wherever the ice thins to zero -- calving fronts, nunataks, an ocean
buffer -- because the whole term carries a factor of :math:`h`.

Both are cured by adding a small linear (:math:`n = 1`) term evaluated at
a **constant** reference thickness:

.. math::
    P = \underbrace{2h\,\frac{A}{n+1}|M_{\rm dev}|^{n+1}}_{\text{creep}}
      + \alpha\,\underbrace{2H_{\rm ref}\,\frac{A_{\rm lin}}{2}|M_{\rm dev}|^2}_{\text{regulariser}},
    \qquad A_{\rm lin} = A\,\tau_c^{\,n-1}

The linear term is positive-definite in :math:`M` for any :math:`A > 0`,
so it keeps the block non-singular at zero stress; and because it uses
:math:`H_{\rm ref}` rather than :math:`h`, it stays positive-definite as
:math:`h \to 0`.  Stress-matching through :math:`A_{\rm lin} = A\tau_c^{n-1}`
makes the two mechanisms agree at :math:`|M_{\rm dev}| = \tau_c`, so
:math:`\alpha` sets how much regularisation is added relative to the real
rheology at the reference stress.

This is the same composite the ismip7 and peninsula inversions use.  The
residual form below is the ``M``-stationarity of that potential together
with the strain-rate coupling, written directly so it can be assembled
alongside a residual-form friction closure.
"""

from firedrake import (
    Constant, conditional, eq, inner, tr, sym, grad, dx, max_value,
)

#: Cross-over stress for stress-matching the linear regulariser [MPa].
TAU_C = 0.1

#: Default weight on the linear regulariser.
ALPHA = 1e-4

#: Reference thickness for the linear regulariser [m].
H_REF = 100.0

#: Floor added inside the stress invariants, in MPa^2, i.e. a ~1 kPa stress
#: floor.  Needed because :math:`(M^2)^{(n-1)/2}` has derivative
#: :math:`\propto (M^2)^{(n-3)/2}`, which for :math:`n < 3` **diverges** at
#: :math:`M = 0` -- so a grain-boundary-sliding layer with :math:`n = 1.8`
#: makes the Jacobian infinite at a cold start, not merely rank-deficient.
#: (For :math:`n \ge 3` the derivative vanishes instead, which the linear
#: regulariser already covers.)  Small against real membrane stresses of
#: 50-200 kPa.
STRESS_EPS = 1e-6


def deviator_inner(M, Mt, d=2):
    r"""Deviatoric inner product :math:`M:M_t - \tfrac{\mathrm{tr}M\,\mathrm{tr}M_t}{d+1}`."""
    return inner(M, Mt) - tr(M) * tr(Mt) / (d + 1)


def second_invariant(M, d=2):
    r""":math:`\tfrac12\left(M:M - \tfrac{(\mathrm{tr}M)^2}{d+1}\right)`."""
    return (inner(M, M) - tr(M) ** 2 / (d + 1)) / 2


def membrane_residual(M, Mt, u, h, A, n, *, n_val=None, tau_c=TAU_C,
                      alpha=ALPHA, H_ref=H_REF, d=2, h_floor=0.0,
                      extra_linear=None, stress_eps=STRESS_EPS):
    r"""Composite flow law plus strain-rate coupling, in residual form.

    Returns the ``M``-block of the dual residual:

    .. math::
        \int h A |M_{\rm dev}|^{n-1} M_{\rm dev}\!:\!M_t
        + \alpha H_{\rm ref} A \tau_c^{\,n-1} M_{\rm dev}\!:\!M_t
        - h\,\varepsilon(u)\!:\!M_t \; dx

    Parameters
    ----------
    M, Mt : UFL expressions
        Membrane stress and its test function.
    u : UFL expression
        Velocity of the same layer.
    h : UFL expression
        Layer thickness.
    A : UFL expression
        Rate factor, e.g. ``A0 * exp(phi)`` with ``phi`` a log control.
    n : Constant
        Flow-law exponent.  Mutable, so it can be ramped 1 -> ``n_val``.
    n_val : float, optional
        The *final* exponent, used to fix the stress-matching power
        :math:`\tau_c^{n-1}` so the regulariser does not move during an
        n-continuation.  Defaults to ``float(n)``.
    h_floor : float
        Thickness floor applied **only** here, to give the velocity some
        coercivity at ice-free nodes.  It must not be applied to the
        driving stress: clamping ``H`` there fabricates a spurious
        :math:`\rho g H_{\rm floor}\nabla s` and blows the buffer velocity
        up.
    extra_linear : UFL expression, optional
        Additional weight on the linear regulariser, e.g. a
        grounding-zone-gated collar ``alpha_gl * (1 - He)`` that damps a
        frictionless shelf without touching the grounded trunk's rheology.
    """
    n_val = float(n) if n_val is None else n_val
    h_v = max_value(h, Constant(h_floor)) if h_floor else h

    M2 = second_invariant(M, d) + Constant(stress_eps)
    Mn = conditional(eq(n, 1), Constant(1.0), M2 ** ((n - 1) / 2))
    dev = deviator_inner(M, Mt, d)

    linear = H_ref * A * Constant(tau_c) ** (n_val - 1) * dev
    F = (h_v * A * Mn * dev + Constant(alpha) * linear
         - h_v * inner(sym(grad(u)), Mt)) * dx
    if extra_linear is not None:
        F += extra_linear * linear * dx
    return F


def interlayer_residual(S, sigma, u_above, u_below, h_above, h_below, A, n,
                        *, n_val=None, tau_c=TAU_C, alpha=ALPHA,
                        stress_eps=STRESS_EPS):
    r"""Interlayer shear closure with the same linear regularisation.

    The multilayer interlayer stress obeys

    .. math::  A|S|^{n-1}S = \frac{u^{l+1} - u^l}{h^{l+1} + h^l}

    whose Jacobian shares the :math:`|S|^{n-1}` degeneracy at ``S = 0``.
    The regulariser is stress-matched the same way.  Unlike the membrane
    term this one needs no reference thickness: the layer thicknesses
    appear in the velocity-jump normalisation, not as a prefactor that can
    vanish.
    """
    n_val = float(n) if n_val is None else n_val
    S2 = inner(S, S) + Constant(stress_eps)
    Sn = conditional(eq(n, 1), Constant(1.0), S2 ** ((n - 1) / 2))
    A_lin = A * Constant(tau_c) ** (n_val - 1)
    du = (u_above - u_below) / (h_above + h_below)
    return inner((A * Sn + Constant(alpha) * A_lin) * S - du, sigma) * dx
