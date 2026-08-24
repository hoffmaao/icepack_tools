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
      + \alpha\,\underbrace{2H_{\rm ref}\,\frac{A_{\rm reg}}{2}|M_{\rm dev}|^2}_{\text{regulariser}},
    \qquad A_{\rm reg} = A\,\tau_c^{\,n-1}

The linear term is positive-definite in :math:`M` for any :math:`A > 0`,
so it keeps the block non-singular at zero stress; and because it uses
:math:`H_{\rm ref}` rather than :math:`h`, it stays positive-definite as
:math:`h \to 0`.  Stress-matching through :math:`A_{\rm reg} = A\tau_c^{n-1}`
makes the two mechanisms agree at :math:`|M_{\rm dev}| = \tau_c`, so
:math:`\alpha` sets how much regularisation is added relative to the real
rheology at the reference stress.

Separately, and **not** to be confused with :math:`A_{\rm reg}`, the
closures take an optional diffusion-creep prefactor ``A_lin``: a genuine
:math:`n = 1` mechanism acting in parallel with dislocation creep, at the
**layer** thickness,

.. math::
    A|M_{\rm dev}|^{n-1}M_{\rm dev} \;\longrightarrow\;
    \left(A|M_{\rm dev}|^{n-1} + A_{\rm lin}\right) M_{\rm dev}

The two are deliberately distinct.  :math:`A_{\rm reg}` is a numerical
device scaled by a tiny :math:`\alpha` at a constant :math:`H_{\rm ref}`,
whose only job is to keep the membrane block positive-definite as
:math:`h \to 0`.  ``A_lin`` is physics: Goldsby & Kohlstedt's composite
makes Glen's :math:`n = 3` an effective average over several mechanisms,
so ``A_lin`` carries :math:`h` and vanishes with the ice like any real
deformation term.  Both belong; neither replaces the other.

That creep-plus-regulariser potential :math:`P` -- without ``A_lin`` --
is the same composite the ismip7 and peninsula inversions use.  The
residual form below is its ``M``-stationarity together with the
strain-rate coupling, written directly so it can be assembled alongside a
residual-form friction closure, with the optional diffusion term added to
that residual.
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

#: Diffusion-creep (n = 1) prefactor [MPa^-1 yr^-1].  A *physical*
#: mechanism in parallel with dislocation creep, not a numerical device:
#: Goldsby & Kohlstedt's composite makes Glen's n = 3 an effective average
#: of several mechanisms, and the Thwaites multilayer runs carry this same
#: value.  It happens to condition the dual system too -- being n = 1 its
#: Hessian contribution is constant and non-zero at M = 0, which is worth
#: ~3 orders of magnitude over the alpha-weighted regulariser alone.
A_DIFFUSION = 1e-3

#: Floor [m] on the summed layer thickness in the interlayer velocity-jump
#: normalisation.  The closure divides by ``h_above + h_below``, the summed
#: thickness of the two layers meeting at that interface -- ``2H/L`` for
#: ``L`` uniform layers, and so exactly zero wherever the domain has
#: ice-free nodes, e.g. an ocean buffer past a calving front.  Without this
#: the interlayer residual is NaN there and the solve dies at iteration 0
#: with DIVERGED_FUNCTION_NANORINF before any Newton step.  This package
#: is meant to be well posed at zero thickness, so the guard belongs here.
#:
#: The floor engages once that sum drops below it, i.e. below a column
#: thickness of ``L * H_JUMP_FLOOR / 2`` for uniform layers -- 1 m for
#: ``L = 2`` but 5 m for ``L = 10``.  Pick it against the thinnest column
#: whose shear should still be resolved, not against 1 m of ice.
H_JUMP_FLOOR = 1.0

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


def membrane_residual(M, Mt, u, h, A, n, *, A_lin=None, n_val=None,
                      tau_c=TAU_C, alpha=ALPHA, H_ref=H_REF, d=2,
                      h_floor=0.0, extra_linear=None,
                      stress_eps=STRESS_EPS):
    r"""Composite flow law plus strain-rate coupling, in residual form.

    Returns the ``M``-block of the dual residual:

    .. math::
        \int h \left(A |M_{\rm dev}|^{n-1} + A_{\rm lin}\right)
               M_{\rm dev}\!:\!M_t
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
    A_lin : UFL expression, optional
        Diffusion-creep (:math:`n = 1`) prefactor, in parallel with
        dislocation creep and carrying the **layer** thickness, so it
        vanishes with the ice.  ``None`` (the default) leaves diffusion
        creep out entirely.  See :data:`A_DIFFUSION` for a typical value.
        Distinct from the ``alpha``/``H_ref`` regulariser, whose
        prefactor is :math:`A_{\rm reg} = A\tau_c^{n-1}`: that one is a
        numerical device at a *constant* reference thickness.  Note that
        ``A_lin`` is deliberately not scaled by the inverted log-fluidity
        the way ``A`` usually is -- see ``dual_residual``.
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
    if A_lin is not None:
        # Diffusion creep, n = 1, at the LAYER thickness -- a mechanism in
        # parallel with the creep term, so it carries h and vanishes with
        # the ice.  Distinct from the alpha/H_ref regulariser above, which
        # exists only to keep the block positive-definite as h -> 0.
        F += h_v * A_lin * dev * dx
    if extra_linear is not None:
        F += extra_linear * linear * dx
    return F


def interlayer_residual(S, sigma, u_above, u_below, h_above, h_below, A, n,
                        *, A_lin=None, n_val=None, tau_c=TAU_C, alpha=ALPHA,
                        stress_eps=STRESS_EPS, h_jump_floor=H_JUMP_FLOOR):
    r"""Interlayer shear closure with the same linear regularisation.

    The multilayer interlayer stress obeys

    .. math::
        \left(A|S|^{n-1} + \alpha A\tau_c^{\,n-1} + A_{\rm lin}\right) S
        = \frac{u^{l+1} - u^l}{h^{l+1} + h^l}

    whose Jacobian shares the :math:`|S|^{n-1}` degeneracy at ``S = 0``.
    The regulariser is stress-matched the same way.  Unlike the membrane
    term the layer thicknesses enter as a denominator rather than as a
    prefactor, so no reference thickness is needed for coercivity -- but
    that denominator does vanish with the column, and unguarded it makes
    the residual NaN at ice-free nodes.  ``h_jump_floor`` [m] floors
    :math:`h^{l+1} + h^l` in that normalisation and nowhere else; see
    :data:`H_JUMP_FLOOR` for the default and for the column thickness at
    which it engages.

    ``A_lin`` is the optional diffusion-creep (:math:`n = 1`) prefactor,
    the same mechanism ``membrane_residual`` takes and again distinct from
    the stress-matched regulariser :math:`A_{\rm reg} = A\tau_c^{n-1}`.
    ``None`` (the default) leaves it out.
    """
    n_val = float(n) if n_val is None else n_val
    S2 = inner(S, S) + Constant(stress_eps)
    Sn = conditional(eq(n, 1), Constant(1.0), S2 ** ((n - 1) / 2))
    A_reg = A * Constant(tau_c) ** (n_val - 1)
    creep = A * Sn + Constant(alpha) * A_reg
    if A_lin is not None:
        creep = creep + A_lin          # diffusion, n = 1, in parallel
    # Guard the velocity-jump normalisation: h_above + h_below is the summed
    # thickness of the two adjacent layers, so it vanishes with the column
    # and is exactly 0 at ice-free nodes.  There is no ice to shear there,
    # so any finite value works; the floor simply keeps the residual finite.
    du = (u_above - u_below) / max_value(h_above + h_below,
                                         Constant(h_jump_floor))
    return inner(creep * S - du, sigma) * dx
