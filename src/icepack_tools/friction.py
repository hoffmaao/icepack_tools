r"""Basal friction laws for the icepack2 dual form.

Three laws are available -- Weertman, Budd and regularised Coulomb (see
:data:`LAWS`) -- and all three are written as a variational **residual
that closes the basal stress directly**, not as a dissipation potential.
That property is shared, not incidental to any one law, so it is worth
stating plainly why.

The dual form carries the basal stress :math:`\tau` as an unknown.  The
usual ``icepack2.model.minimization.friction_power`` closes it through a
power-law potential, whose Hessian scales like :math:`|\tau|^{m-1}` and is
therefore **rank-deficient at** :math:`\tau = 0` for :math:`m > 1` -- a
cold start has no descent direction.  A regularised-Coulomb *potential*
is worse still: writing it as a complementary energy gives

.. math::  P = \int \frac{u_0\beta^2}{m+1}\bigl[-\ln(1 - r^{m+1})\bigr]\,dx,
           \qquad r = |\tau|/\beta^2,

which is singular at the Coulomb limit :math:`r \to 1`.  For a fast
tidewater glacier that limit is not a corner case but the operating
point: with :math:`u_0 = 300` m/yr, a trunk moving 2000 m/yr sits at
:math:`r \approx 0.97`, and Newton cannot reach it from :math:`\tau = 0`.

Closing :math:`\tau = \tau_b(u)` as a residual instead makes the
basal-stress block of the dual system the **identity** -- perfectly
conditioned at :math:`\tau = 0` -- and needs no continuation in the
sliding exponent, whichever law supplies :math:`\tau_b`.  For the
regularised-Coulomb law the Coulomb limit is reached through a harmonic
blend that is smooth everywhere:

.. math::
    \tau_W   &= C_{w0}\,e^{\theta H_e}\,|u|_{\rm reg}^{1/m}  \\
    \tau_{\rm cap} &= c_0 N                                   \\
    \tau_b   &= \frac{\tau_W\,\tau_{\rm cap}}{\tau_W + \tau_{\rm cap}}

which is Weertman at low speed, the Coulomb cap at high speed, and
**exactly zero** where the ice floats, because :math:`N \to 0` there.
Budd caps the same :math:`\tau_W` by a normalised effective pressure and
likewise vanishes afloat, there gated by the grounded indicator rather
than by the sign of :math:`N`; plain Weertman has no :math:`N` factor at
all, so it keeps drag on floating ice and must be masked if that matters.

Ported and generalised from ``ismip7/icepack2_tools/dual_friction.py``
(itself derived from ``gia-icepack/scripts/ase_model.py:build_F_rc``).
"""

import ufl
from firedrake import (
    Constant, Function, FunctionSpace, conditional, dx, exp, gt, inner,
    max_value, min_value, sqrt,
)

from .constants import ice_density, gravity
from .geometry import cg1_lift, surface_slope
from .grounding import effective_pressure, grounded_mask, GL_WIDTH

#: Coulomb-cap coefficient, :math:`\tau_{\rm cap} = c_0 N` (Tsai, Schoof).
C0 = 0.5

#: Velocity regularisation [m/yr], keeps the stress finite at ``u = 0``.
U_MIN = 1.0


def weertman_anchor(H, s, u_obs, m_slide, Q, rho_I=ice_density, g=gravity,
                    u_floor=1.0, name="C_w0"):
    r"""Driving-stress-consistent Weertman coefficient.

    .. math::  C_{w0} = \tau_d\,/\,|u_{\rm obs}|^{1/m},
               \qquad \tau_d = \rho_I g H |\nabla s|

    Fixing this as an anchor is what makes ``theta = 0`` a *balanced*
    starting control rather than an arbitrary one: at :math:`u = u_{\rm
    obs}` and :math:`\theta = 0` the Weertman branch returns exactly
    :math:`\tau_d`.  The inverted ``theta`` is then an O(1) logarithmic
    adjustment instead of having to carry the whole friction magnitude,
    which is both better conditioned and easier to regularise.

    ``s`` may be CG1 or DG0; :func:`~icepack_tools.geometry.surface_slope`
    handles both.

    The driving stress is reduced to DG0 and lifted before it reaches a
    continuous ``Q``, and that step is load-bearing, not tidiness.
    :math:`\nabla s` is **discontinuous** -- cell-wise constant for a CG1
    surface -- and interpolating a discontinuous expression into a
    continuous space is not well defined at a shared node: Firedrake
    evaluates cell by cell and the node keeps whichever cell wrote last,
    so the value depends on cell numbering, hence on the mesh partition,
    hence on the MPI rank count.  Straight interpolation gave 77 % of the
    nodes of a Store mesh a different anchor on 4 ranks than on 1, by up
    to a factor of 20 at a node, which moved the converged velocity by
    26 % on 500 m thick ice.  A friction anchor that changes with the rank
    count makes every result downstream of it unreproducible.

    :func:`~icepack_tools.geometry.cg1_lift` instead gives each node the
    area-weighted mean of its adjacent cells: well defined, independent of
    numbering, and a convex combination so it cannot overshoot.  It is
    biased at the domain boundary, which is acceptable here because this
    is a *fixed reference scaling* rather than a flux -- the caveat
    ``cg1_lift`` documents -- and ``theta`` absorbs any offset it leaves.

    Only :math:`\tau_d` goes through DG0; :math:`|u_{\rm obs}|` is already
    continuous and stays at its nodal values.
    """
    grad_s = surface_slope(s)
    tau_d = rho_I * g * H * sqrt(inner(grad_s, grad_s) + Constant(1e-12))
    if Q.ufl_element().sobolev_space == ufl.H1:
        mesh = Q.mesh()
        DG = FunctionSpace(mesh, "DG", 0)
        tau_d = cg1_lift(Function(DG).interpolate(tau_d))
    speed = max_value(sqrt(inner(u_obs, u_obs)), Constant(u_floor))
    return Function(Q, name=name).interpolate(tau_d / speed ** (1.0 / m_slide))


#: Friction laws this module can close.  All three return a basal-stress
#: *magnitude*, closed by :func:`friction_residual` into an identity
#: tau-block, so none of them needs a continuation in the sliding
#: exponent.  Names match ismip7's ``fric_law``.
LAWS = ("regularized_coulomb", "budd", "weertman")


def basal_stress(u, C_w0, theta, H, s, b, m_slide, *, law="regularized_coulomb",
                 c0=C0, u_min=U_MIN, eps_tauc=0.0, He=None, gl_width=GL_WIDTH,
                 c_w0_floor=0.0, N_ref=None, nhat_floor=0.0, nhat_cap=3.0):
    r"""Basal stress magnitude :math:`\tau_b` (scalar, MPa) for one of :data:`LAWS`.

    All three share the Weertman branch

    .. math::  \tau_W = C_{w0}\,e^{\theta H_e}\,|u|_{\rm reg}^{1/m}

    and differ only in how the bed's strength is capped:

    ``weertman``
        :math:`\tau_b = \tau_W`.  No cap, so drag does **not** vanish on
        floating ice by itself -- pair it with a mask if that matters.
    ``budd``
        :math:`\tau_b = \tau_W H_e \hat N`, with
        :math:`\hat N = N/N_{\rm ref}` the *normalised* effective pressure.
        With ``N_ref=None`` (the inversion geometry) :math:`\hat N = 1` on
        grounded ice, so the inferred friction is preserved and the
        effective-pressure feedback is a *relative* change as the geometry
        evolves.  The grounded indicator :math:`H_e` -- not the sign of
        :math:`N` -- is what holds :math:`\tau_b` at zero afloat.
    ``regularized_coulomb``
        :math:`\tau_b = \tau_W\tau_{\rm cap}/(\tau_W + \tau_{\rm cap})`
        with :math:`\tau_{\rm cap} = c_0 N`.

    ``budd`` and ``regularized_coulomb`` both give zero drag to machine
    precision where the ice floats: ``budd`` because :math:`H_e` vanishes
    there, ``regularized_coulomb`` because :math:`N` enters as a *factor*,
    so the roundoff residue of :math:`p_I - p_W` passes straight through
    rather than being amplified.

    Parameters
    ----------
    u : UFL expression
        Sliding velocity (the basal layer's velocity in a multilayer model).
    C_w0 : Function
        Weertman anchor from :func:`weertman_anchor`.
    theta : Function
        Log friction control.  Gated by the grounded indicator, so
        ``dJ/dtheta`` is identically zero on floating ice.
    H, s, b : Function
        Thickness, surface and bed.
    m_slide : float or Constant
        Sliding exponent.  **Fixed** -- the residual closure needs no
        continuation in ``m``.
    c0 : float
        Coulomb-cap coefficient.
    u_min : float
        Velocity regularisation [m/yr].
    eps_tauc : float
        Yield floor [MPa].  Leave at 0 for bit-exact zero drag where ``N = 0``.
    He : UFL expression, optional
        Precomputed grounded indicator; built from ``H, b`` if omitted.
    c_w0_floor : float
        Coercivity floor on ``C_w0``, needed only where the mesh has
        ice-free nodes (``H = 0``) at which both ``C_w0`` and the viscous
        coupling ``H M`` vanish and the velocity loses all coercivity.  It
        is absorbed by ``theta`` on grounded ice.  Under ``budd`` and
        ``regularized_coulomb`` it does not leak onto real shelves either
        (those keep ``N = 0``, hence ``tau_b = 0`` exactly); under
        ``weertman`` there is no ``N`` factor, so a floored ``C_w0`` does
        produce shelf drag.  Leave at 0 when ``H`` is clamped positive
        upstream.
    N_ref : Function or None
        Budd only.  Reference effective pressure for the normalisation.
        ``None`` uses the current ``N``, giving ``N_hat = 1`` on grounded
        ice -- correct at the inversion geometry.
    nhat_floor : float
        Budd only, and **requires an explicit** ``N_ref``.  PISM-style
        delta floor on ``N_hat`` as a fraction of local overburden (Bueler
        & van Pelt 2015 use ~0.02), which removes the frictionless
        degeneracy near flotation.  The floor evolves with the overburden,
        so it decays as the ice thins.  0 disables it.

        The floor is measured against ``N_ref``, so it only has PISM
        semantics when ``N_ref`` is a frozen snapshot.  With
        ``N_ref=None`` the reference *is* the current ``N``, ``N_hat`` is
        already identically 1 on grounded ice, and the term would
        *amplify* near-flotation drag up to ``nhat_cap`` rather than floor
        anything -- so that combination is rejected outright.
    nhat_cap : float
        Budd only.  Upper bound on ``N_hat`` (Joughin's ``reduceNearGLBeta``).
    """
    if law not in LAWS:
        raise ValueError(f"unknown friction law {law!r}; expected one of {LAWS}")
    if law == "budd" and nhat_floor > 0.0 and N_ref is None:
        raise ValueError(
            "nhat_floor > 0 requires an explicit N_ref: the PISM floor "
            "semantics only exist against a frozen N_ref snapshot.  With "
            "N_ref=None the reference is the current N, so N_hat is "
            "identically 1 on grounded ice and the floor term "
            "nhat_floor * p_I / N amplifies near-flotation drag up to "
            "nhat_cap instead of flooring it.  Pass a frozen N_ref (e.g. the "
            "inversion-geometry effective pressure) or set nhat_floor=0."
        )
    if He is None:
        He = grounded_mask(H, b, gl_width=gl_width)
    N = effective_pressure(H, s)

    u_reg = sqrt(inner(u, u) + Constant(u_min) ** 2)
    C_eff = max_value(C_w0, Constant(c_w0_floor)) if c_w0_floor else C_w0
    tau_W = C_eff * exp(theta * He) * u_reg ** (1.0 / m_slide)

    if law == "weertman":
        return tau_W

    if law == "budd":
        # N_hat is the NORMALISED effective pressure, so the inverted
        # friction is preserved at the reference geometry and the
        # effective-pressure feedback is a relative change thereafter.
        # Floor the denominator ALWAYS, not just to guard the exact-zero
        # branch: UFL evaluates both sides of a conditional, so an
        # unguarded 0/0 on the shelf poisons the Jacobian with a NaN even
        # though the conditional selects 0.
        Nr = max_value(N if N_ref is None else N_ref, Constant(1e-6))
        N_hat = min_value(N / Nr, Constant(nhat_cap))
        if nhat_floor > 0.0:
            p_I = ice_density * gravity * max_value(H, Constant(1.0))
            N_hat = min_value(max_value(N_hat, Constant(nhat_floor) * p_I / Nr),
                              Constant(nhat_cap))
        # He, not the sign of N, is what enforces the zero afloat.  N is
        # the *cancelling* difference p_I - p_W, which on a shelf is a
        # roundoff residue rather than exactly 0, so gt(N, 0) lets an
        # O(1e-16) N through -- and with N_ref equally tiny there, the
        # floor term nhat_floor * p_I / Nr then amplifies it all the way
        # to nhat_cap.  He is a function of height above flotation, so it is
        # 0 hundreds of metres below flotation whatever N's roundoff does,
        # and it is continuous where the conditional was not.  The
        # conditional stays as a harmless guard on the sign of N.
        N_hat = He * conditional(gt(N, Constant(0.0)), N_hat, Constant(0.0))
        return tau_W * N_hat

    # regularized_coulomb: harmonic blend -> tau_W at low speed, -> tau_cap
    # at high speed, and -> 0 exactly where N = 0.  The max_value guards
    # 0/0 on the shelf for the same reason as above.
    tau_cap = max_value(Constant(c0) * N, Constant(eps_tauc))
    return tau_W * tau_cap / max_value(tau_W + tau_cap, Constant(1e-15))


def friction_residual(tau, sigma, u, tau_b, u_min=U_MIN):
    r"""Close the dual basal stress: :math:`\tau = -\tau_b\,u/|u|_{\rm reg}`.

    Linear in ``tau``, so the basal-stress block of the Jacobian is the
    identity -- non-singular at ``tau = 0``, which is what makes a cold
    start possible.

    The sign follows icepack2: :math:`\tau` opposes the velocity.
    """
    u_reg = sqrt(inner(u, u) + Constant(u_min) ** 2)
    return inner(tau + tau_b * u / u_reg, sigma) * dx


def ocean_drag(u, H, drag, h_ocean=10.0, u_min=U_MIN, cellwise=True):
    r"""Linear drag confined to ice-free cells, ramping to zero at ``h_ocean``.

    Frictionless floor cells (``C_w0`` and ``N`` both ~0) have no velocity
    coercivity, and a thick front adjacent to them pumps momentum through
    the surface-jump facet term into the degenerate side.  This bounds
    that without touching real ice.

    ``cellwise`` is what makes "without touching real ice" true.  The
    basal stress lives in DG0, so :func:`friction_residual` sets each
    cell's :math:`\tau` to the cell *average* of this term -- and with a
    continuous ``H`` the ramp is only **pointwise** zero above
    ``h_ocean``, so a front cell straddling ``H = h_ocean`` averages in a
    share of the drag and carries it as if it were bed traction.  On a
    250 m Store mesh that put up to 338 kPa on cells whose centroid held
    13-23 m of ice and whose effective pressure was exactly zero, more
    than the trunk's own driving stress.

    Reducing ``H`` to DG0 first makes the support a union of whole cells,
    matching the space the stress is resolved in: a cell is ice-free or
    it is not.  Pass ``cellwise=False`` to keep the pointwise ramp -- for
    a DG0 ``H``, or a non-``Function`` expression, it is already what you
    get.
    """
    u_reg = sqrt(inner(u, u) + Constant(u_min) ** 2)
    h = H
    if cellwise and isinstance(H, Function):
        if H.function_space().ufl_element().sobolev_space == ufl.H1:
            DG = FunctionSpace(H.function_space().mesh(), "DG", 0)
            h = Function(DG).interpolate(H)
    ramp = max_value(Constant(0.0), Constant(1.0) - h / Constant(h_ocean))
    return Constant(drag) * ramp * u_reg


def speed_limiter(u, u_lim, k_lim=1e-3, u_min=U_MIN):
    r"""Supplemental drag above ``u_lim``, exactly zero below it.

    A backstop for runaway fronts; set ``u_lim`` well above any physical
    speed so real flow never feels it.
    """
    u_reg = sqrt(inner(u, u) + Constant(u_min) ** 2)
    k = k_lim if isinstance(k_lim, Constant) else Constant(k_lim)
    return k * max_value(Constant(0.0), u_reg - Constant(u_lim))
