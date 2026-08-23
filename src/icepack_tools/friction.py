r"""Regularised-Coulomb basal friction for the icepack2 dual form.

Written as a variational **residual that closes the basal stress
directly**, not as a dissipation potential.  This is the crux of the
module, so it is worth stating plainly why.

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

Closing :math:`\tau = \tau_{RC}(u)` as a residual instead makes the
basal-stress block of the dual system the **identity** -- perfectly
conditioned at :math:`\tau = 0` -- and needs no continuation in the
sliding exponent.  The Coulomb limit is reached through a harmonic blend
that is smooth everywhere:

.. math::
    \tau_W   &= C_{w0}\,e^{\theta H_e}\,|u|_{\rm reg}^{1/m}  \\
    \tau_{\rm cap} &= c_0 N                                   \\
    \tau_b   &= \frac{\tau_W\,\tau_{\rm cap}}{\tau_W + \tau_{\rm cap}}

which is Weertman at low speed, the Coulomb cap at high speed, and
**exactly zero** where the ice floats, because :math:`N \to 0` there.

Ported and generalised from ``ismip7/icepack2_tools/dual_friction.py``
(itself derived from ``gia-icepack/scripts/ase_model.py:build_F_rc``).
"""

from firedrake import (
    Constant, Function, sqrt, inner, exp, max_value, dx,
)

from .constants import ice_density, gravity
from .geometry import surface_slope
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
    """
    grad_s = surface_slope(s)
    tau_d = rho_I * g * H * sqrt(inner(grad_s, grad_s) + Constant(1e-12))
    speed = max_value(sqrt(inner(u_obs, u_obs)), Constant(u_floor))
    return Function(Q, name=name).interpolate(tau_d / speed ** (1.0 / m_slide))


def basal_stress(u, C_w0, theta, H, s, b, m_slide, *, c0=C0, u_min=U_MIN,
                 eps_tauc=0.0, He=None, gl_width=GL_WIDTH, c_w0_floor=0.0):
    r"""Regularised-Coulomb basal stress magnitude :math:`\tau_b` (scalar, MPa).

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
        does not leak onto real shelves (those keep ``N = 0``, hence
        ``tau_b = 0`` exactly) and is absorbed by ``theta`` on grounded ice.
        Leave at 0 when ``H`` is clamped positive upstream.
    """
    if He is None:
        He = grounded_mask(H, b, gl_width=gl_width)
    N = effective_pressure(H, s)

    u_reg = sqrt(inner(u, u) + Constant(u_min) ** 2)
    C_eff = max_value(C_w0, Constant(c_w0_floor)) if c_w0_floor else C_w0
    tau_W = C_eff * exp(theta * He) * u_reg ** (1.0 / m_slide)
    tau_cap = max_value(Constant(c0) * N, Constant(eps_tauc))

    # Harmonic blend: -> tau_W at low speed, -> tau_cap at high speed, and
    # -> 0 exactly where N = 0.  The max_value guards 0/0 on the shelf,
    # where UFL would otherwise poison the Jacobian with a NaN.
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


def ocean_drag(u, H, drag, h_ocean=10.0, u_min=U_MIN):
    r"""Linear drag confined to ice-free cells, ramping to zero at ``h_ocean``.

    Frictionless floor cells (``C_w0`` and ``N`` both ~0) have no velocity
    coercivity, and a thick front adjacent to them pumps momentum through
    the surface-jump facet term into the degenerate side.  This bounds
    that without touching real ice.  Exactly zero for ``H >= h_ocean``.
    """
    u_reg = sqrt(inner(u, u) + Constant(u_min) ** 2)
    ramp = max_value(Constant(0.0), Constant(1.0) - H / Constant(h_ocean))
    return Constant(drag) * ramp * u_reg


def speed_limiter(u, u_lim, k_lim=1e-3, u_min=U_MIN):
    r"""Supplemental drag above ``u_lim``, exactly zero below it.

    A backstop for runaway fronts; set ``u_lim`` well above any physical
    speed so real flow never feels it.
    """
    u_reg = sqrt(inner(u, u) + Constant(u_min) ** 2)
    k = k_lim if isinstance(k_lim, Constant) else Constant(k_lim)
    return k * max_value(Constant(0.0), u_reg - Constant(u_lim))
