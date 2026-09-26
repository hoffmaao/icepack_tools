r"""Momentum balance and full dual residuals, single- and multi-layer.

The per-layer momentum term is written out rather than taken from
``icepack2.model.variational`` for one reason: the membrane coupling may
need a thickness floor for coercivity at ice-free nodes, while the
driving stress must keep the true thickness.  Clamping ``H`` in the
driving term fabricates a spurious :math:`\rho g H_{\rm floor}\nabla s`
and blows the buffer velocity up.

The pair

.. code::

    -h M : eps(v) + ...  - rho_I g H grad(s) . v   dx
    rho_I g avg(H) jump(s, nu) . avg(v)            dS

is the distributional gradient of a possibly discontinuous surface,
written as broken cell gradient plus facet jump.  It is correct for a CG1
*or* a DG0 surface: with CG1 the jump vanishes and the cell term carries
everything, with DG0 the cell gradient vanishes and the facet term does.
Do not delete the seemingly dead ``grad(s)`` under DG0 -- that would break
the CG1 path -- and the facet term is not an add-on but the whole driving
stress in the DG0 case.
"""

from firedrake import (
    Constant, FacetNormal, avg, dS, dx, exp, grad, inner, jump, max_value,
    split, sqrt, sym, TestFunction,
)
from icepack2 import model as _icepack2_model

from .constants import ice_density, water_density, gravity
from .friction import basal_stress, friction_residual
from .viscosity import (membrane_residual, interlayer_residual,
                        H_JUMP_FLOOR)


def momentum_residual(u, v, M, h, s, H, mesh, *, tau=None, stress_above=None,
                      stress_below=None, h_floor=0.0, rho_I=ice_density,
                      g=gravity):
    r"""One layer's momentum balance, in residual form.

    ``tau`` is the basal stress (bottom layer only), ``stress_above`` and
    ``stress_below`` the interlayer stresses on the two interfaces.  Signs
    follow ``multilayer.model.variational.momentum_balance``.

    ``H`` is the *total* column thickness, used for the driving stress;
    ``h`` is this layer's share, used for the membrane coupling.

    There is deliberately no ``layer_fraction`` here, unlike in
    ``calving_terminus``: this layer's share of the driving stress is
    already carried by ``h``, its own thickness, so scaling by a layer
    fraction on top would double-count it.  The back-pressure in
    ``calving_terminus`` does need one, because it goes as the total
    column ``H`` squared and has to be split across layers explicitly.
    """
    h_v = max_value(h, Constant(h_floor)) if h_floor else h
    nu = FacetNormal(mesh)

    F = (-h_v * inner(M, sym(grad(v)))
         - rho_I * g * h * inner(grad(s), v)) * dx
    F += rho_I * g * avg(h) * inner(jump(s, nu), avg(v)) * dS
    if tau is not None:
        F += inner(tau, v) * dx
    if stress_above is not None:
        F += inner(stress_above, v) * dx
    if stress_below is not None:
        F -= inner(stress_below, v) * dx
    return F


def calving_terminus(u, v, H, s, outflow_ids, layer_fraction=1.0):
    r"""Ocean back-pressure on an outflow boundary.

    Needed only when the calving front *is* the domain boundary.  If the
    domain is buffered past the front so the terminus is interior, leave
    this out: icepack2's dual form is well posed at zero thickness and the
    front's stress balance comes out of the thickness gradient itself.
    Adding it there double-counts.
    """
    return Constant(layer_fraction) * _icepack2_model.variational.calving_terminus(
        velocity=u, thickness=H, surface=s, outflow_ids=tuple(outflow_ids)
    )


def dual_residual(z, theta, phi, *, H, s, b, h_layers, C_w0,
                  A_layers, n_consts, n_vals, m_slide, mesh,
                  law="regularized_coulomb",
                  layer_fractions=None, tau_c=0.1, alpha=1e-4,
                  H_ref=100.0, A_lin_layers=None, c0=0.5, u_min=1.0,
                  eps_tauc=0.0, N_ref=None, nhat_floor=0.0, nhat_cap=3.0,
                  c_w0_floor=0.0, h_visc_floor=0.0, alpha_gl=0.0,
                  u_lim=0.0, k_lim=1e-3, gl_width=10.0,
                  h_jump_floor=H_JUMP_FLOOR,
                  outflow_ids=None, subelement=None,
                  rho_I=ice_density, rho_W=water_density, g=gravity):
    r"""Full dual residual for an ``L``-layer column, ``L >= 1``.

    **``L = 1`` is the ordinary single-layer icepack2 dual model** -- the
    interlayer loop is empty and the state reduces to ``(u, M, tau)`` on
    ``V x Sigma x T``, byte-identical to what the single-layer consumers
    build by hand.  So one builder serves both; there is no separate
    single-layer code path to keep in step.

    The mixed state is ordered as :func:`icepack_tools.spaces.dual_function_space`
    builds it:

    .. code::

        z = [u^1, M^1, S^0,  u^2, M^2, S^1,  ...]

    with ``S^0`` the basal stress and ``S^l`` (l > 0) the interlayer stress
    between layers ``l`` and ``l+1``.

    Controls
    --------
    theta : Function
        Log friction adjustment, gated to grounded ice.
    phi : Function
        Log fluidity adjustment.  Applied to the layers flagged in
        ``A_layers`` by passing an already-scaled expression.

    Other Parameters
    ----------------
    A_lin_layers : sequence, optional
        Per-layer diffusion-creep (:math:`n = 1`) prefactors, one entry
        per layer, alongside ``A_layers``/``n_consts``/``n_vals``.
        ``None`` (the default) turns diffusion creep off everywhere; an
        individual entry may be ``None`` to disable it for that layer.
        Layer ``l``'s membrane closure takes ``A_lin_layers[l]``; the
        closure on interface ``l`` takes ``A_lin_layers[l - 1]``, the
        layer *below* the interface, matching the ``A_layers[l - 1]``
        convention already used there.

        Per-layer because diffusion creep's prefactor depends on
        temperature and grain size, so a warm basal layer and a cold
        surface layer want different values.

        These prefactors are **not** multiplied by ``exp(phi)``.
        Diffusion creep is treated as a fixed physical mechanism whose
        prefactor is prescribed rather than inferred; the inverted
        log-fluidity ``phi`` deliberately controls only the
        dislocation-creep component.  The consequence is worth stating
        plainly: at low deviatoric stress diffusion carries a large share
        of the effective fluidity, and ``phi`` cannot adjust that share.
        In this package's convention, for the ``n = 4``, ``A = 46`` layer
        with ``A_lin = 1e-3``, diffusion is 98 % of the effective-fluidity
        bracket at 10 kPa, 80 % at 25 kPa, 33 % at 50 kPa and 6 % at
        100 kPa; for an ``n = 1.8``, ``A = 0.451`` layer it is 10 % at
        10 kPa falling to 2 % at 100 kPa.

    h_jump_floor : float
        Floor [m] on ``h_above + h_below`` in the interlayer velocity-jump
        normalisation, which vanishes with the column and is exactly zero
        at ice-free nodes -- an ocean buffer past a calving front -- where
        the unguarded division makes the residual NaN.  It floors nothing
        but that denominator: not the driving stress, not the effective
        pressure, not the membrane coupling, which has its own
        ``h_visc_floor``.  Defaults to
        :data:`icepack_tools.viscosity.H_JUMP_FLOOR`; see it for the value
        and for the column thickness at which the floor engages.

    law : str
        One of :data:`icepack_tools.friction.LAWS` -- ``regularized_coulomb``
        (default), ``budd`` or ``weertman``.  ``N_ref``, ``nhat_floor`` and
        ``nhat_cap`` apply to ``budd`` only; ``c0`` and ``eps_tauc`` to
        ``regularized_coulomb`` only.

    subelement : :class:`icepack_tools.grounding.SubelementGrounding`, optional
        Integrate the basal friction over the grounded part of each cell
        only, ISSM's ``SubelementFriction2`` (Seroussi et al., 2014, SEP2),
        instead of switching a whole cell on its cell-mean height above
        flotation.  On a fully grounded cell the friction is the unchanged
        Weertman stress over the whole cell; on a partly grounded one it is
        the same stress at the degree-2 Gauss points of the grounded part;
        a floating cell has none.  The basal stress, a single vector per
        cell, is then applied at the centroid of the grounded part, where
        ISSM pairs each point's drag with the test function at that point:
        the difference is second order in the variation of the drag across
        a cell, and on every other cell it is exactly the usual term.  The
        caller keeps the quadrature current (``subelement.update``) whenever
        the geometry changes; it is fixed during a solve, so the Jacobian
        keeps its structure.

        Exact only where the grounded stress needs nothing but the
        velocity and the controls: ``law="weertman"``, and ``law="budd"``
        with ``N_ref=None`` and no floor, whose grounded ``N_hat`` is
        identically 1.  A law that reads the effective pressure would need
        it at the points too (``rho_I g haf`` there), which this does not
        yet do, so those combinations are refused.  The smooth ``He`` of
        ``gl_width`` plays no part: the grounded part is exact.

    rho_I, rho_W, g : float
        Ice density, seawater density and gravity in icepack2 units
        (MPa, m, yr), defaulting to icepack2's constants.  They enter the
        driving stress, the grounded indicator and the effective pressure.
        A consumer whose protocol fixes other values -- MISMIP+ and
        CalvingMIP both prescribe 1028 kg/m^3 seawater against icepack2's
        1024 -- passes them here so the friction cap and the grounding
        indicator vanish at the same flotation thickness as its own
        hydrostatic surface, instead of a few metres away from it.  The
        terminus back-pressure (``outflow_ids``) still uses icepack2's
        constants, as it comes straight from ``icepack2.model``.

    Notes
    -----
    Every block is closed in residual form, so the Jacobian is
    non-singular at zero stress and no continuation in ``m`` is needed.
    With ``A_lin_layers`` set, the :math:`n = 1` diffusion term leaves a
    constant, non-zero Hessian contribution at ``M = 0``, so no
    continuation in ``n`` is needed either: ``n_consts`` may be set
    straight to their targets and the whole staged ramp disappears.  They
    stay mutable Constants because an n-ramp is still *required* when
    diffusion creep is off, and ``n_vals`` fixes the stress-matching
    powers so the linear regulariser does not move during such a ramp.
    """
    from .grounding import grounded_mask

    num_layers = len(h_layers)
    if layer_fractions is None:
        layer_fractions = [1.0 / num_layers] * num_layers
    if A_lin_layers is None:
        A_lin_layers = [None] * num_layers
    elif len(A_lin_layers) != num_layers:
        raise ValueError(
            f"A_lin_layers has {len(A_lin_layers)} entries, expected one per "
            f"layer ({num_layers})"
        )

    fields = split(z)
    if len(fields) != 3 * num_layers:
        raise ValueError(
            f"z lives in a {len(fields)}-block space but h_layers describes "
            f"{num_layers} layer(s) ({3 * num_layers} blocks expected).  "
            f"dual_function_space(mesh, num_layers) and layer_thicknesses(H, "
            f"num_layers) must agree on the layer count; otherwise the extra "
            f"blocks appear in no term and the Jacobian is structurally "
            f"singular."
        )
    tests = split(TestFunction(z.function_space()))
    He = grounded_mask(H, b, gl_width=gl_width, rho_I=rho_I, rho_W=rho_W)

    F = None
    for l in range(num_layers):
        u_l, M_l, S_l = fields[3 * l], fields[3 * l + 1], fields[3 * l + 2]
        v_l, Mt_l, _ = tests[3 * l], tests[3 * l + 1], tests[3 * l + 2]
        h_l = h_layers[l]

        term = membrane_residual(
            M_l, Mt_l, u_l, h_l, A_layers[l], n_consts[l], n_val=n_vals[l],
            A_lin=A_lin_layers[l], tau_c=tau_c, alpha=alpha, H_ref=H_ref,
            h_floor=h_visc_floor,
            extra_linear=(Constant(alpha_gl) * (Constant(1.0) - He)
                          if alpha_gl > 0 else None),
        )
        term += momentum_residual(
            u_l, v_l, M_l, h_l, s, H, mesh,
            tau=S_l if l == 0 and subelement is None else None,
            stress_above=fields[3 * (l + 1) + 2] if l < num_layers - 1 else None,
            stress_below=S_l if l > 0 else None,
            h_floor=h_visc_floor, rho_I=rho_I, g=g,
        )
        if outflow_ids:
            term += calving_terminus(u_l, v_l, H, s, outflow_ids,
                                     layer_fraction=layer_fractions[l])
        if l == 0 and subelement is not None:
            # the basal stress acts on the grounded part of its cell
            term += inner(S_l, subelement.at_centroid(v_l)) * dx
        F = term if F is None else F + term

    # basal stress: residual closure for the chosen `law` on layer 0
    u_b = fields[0]
    if subelement is None:
        tau_b = basal_stress(u_b, C_w0, theta, H, s, b, m_slide, law=law, c0=c0,
                             u_min=u_min, eps_tauc=eps_tauc, He=He,
                             gl_width=gl_width, c_w0_floor=c_w0_floor,
                             N_ref=N_ref, nhat_floor=nhat_floor,
                             nhat_cap=nhat_cap, rho_I=rho_I, rho_W=rho_W, g=g)
        if u_lim > 0.0:
            from .friction import speed_limiter
            tau_b = tau_b + speed_limiter(u_b, u_lim, k_lim, u_min)
        F += friction_residual(fields[2], tests[2], u_b, tau_b, u_min=u_min)
    else:
        F += _subelement_friction(subelement, fields[2], tests[2], u_b, C_w0,
                                  theta, m_slide, law=law, N_ref=N_ref,
                                  nhat_floor=nhat_floor, u_min=u_min,
                                  c_w0_floor=c_w0_floor, u_lim=u_lim,
                                  k_lim=k_lim)

    # interlayer shear closures
    for l in range(1, num_layers):
        F += interlayer_residual(
            fields[3 * l + 2], tests[3 * l + 2],
            u_above=fields[3 * l], u_below=fields[3 * (l - 1)],
            h_above=h_layers[l], h_below=h_layers[l - 1],
            A=A_layers[l - 1], n=n_consts[l - 1], n_val=n_vals[l - 1],
            A_lin=A_lin_layers[l - 1], tau_c=tau_c, alpha=alpha,
            h_jump_floor=h_jump_floor,
        )
    return F


#: Backwards-compatible alias.  The builder was named for the multilayer
#: case before it was generalised; ``L = 1`` was always the single-layer
#: model, so the name was misleading rather than the code.
multilayer_rc_residual = dual_residual


def _subelement_friction(sub, tau, sigma, u, C_w0, theta, m_slide, *, law,
                         N_ref, nhat_floor, u_min, c_w0_floor, u_lim, k_lim):
    r"""The basal-stress closure integrated over each cell's grounded part.

    ``tau = -(1/|K|) int_{K_g} tau_W(u) u/|u| dx`` on every cell ``K`` with
    grounded part ``K_g``: the whole cell when it is fully grounded (the
    ordinary rule), the degree-2 Gauss points of ``K_g`` when it is partly
    grounded (:class:`icepack_tools.grounding.SubelementGrounding`), and
    nothing afloat.  ``tau_W = C_w0 exp(theta) |u|^(1/m)``, Weertman with the
    friction control, unscaled: SEP2 leaves the friction on the grounded
    part unchanged.  See :func:`dual_residual`, ``subelement``.
    """
    if law == "budd":
        if N_ref is not None or nhat_floor > 0.0:
            raise NotImplementedError(
                "subelement friction with law='budd' needs N_ref=None and "
                "nhat_floor=0: with a reference effective pressure the "
                "grounded N_hat is not 1 and would have to be evaluated at "
                "the grounded-part points (rho_I g haf there), which is not "
                "implemented")
    elif law != "weertman":
        raise NotImplementedError(
            f"subelement friction is exact for 'weertman' and N_ref-free "
            f"'budd' only, not {law!r}: a law that reads the effective "
            f"pressure needs it at the grounded-part points")

    def drag(ev):
        # the Weertman stress times u/|u| with every field at one point
        u_q = ev(u)
        u_reg = sqrt(inner(u_q, u_q) + Constant(u_min) ** 2)
        C = ev(C_w0) if not isinstance(C_w0, Constant) else C_w0
        if c_w0_floor:
            C = max_value(C, Constant(c_w0_floor))
        return C * exp(ev(theta)) * u_reg ** (1.0 / m_slide - 1.0) * u_q

    u_reg = sqrt(inner(u, u) + Constant(u_min) ** 2)
    F = inner(tau, sigma) * dx
    F += sub.full * inner(drag(lambda f: f), sigma) * dx
    F += inner(sub.integrand(drag), sigma) * dx
    if u_lim > 0.0:
        from .friction import speed_limiter
        F += inner(speed_limiter(u, u_lim, k_lim, u_min) * u / u_reg, sigma) * dx
    return F
