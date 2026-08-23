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
    Constant, FacetNormal, avg, dS, dx, grad, inner, jump, max_value, split,
    sym, TestFunction,
)
from icepack2 import model as _icepack2_model

from .constants import ice_density, gravity
from .friction import basal_stress, friction_residual
from .viscosity import membrane_residual, interlayer_residual


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


def multilayer_rc_residual(z, theta, phi, *, H, s, b, h_layers, C_w0,
                           A_layers, n_consts, n_vals, m_slide, mesh,
                           layer_fractions=None, tau_c=0.1, alpha=1e-4,
                           H_ref=100.0, A_lin_layers=None, c0=0.5, u_min=1.0,
                           eps_tauc=0.0,
                           c_w0_floor=0.0, h_visc_floor=0.0, alpha_gl=0.0,
                           ocean_drag_coeff=0.0, h_ocean=10.0,
                           u_lim=0.0, k_lim=1e-3, gl_width=10.0,
                           outflow_ids=None):
    r"""Full dual residual for an ``L``-layer column on a Coulomb bed.

    The mixed state is ordered as ``multilayer`` builds it:

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
    tests = split(TestFunction(z.function_space()))
    He = grounded_mask(H, b, gl_width=gl_width)

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
            tau=S_l if l == 0 else None,
            stress_above=fields[3 * (l + 1) + 2] if l < num_layers - 1 else None,
            stress_below=S_l if l > 0 else None,
            h_floor=h_visc_floor,
        )
        if outflow_ids:
            term += calving_terminus(u_l, v_l, H, s, outflow_ids,
                                     layer_fraction=layer_fractions[l])
        F = term if F is None else F + term

    # basal stress: regularised-Coulomb residual closure on layer 0
    u_b = fields[0]
    tau_b = basal_stress(u_b, C_w0, theta, H, s, b, m_slide, c0=c0,
                         u_min=u_min, eps_tauc=eps_tauc, He=He,
                         gl_width=gl_width, c_w0_floor=c_w0_floor)
    if ocean_drag_coeff > 0.0:
        from .friction import ocean_drag
        tau_b = tau_b + ocean_drag(u_b, H, ocean_drag_coeff, h_ocean, u_min)
    if u_lim > 0.0:
        from .friction import speed_limiter
        tau_b = tau_b + speed_limiter(u_b, u_lim, k_lim, u_min)
    F += friction_residual(fields[2], tests[2], u_b, tau_b, u_min=u_min)

    # interlayer shear closures
    for l in range(1, num_layers):
        F += interlayer_residual(
            fields[3 * l + 2], tests[3 * l + 2],
            u_above=fields[3 * l], u_below=fields[3 * (l - 1)],
            h_above=h_layers[l], h_below=h_layers[l - 1],
            A=A_layers[l - 1], n=n_consts[l - 1], n_val=n_vals[l - 1],
            A_lin=A_lin_layers[l - 1], tau_c=tau_c, alpha=alpha,
        )
    return F
