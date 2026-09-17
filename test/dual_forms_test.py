r"""Is the builder actually general, or just the multilayer case in disguise?

Claims the other icepack repos would rely on:

  * ``dual_function_space(mesh, 1)`` is the ordinary single-layer dual
    space.  The single-layer consumers build ``Z = V * Sigma * T`` by hand
    (see ``ismip7/antarctica/scripts/diagnostic_solve.py``); this checks
    the helper reproduces it element for element, so adopting the helper
    is not a silent change of discretisation.
  * ``dual_residual`` at ``L = 1`` solves a single-layer problem, with the
    interlayer machinery inert rather than merely unused.
  * every law leaves the (tau, tau) Jacobian block equal to the mass
    matrix at ``z = 0`` -- the residual closure is not traded away by any
    of them, so none needs a continuation in the sliding exponent.
  * all three friction laws close and solve, and the two with an
    effective-pressure cap give *exactly* zero drag on floating ice while
    Weertman -- correctly, since it has no cap -- does not.
  * the two entry points that must agree on ``L`` say so when they do
    not, and Budd's PISM floor is only accepted where it means what it
    says (against a frozen ``N_ref``), where its amplification and its
    grounding-line jump are then pinned.
  * a column that goes genuinely ice-free -- ``H = 0``, as an ocean
    buffer seaward of the calving front is -- still assembles a finite
    multilayer residual, and does so *because* of ``h_jump_floor``:
    turn the floor off and the same assembly is NaN.

    python -u dual_forms_test.py
"""
import numpy as np
import firedrake
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace, TensorFunctionSpace,
    SpatialCoordinate, DirichletBC, NonlinearVariationalProblem,
    NonlinearVariationalSolver, as_vector, dx, inner, sqrt, max_value,
)

from icepack_tools.constants import ice_density as rho_I, water_density as rho_W
from icepack_tools.friction import basal_stress, weertman_anchor, LAWS
from icepack_tools.grounding import (
    effective_pressure, grounded_mask, height_above_flotation,
)
from icepack_tools.momentum import dual_residual
from icepack_tools.spaces import (
    dual_function_space, layer_thicknesses, split_layers,
)
from icepack_tools.viscosity import A_DIFFUSION, H_JUMP_FLOOR

M_SLIDE = 3.0

# Bounds on machine-zero shelf drag, quoted verbatim in readme.md so the two
# cannot disagree.  Afloat, tau_b is *analytically* zero for a capped law: N
# is clamped to max(p_I - p_W, 0) and enters tau_b as a multiplicative factor,
# so there is no threshold to sit on the wrong side of.  tau is a solved-for
# unknown though, so what comes back is the linear solve's roundoff on that
# zero -- it drifts whenever anything upstream perturbs the system, hence a
# bound with room for that rather than the printed value.
#
# budd is kept orders tighter because it gates on the grounded indicator He as
# well as capping on N, so its shelf residue is the tau solve's own roundoff
# rather than the cap's; a single bound loose enough for both would stop
# testing that.  Its residue is also the more volatile of the two -- it moved
# six decades under this branch's anchor change, where the coulomb one moved a
# factor of 1.3 -- so it carries the wider margin.
SHELF_DRAG_MAX_KPA = {              # observed:
    "regularized_coulomb": 1e-13,   #   5.3e-16 kPa
    "budd": 1e-15,                  #   1.6e-20 kPa
}
SHELF_DRAG_MAX_REL = {              # observed, as a fraction of grounded max:
    "regularized_coulomb": 1e-14,   #   1.7e-18
    "budd": 1e-18,                  #   4.9e-23
}
# weertman has no effective-pressure cap, so it must NOT be machine zero --
# without this the two bounds above could pass vacuously.
SHELF_DRAG_MIN_REL_UNCAPPED = 1e-6  # observed 1.9e-02

FC = {"quadrature_degree": 4}
SPARAMS = {
    "snes_type": "newtonls", "snes_max_it": 100,
    "snes_linesearch_type": "nleqerr", "snes_divergence_tolerance": -1,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    # the dual form is a saddle point: the whole (u, u) diagonal is zero
    "mat_mumps_icntl_14": 200, "mat_mumps_icntl_24": 1, "mat_mumps_cntl_3": 1e-6,
}


def build(nx=20, Lx=40e3, Ly=12e3):
    """Slab: grounded inland, marine and thinning toward x = Lx."""
    mesh = firedrake.RectangleMesh(nx, max(4, nx // 3), Lx, Ly, diagonal="crossed")
    x, y = SpatialCoordinate(mesh)
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    H = Function(Q).interpolate(Constant(900.0) - Constant(840.0) * x / Constant(Lx))
    b = Function(Q).interpolate(Constant(100.0) - Constant(800.0) * x / Constant(Lx))
    s = Function(Q).interpolate(
        max_value(b + H, Constant(1.0 - rho_I / rho_W) * H))
    u_obs = Function(V).interpolate(
        as_vector([Constant(80.0) + Constant(900.0) * x / Constant(Lx), Constant(0.0)]))
    return mesh, Q, H, b, s, u_obs


def test_space_matches_handbuilt():
    """L = 1 must reproduce the space single-layer consumers write by hand."""
    mesh, *_ = build(nx=8)
    Z = dual_function_space(mesh, 1)

    cg1 = firedrake.FiniteElement("CG", "triangle", 1)
    dg0 = firedrake.FiniteElement("DG", "triangle", 0)
    V = VectorFunctionSpace(mesh, cg1)
    Sigma = TensorFunctionSpace(mesh, dg0, symmetry=True)
    T = VectorFunctionSpace(mesh, dg0)
    Z_ref = V * Sigma * T

    assert len(Z) == len(Z_ref) == 3, f"L=1 should give 3 blocks, got {len(Z)}"
    assert Z.dim() == Z_ref.dim(), f"{Z.dim()} != {Z_ref.dim()}"
    for i, (a, e) in enumerate(zip(Z, Z_ref)):
        assert a.ufl_element() == e.ufl_element(), (
            f"block {i}: {a.ufl_element()} != {e.ufl_element()}")
    print(f"  L=1 space matches hand-built V*Sigma*T exactly "
          f"({Z.dim():,} DOFs, 3 blocks)")

    Z2 = dual_function_space(mesh, 2)
    assert len(Z2) == 6 and Z2.dim() == 2 * Z.dim()
    print(f"  L=2 space is the same triple twice ({Z2.dim():,} DOFs, 6 blocks)")


def build_case(num_layers, law, n_vals, A_vals, fractions=None, nx=20):
    """Geometry, mixed space and residual at a cold start ``z = 0``."""
    mesh, Q, H, b, s, u_obs = build(nx=nx)
    Z = dual_function_space(mesh, num_layers)
    h_layers = layer_thicknesses(H, num_layers, fractions=fractions)
    C_w0 = weertman_anchor(H, s, u_obs, M_SLIDE, Q)
    theta, phi = Function(Q), Function(Q)

    z = Function(Z)                                    # cold start
    F = dual_residual(
        z, theta, phi, H=H, s=s, b=b, h_layers=h_layers, C_w0=C_w0,
        A_layers=[Constant(a) for a in A_vals],
        n_consts=[Constant(v) for v in n_vals], n_vals=n_vals,
        A_lin_layers=[Constant(A_DIFFUSION)] * num_layers,
        m_slide=M_SLIDE, mesh=mesh, law=law, layer_fractions=fractions,
        h_visc_floor=10.0, c_w0_floor=1e-6,
    )
    return mesh, Q, H, b, s, Z, z, F


def test_identity_tau_block():
    """Every law must leave the basal-stress block of the Jacobian alone.

    This is the property the package exists for: closing tau as a residual
    makes the (tau, tau) block the T-space mass matrix -- the identity up
    to cell volume -- so it is non-singular at tau = 0 and no continuation
    in ``m`` is needed.  A potential-form closure would put ``|tau|^(m-1)``
    there and be rank-deficient at the cold start.  Checked for all three
    laws, because the point is that none of them trades it away.
    """
    for law in LAWS:
        mesh, Q, H, b, s, Z, z, F = build_case(1, law, [3.0], [20.0], nx=8)
        J = firedrake.assemble(firedrake.derivative(F, z), mat_type="nest",
                               form_compiler_parameters=FC)
        blk = J.petscmat.getNestSubMatrix(2, 2)          # d(tau residual)/dtau
        T = Z.sub(2)
        mass = firedrake.assemble(
            inner(firedrake.TrialFunction(T), firedrake.TestFunction(T)) * dx,
            form_compiler_parameters=FC).petscmat
        diff = blk.copy()
        diff.axpy(-1.0, mass)
        diag = np.asarray(blk.getDiagonal())
        assert diff.norm() == 0.0, (
            f"{law}: (tau, tau) block differs from the mass matrix by "
            f"{diff.norm():.3e} -- the closure is not a residual one")
        assert diag.min() > 0.0, f"{law}: singular tau-block at z = 0"
        print(f"  {law:<22} |J_tau,tau - mass| = {diff.norm():.1e}, "
              f"min diagonal {diag.min():.3e} > 0")


def solve_case(num_layers, law, n_vals, A_vals, fractions=None, nx=20):
    mesh, Q, H, b, s, Z, z, F = build_case(
        num_layers, law, n_vals, A_vals, fractions=fractions, nx=nx)
    bcs = [DirichletBC(Z.sub(3 * l), Constant((80.0, 0.0)), (1,))
           for l in range(num_layers)]
    problem = NonlinearVariationalProblem(F, z, bcs=bcs,
                                          form_compiler_parameters=FC)
    solver = NonlinearVariationalSolver(problem, solver_parameters=SPARAMS)
    solver.solve()
    return mesh, Q, H, b, s, z, solver.snes.getIterationNumber()


def shelf_drag(mesh, Q, H, b, s, z, haf_afloat=-100.0):
    """(max |tau_b| on ice 100 m below flotation, max |tau_b| anywhere).

    Floating cells are picked out by *geometry*, not by ``N <= 0``.  ``N``
    is the cancelling difference ``p_I - p_W``, so on a shelf it is a
    roundoff residue of either sign rather than 0, and a friction law
    gated on ``N > 0`` acts on exactly the cells an ``N <= 0`` mask drops
    -- such a mask cannot fail.  Height above flotation is built from
    ``H`` and ``b`` alone, so it says the same thing whatever ``N``'s
    last bits do.
    """
    DG = FunctionSpace(mesh, "DG", 0)
    haf = Function(DG).interpolate(height_above_flotation(H, b))
    tb = Function(DG).interpolate(sqrt(inner(z.subfunctions[2], z.subfunctions[2])))
    afloat = haf.dat.data_ro < haf_afloat
    assert afloat.any(), "geometry has no floating cells; test is vacuous"
    return (float(np.abs(tb.dat.data_ro[afloat]).max()),
            float(np.abs(tb.dat.data_ro).max()))


def build_ice_free_case(num_layers, h_jump_floor, nx=20, Lx=40e3):
    """Same slab, but calving at 0.6 Lx so the buffer beyond it is ice-free.

    ``H`` is clamped at exactly 0 rather than tapered to a positive
    minimum, which is what an ocean buffer seaward of the front actually
    is and what every other case here fails to be.  Nothing else changes:
    the surface follows the flotation rule, so the ice-free buffer has
    ``s = max(b, 0) = 0``.
    """
    mesh, Q, H, b, s, u_obs = build(nx=nx, Lx=Lx)
    x, y = SpatialCoordinate(mesh)
    H.interpolate(max_value(
        Constant(900.0) * (Constant(1.0) - x / Constant(0.6 * Lx)),
        Constant(0.0)))
    s.interpolate(max_value(b + H, Constant(1.0 - rho_I / rho_W) * H))

    n_ice_free = int((H.dat.data_ro == 0.0).sum())
    assert n_ice_free > 0, "no ice-free vertices; the test would be vacuous"

    Z = dual_function_space(mesh, num_layers)
    z = Function(Z)                                    # cold start
    F = dual_residual(
        z, Function(Q), Function(Q), H=H, s=s, b=b,
        h_layers=layer_thicknesses(H, num_layers),
        C_w0=weertman_anchor(H, s, u_obs, M_SLIDE, Q),
        A_layers=[Constant(20.0)] * num_layers,
        n_consts=[Constant(3.0)] * num_layers, n_vals=[3.0] * num_layers,
        A_lin_layers=[Constant(A_DIFFUSION)] * num_layers,
        m_slide=M_SLIDE, mesh=mesh, h_visc_floor=10.0, c_w0_floor=1e-6,
        h_jump_floor=h_jump_floor,
    )
    r = firedrake.assemble(F, form_compiler_parameters=FC)
    blocks = [np.asarray(d) for d in r.dat.data_ro]
    return n_ice_free, blocks


def test_ice_free_column_is_finite(num_layers=2):
    """H = 0 must give a finite residual -- and the floor must be why.

    The interlayer closure divides the velocity jump by ``h_above +
    h_below``, the summed thickness of the two layers meeting at the
    interface.  That is exactly 0 where there is no ice, and at a cold
    start the numerator is 0 too, so the unguarded quotient is 0/0: the
    residual is NaN before Newton takes a step and the solve dies with
    DIVERGED_FUNCTION_NANORINF at iteration 0.  ``h_jump_floor`` bounds
    that denominator away from 0 and nothing else.

    Both halves are asserted.  Finite-with-the-floor alone would pass
    just as happily if the geometry had no ice-free node, so the same
    assembly is repeated with ``h_jump_floor = 0`` and *required* to go
    NaN -- if it does not, this test is not testing the guard.
    """
    n_ice_free, blocks = build_ice_free_case(num_layers, H_JUMP_FLOOR)
    for i, blk in enumerate(blocks):
        bad = int((~np.isfinite(blk)).sum())
        assert bad == 0, (
            f"block {i} of the L={num_layers} residual has {bad} non-finite "
            f"entries at {n_ice_free} ice-free vertices")
    print(f"  L={num_layers}, {n_ice_free} ice-free vertices: all "
          f"{sum(b.size for b in blocks):,} residual entries finite")

    _, unguarded = build_ice_free_case(num_layers, 0.0)
    nan_blocks = {i: int((~np.isfinite(b)).sum())
                  for i, b in enumerate(unguarded) if not np.isfinite(b).all()}
    assert nan_blocks, (
        "h_jump_floor = 0 assembled cleanly, so the guard is not what keeps "
        "the residual finite and this test proves nothing")
    interlayer = {3 * l + 2 for l in range(1, num_layers)}
    assert set(nan_blocks) <= interlayer, (
        f"NaNs outside the interlayer-stress blocks {sorted(interlayer)}: "
        f"{nan_blocks} -- something other than the jump division divides by H")
    print(f"  with h_jump_floor = 0 the same assembly is NaN in "
          f"{', '.join(f'block {i} ({c} entries)' for i, c in nan_blocks.items())}")


def test_layer_count_guards():
    """The two entry points must agree on L, loudly rather than in MUMPS."""
    mesh, Q, H, b, s, u_obs = build(nx=6)
    Z = dual_function_space(mesh, 2)
    z = Function(Z)

    # L=2 space, L=1 thicknesses: layer 1's blocks would appear in no term
    try:
        dual_residual(
            z, Function(Q), Function(Q), H=H, s=s, b=b,
            h_layers=layer_thicknesses(H, 1), C_w0=weertman_anchor(
                H, s, u_obs, M_SLIDE, Q),
            A_layers=[Constant(20.0)], n_consts=[Constant(3.0)], n_vals=[3.0],
            m_slide=M_SLIDE, mesh=mesh,
        )
    except ValueError as e:
        assert "6-block" in str(e) and "1 layer" in str(e), str(e)
        print(f"  layer-count mismatch rejected: {str(e).splitlines()[0][:60]}...")
    else:
        raise AssertionError("L mismatch not caught; Jacobian would be singular")

    try:
        layer_thicknesses(H, 2, fractions=[0.15, 0.8])
    except ValueError as e:
        assert "sum to 1" in str(e), str(e)
        print("  fractions that do not sum to 1 rejected "
              "(0.95 would silently drop 5% of the column)")
    else:
        raise AssertionError("fractions summing to 0.95 accepted")

    layers = split_layers(z, 2)
    assert layers[0]["interlayer_stress"] is layers[0]["basal_stress"], (
        "layer 0's two stress names must be the same unknown")
    assert all("interlayer_stress" in lyr for lyr in layers), (
        "interlayer_stress must key the stress on every layer, layer 0 included")
    print("  split_layers keys the stress uniformly as interlayer_stress, "
          "with basal_stress aliasing layer 0")


def test_budd_nhat_floor():
    """Budd's PISM floor: rejected without N_ref, pinned with it.

    ``nx=200`` puts grounded cell centroids within ``nhat_floor * H`` (~8 m)
    of flotation, where the floor bites; at ``nx=40`` none did, and the
    amplification the test used to see came only from floating cells whose
    ``p_I - p_W`` cancelled to a *positive* roundoff residue, so that the
    ``gt(N, 0)`` gate inside the law passed there.  That is the
    configuration the HAF gate exists for (``He`` alone still let a cell
    floating by a few metres through, inside the He band; see
    ``friction_gate_test.py``), and the one an ``N <= 0`` mask could never
    see.
    """
    mesh, Q, H, b, s, u_obs = build(nx=200)
    C_w0 = weertman_anchor(H, s, u_obs, M_SLIDE, Q)
    theta = Function(Q)
    args = (u_obs, C_w0, theta, H, s, b, M_SLIDE)

    # with N_ref=None the reference IS N, so N_hat == 1 and the "floor"
    # would amplify instead: reject rather than quietly invert its meaning
    try:
        basal_stress(*args, law="budd", nhat_floor=0.02, nhat_cap=3.0)
    except ValueError as e:
        assert "N_ref" in str(e), str(e)
        print("  budd + nhat_floor + N_ref=None rejected "
              "(the floor has no meaning against a moving reference)")
    else:
        raise AssertionError("nhat_floor with N_ref=None accepted")
    basal_stress(*args, law="budd", nhat_floor=0.0)      # still fine at 0

    # with a frozen N_ref the branch is legitimate; pin what it does
    DG = FunctionSpace(mesh, "DG", 0)
    N = Function(DG).interpolate(effective_pressure(H, s))
    N_ref = Function(DG).assign(N)                       # frozen snapshot
    haf = Function(DG).interpolate(height_above_flotation(H, b)).dat.data_ro
    He = Function(DG).interpolate(grounded_mask(H, b)).dat.data_ro
    nhat_cap = 3.0

    tW = Function(DG).interpolate(basal_stress(*args, law="weertman")).dat.data_ro
    assert (tW > 0).all(), "tau_W must be positive everywhere"

    def budd(nhat_floor):
        return Function(DG).interpolate(
            basal_stress(*args, law="budd", N_ref=N_ref,
                         nhat_floor=nhat_floor, nhat_cap=nhat_cap)
        ).dat.data_ro.copy()

    cases = [("floor 0.02", budd(0.02)), ("floor off", budd(0.0)),
             ("floor 0.5", budd(0.5))]
    tau_b, tau_0, tau_sat = (t for _, t in cases)

    # Masks are GEOMETRIC.  Masking floating cells by N <= 0 would drop
    # precisely the cells the law's gt(N, 0) gate lets through, so an
    # assertion behind such a mask cannot fail; height above flotation is
    # built from H and b alone and says the same thing whatever the last
    # bits of the cancelling difference p_I - p_W do.
    afloat = haf < -100.0            # ten grounding-zone widths below flotation
    grounded = haf > 100.0           # He is 1 to within 1.5e-9 there
    gl_zone = (haf > 0.0) & (haf < 30.0) & (N.dat.data_ro > 0.0)   # the floor's remit: grounded, near flotation
    assert afloat.any() and grounded.any() and gl_zone.any(), (
        "geometry must straddle the grounding line")
    residue = int((N.dat.data_ro[afloat] > 0.0).sum())       # reporting only

    def nhat_on(tau, mask):
        """N_hat: what is left of tau_b once tau_W and the He gate are out."""
        return tau[mask] / (tW[mask] * He[mask])

    for label, tau in cases:
        # HAF > 0, not the sign of N, is what holds the shelf at zero
        assert np.abs(tau[afloat]).max() == 0.0, (
            f"budd ({label}) must give bit-exact zero drag 100 m below "
            f"flotation; got {np.abs(tau[afloat]).max():.3e} MPa on "
            f"{int((np.abs(tau[afloat]) > 0.0).sum())} of {afloat.sum()} cells")
        # and the same bound holds everywhere, whatever N does: the floor
        # can never lift tau_b above the grounded mask's envelope
        assert (tau <= nhat_cap * tW * He + 1e-12 * tW).all(), (
            f"budd ({label}) breaks its nhat_cap * tau_W * He envelope, so "
            f"the grounded mask is not what bounds the drag")

    # a frozen N_ref taken at this geometry gives N_hat = 1 wherever HAF > 0
    assert np.allclose(nhat_on(tau_0, grounded), 1.0), (
        f"N_ref frozen at the current geometry should give N_hat = 1, got "
        f"{nhat_on(tau_0, grounded).min()}..{nhat_on(tau_0, grounded).max()}")
    # 2 % of overburden is far below N on well-grounded ice: no effect there
    assert np.allclose(tau_b[grounded], tau_0[grounded]), (
        "the floor must be inert away from flotation")
    # near flotation it bites, amplifying N_hat and saturating at nhat_cap
    nhat = nhat_on(tau_b, gl_zone)
    assert np.allclose(nhat_on(tau_0, gl_zone), 1.0)
    assert nhat.max() > 1.0 + 1e-9, (
        "the floor should raise N_hat above 1 near flotation; it did not, so "
        "this branch is still untested")
    assert nhat.max() <= nhat_cap + 1e-9, "nhat_cap did not bind"
    assert (tau_b >= tau_0 - 1e-15 * tW).all(), "the floor must never reduce drag"
    nhat_sat = nhat_on(tau_sat, gl_zone)
    assert np.isclose(nhat_sat.max(), nhat_cap), (
        f"nhat_cap should clamp the saturated floor to {nhat_cap}, got "
        f"{nhat_sat.max()}")
    n_amplified = int((nhat > 1.0 + 1e-9).sum())
    print(f"  frozen N_ref: N_hat = 1 wherever N > 0 without the floor; with "
          f"it, {n_amplified}/{gl_zone.sum()} grounding-zone cells amplify to "
          f"max {nhat.max():.2f} (cap {nhat_cap:.1f}), and nhat_floor=0.5 "
          f"saturates at the cap ({nhat_sat.max():.2f})")
    print(f"  100 m below flotation the HAF gate holds tau_b at exactly 0 on all "
          f"{afloat.sum()} cells, floor or no floor -- including the "
          f"{residue} where p_I - p_W cancels to a positive residue and the "
          f"gt(N, 0) gate alone would have let nhat_cap * tau_W through")


def main():
    print("Generality of the dual residual builder.\n")

    print("1. single-layer space")
    test_space_matches_handbuilt()

    print("\n2. L = 1 solves as an ordinary single-layer model")
    mesh, Q, H, b, s, z, its = solve_case(1, "regularized_coulomb", [3.0], [20.0])
    sp = np.hypot(*z.subfunctions[0].dat.data_ro.T)
    assert len(z.subfunctions) == 3, "L=1 state should have exactly 3 blocks"
    print(f"  n=3 Glen, cold start, converged in {its} its; "
          f"max speed {sp.max():.1f} m/yr, {len(z.subfunctions)} blocks")

    print("\n3. every law keeps the identity tau-block at a cold start")
    test_identity_tau_block()

    print("\n4. every friction law closes and solves (L = 1, n = 3)")
    print(f"  {'law':<22} {'its':>4} {'max speed':>12} {'shelf drag':>26}")
    print("  " + "-" * 68)
    shelf, shelf_kpa = {}, {}
    for law in LAWS:
        mesh, Q, H, b, s, z, its = solve_case(1, law, [3.0], [20.0])
        sp = np.hypot(*z.subfunctions[0].dat.data_ro.T).max()
        worst, scale = shelf_drag(mesh, Q, H, b, s, z)
        shelf[law] = worst / max(scale, 1e-300)
        shelf_kpa[law] = 1e3 * worst
        print(f"  {law:<22} {its:>4} {sp:>9.1f} m/yr "
              f"{shelf_kpa[law]:>13.3e} kPa ({shelf[law]:.1e})")

    # the capped laws must be exactly zero afloat; weertman has no cap, so
    # asserting it is nonzero keeps the other two assertions meaningful
    why = ("N is clamped to max(p_I - p_W, 0) and enters tau_b as a "
           "multiplicative factor, so afloat tau_b is exactly zero and only "
           "the solve's roundoff on that zero should appear here")
    for law in SHELF_DRAG_MAX_KPA:
        assert shelf_kpa[law] < SHELF_DRAG_MAX_KPA[law], (
            f"{law} leaked drag onto floating ice: {shelf_kpa[law]:.2e} kPa "
            f"exceeds the machine-zero bound "
            f"{SHELF_DRAG_MAX_KPA[law]:g} kPa.  {why}")
        assert shelf[law] < SHELF_DRAG_MAX_REL[law], (
            f"{law} leaked drag onto floating ice: {shelf[law]:.2e} of the "
            f"grounded maximum exceeds the bound "
            f"{SHELF_DRAG_MAX_REL[law]:g}.  {why}")
    assert shelf["weertman"] > SHELF_DRAG_MIN_REL_UNCAPPED, (
        "weertman has no effective-pressure cap, so it should NOT be zero "
        "afloat; if it is, the shelf check is not discriminating")
    print("\n  capped laws are machine-zero afloat; weertman is not, as expected")

    print("\n5. L = 2 composite still works through the same entry point")
    mesh, Q, H, b, s, z, its = solve_case(
        2, "regularized_coulomb", [4.0, 1.8], [46.0, 0.45], fractions=[0.15, 0.85])
    sp_b = np.hypot(*z.subfunctions[0].dat.data_ro.T)
    sp_t = np.hypot(*z.subfunctions[3].dat.data_ro.T)
    assert len(z.subfunctions) == 6
    print(f"  n=4/1.8, cold start, converged in {its} its; "
          f"shear {np.abs(sp_t - sp_b).max():.1f} m/yr")

    print("\n6. an ice-free column still assembles a finite residual")
    test_ice_free_column_is_finite()

    print("\n7. the entry points police their own layer count")
    test_layer_count_guards()

    print("\n8. budd's PISM floor only where it means what it says")
    test_budd_nhat_floor()

    print("\nPASS: one builder covers L=1 and L>1 and all three friction laws")


if __name__ == "__main__":
    main()
