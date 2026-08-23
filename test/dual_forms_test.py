r"""Is the builder actually general, or just the multilayer case in disguise?

Three claims the other icepack repos would rely on:

  * ``dual_function_space(mesh, 1)`` is the ordinary single-layer dual
    space.  The single-layer consumers build ``Z = V * Sigma * T`` by hand
    (see ``ismip7/antarctica/scripts/diagnostic_solve.py``); this checks
    the helper reproduces it element for element, so adopting the helper
    is not a silent change of discretisation.
  * ``dual_residual`` at ``L = 1`` solves a single-layer problem, with the
    interlayer machinery inert rather than merely unused.
  * all three friction laws close and solve, and the two with an
    effective-pressure cap give *exactly* zero drag on floating ice while
    Weertman -- correctly, since it has no cap -- does not.
  * the two entry points that must agree on ``L`` say so when they do
    not, and Budd's PISM floor is only accepted where it means what it
    says (against a frozen ``N_ref``), where its amplification and its
    grounding-line jump are then pinned.

    python -u dual_forms_test.py
"""
import numpy as np
import firedrake
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace, TensorFunctionSpace,
    SpatialCoordinate, DirichletBC, NonlinearVariationalProblem,
    NonlinearVariationalSolver, as_vector, inner, sqrt, max_value,
)

from icepack_tools.constants import ice_density as rho_I, water_density as rho_W
from icepack_tools.friction import basal_stress, weertman_anchor, LAWS
from icepack_tools.grounding import effective_pressure
from icepack_tools.momentum import dual_residual
from icepack_tools.spaces import (
    dual_function_space, layer_thicknesses, split_layers,
)
from icepack_tools.viscosity import A_DIFFUSION

M_SLIDE = 3.0
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


def solve_case(num_layers, law, n_vals, A_vals, fractions=None, nx=20):
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
    bcs = [DirichletBC(Z.sub(3 * l), Constant((80.0, 0.0)), (1,))
           for l in range(num_layers)]
    problem = NonlinearVariationalProblem(F, z, bcs=bcs,
                                          form_compiler_parameters=FC)
    solver = NonlinearVariationalSolver(problem, solver_parameters=SPARAMS)
    solver.solve()
    return mesh, Q, H, s, z, solver.snes.getIterationNumber()


def shelf_drag(mesh, Q, H, s, z):
    """(max |tau_b| on cells where N == 0, max |tau_b| anywhere)."""
    DG = FunctionSpace(mesh, "DG", 0)
    N = Function(DG).interpolate(effective_pressure(H, s))
    tb = Function(DG).interpolate(sqrt(inner(z.subfunctions[2], z.subfunctions[2])))
    afloat = N.dat.data_ro <= 0.0
    assert afloat.any(), "geometry has no floating cells; test is vacuous"
    return (float(np.abs(tb.dat.data_ro[afloat]).max()),
            float(np.abs(tb.dat.data_ro).max()))


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
    """Budd's PISM floor: rejected without N_ref, pinned with it."""
    mesh, Q, H, b, s, u_obs = build(nx=20)
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
    nhat_cap = 3.0

    tau_W = Function(DG).interpolate(basal_stress(*args, law="weertman"))
    tau_b = Function(DG).interpolate(
        basal_stress(*args, law="budd", N_ref=N_ref, nhat_floor=0.02,
                     nhat_cap=nhat_cap))
    tau_0 = Function(DG).interpolate(
        basal_stress(*args, law="budd", N_ref=N_ref, nhat_floor=0.0,
                     nhat_cap=nhat_cap))

    assert (tau_W.dat.data_ro > 0).all(), "tau_W must be positive everywhere"
    nhat = tau_b.dat.data_ro / tau_W.dat.data_ro
    nhat_0 = tau_0.dat.data_ro / tau_W.dat.data_ro
    afloat = N.dat.data_ro <= 0.0
    # above the 1e-6 denominator floor, so N/Nr is exactly 1 for a frozen N_ref
    grounded = N.dat.data_ro >= 1e-6
    assert afloat.any() and grounded.any(), "geometry must straddle the GL"

    # gt(N, 0) drops drag to exactly zero afloat, floor or no floor
    assert np.abs(tau_b.dat.data_ro[afloat]).max() == 0.0, (
        "budd must give bit-exact zero drag afloat even with the floor on")
    # without the floor, a frozen N_ref taken at this geometry gives N_hat = 1
    assert np.allclose(nhat_0[grounded], 1.0), (
        f"N_ref frozen at the current geometry should give N_hat = 1, got "
        f"{nhat_0[grounded].min()}..{nhat_0[grounded].max()}")
    # with it on, near-flotation cells are AMPLIFIED, saturating at nhat_cap
    assert nhat[grounded].max() > 1.0 + 1e-9, (
        "the floor should raise N_hat above 1 near flotation; it did not, so "
        "this branch is still untested")
    assert nhat[grounded].max() <= nhat_cap + 1e-9, "nhat_cap did not bind"
    assert (nhat[grounded] >= nhat_0[grounded] - 1e-12).all(), (
        "the floor must never reduce N_hat")
    n_amplified = int((nhat[grounded] > 1.0 + 1e-9).sum())
    assert n_amplified > 0
    # a floor big enough to saturate pins the nhat_cap branch itself
    tau_sat = Function(DG).interpolate(
        basal_stress(*args, law="budd", N_ref=N_ref, nhat_floor=0.5,
                     nhat_cap=nhat_cap))
    nhat_sat = tau_sat.dat.data_ro / tau_W.dat.data_ro
    assert np.isclose(nhat_sat[grounded].max(), nhat_cap), (
        f"nhat_cap should clamp the saturated floor to {nhat_cap}, got "
        f"{nhat_sat[grounded].max()}")
    assert np.abs(tau_sat.dat.data_ro[afloat]).max() == 0.0, (
        "the cap must not resurrect drag afloat")
    # ...and then fall off a cliff at the grounding line
    print(f"  frozen N_ref: N_hat = 1 on grounded ice without the floor; "
          f"with it, {n_amplified}/{grounded.sum()} grounded cells amplify to "
          f"max {nhat[grounded].max():.2f} (cap {nhat_cap:.1f}), then drop "
          f"discontinuously to 0 afloat; nhat_floor=0.5 saturates at the "
          f"cap ({nhat_sat[grounded].max():.2f}) and is still 0 afloat")


def main():
    print("Generality of the dual residual builder.\n")

    print("1. single-layer space")
    test_space_matches_handbuilt()

    print("\n2. L = 1 solves as an ordinary single-layer model")
    mesh, Q, H, s, z, its = solve_case(1, "regularized_coulomb", [3.0], [20.0])
    sp = np.hypot(*z.subfunctions[0].dat.data_ro.T)
    assert len(z.subfunctions) == 3, "L=1 state should have exactly 3 blocks"
    print(f"  n=3 Glen, cold start, converged in {its} its; "
          f"max speed {sp.max():.1f} m/yr, {len(z.subfunctions)} blocks")

    print("\n3. every friction law closes and solves (L = 1, n = 3)")
    print(f"  {'law':<22} {'its':>4} {'max speed':>12} {'shelf drag':>26}")
    print("  " + "-" * 68)
    shelf = {}
    for law in LAWS:
        mesh, Q, H, s, z, its = solve_case(1, law, [3.0], [20.0])
        sp = np.hypot(*z.subfunctions[0].dat.data_ro.T).max()
        worst, scale = shelf_drag(mesh, Q, H, s, z)
        shelf[law] = worst / max(scale, 1e-300)
        print(f"  {law:<22} {its:>4} {sp:>9.1f} m/yr "
              f"{1e3*worst:>13.3e} kPa ({shelf[law]:.1e})")

    # the capped laws must be exactly zero afloat; weertman has no cap, so
    # asserting it is nonzero keeps the other two assertions meaningful
    for law in ("regularized_coulomb", "budd"):
        assert shelf[law] < 1e-12, f"{law} leaked drag onto floating ice"
    assert shelf["weertman"] > 1e-6, (
        "weertman has no effective-pressure cap, so it should NOT be zero "
        "afloat; if it is, the shelf check is not discriminating")
    print("\n  capped laws are machine-zero afloat; weertman is not, as expected")

    print("\n4. L = 2 composite still works through the same entry point")
    mesh, Q, H, s, z, its = solve_case(
        2, "regularized_coulomb", [4.0, 1.8], [46.0, 0.45], fractions=[0.15, 0.85])
    sp_b = np.hypot(*z.subfunctions[0].dat.data_ro.T)
    sp_t = np.hypot(*z.subfunctions[3].dat.data_ro.T)
    assert len(z.subfunctions) == 6
    print(f"  n=4/1.8, cold start, converged in {its} its; "
          f"shear {np.abs(sp_t - sp_b).max():.1f} m/yr")

    print("\n5. the entry points police their own layer count")
    test_layer_count_guards()

    print("\n6. budd's PISM floor only where it means what it says")
    test_budd_nhat_floor()

    print("\nPASS: one builder covers L=1 and L>1 and all three friction laws")


if __name__ == "__main__":
    main()
