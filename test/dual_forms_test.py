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
from icepack_tools.friction import weertman_anchor, LAWS
from icepack_tools.grounding import effective_pressure
from icepack_tools.momentum import dual_residual
from icepack_tools.spaces import dual_function_space, layer_thicknesses
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

    print("\nPASS: one builder covers L=1 and L>1 and all three friction laws")


if __name__ == "__main__":
    main()
