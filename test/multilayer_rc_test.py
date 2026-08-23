r"""Does the residual form solve from a cold start?

The claim being tested: closing the basal stress as a residual makes its
Jacobian block the identity, so a two-layer composite problem on a
Coulomb bed solves

  * with the sliding exponent ``m`` FIXED at its target -- no
    m-continuation, which the potential form cannot do at all because its
    ``-ln(1 - r^(m+1))`` is singular at the Coulomb limit,
  * from a cold start ``z = 0``, no warm start,

on a geometry with both grounded ice and a floating tongue thinning to
nothing.  The flow-law exponents ``n`` still ramp 1 -> n: at ``M = 0``
the creep term contributes nothing, so only the small linear regulariser
balances the strain-rate coupling and the linearised viscosity is absurd.
ismip7 ramps ``n_flow`` for the same reason.

    python -u multilayer_rc_test.py
"""
import numpy as np
import firedrake
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace, SpatialCoordinate,
    DirichletBC, NonlinearVariationalProblem, NonlinearVariationalSolver,
    as_vector, assemble, dx, inner, sqrt, max_value, min_value,
)

from icepack_tools.constants import ice_density as rho_I, water_density as rho_W
from icepack_tools.friction import weertman_anchor
from icepack_tools.grounding import effective_pressure, grounded_mask
from icepack_tools.momentum import multilayer_rc_residual

# two layers, Store-like composite rheology
FRACTIONS = [0.15, 0.85]
N_VALS = [4.0, 1.8]
A_VALS = [46.0, 0.45]
M_SLIDE = 3.0
N_RAMP = 6

SPARAMS = {
    "snes_type": "newtonls", "snes_max_it": 100,
    "snes_linesearch_type": "nleqerr",
    # A cold start is far from the solution, so the first Newton step is
    # large and the residual can climb before it falls.  Leaving PETSc's
    # default divergence tolerance in place aborts on that first step.
    "snes_divergence_tolerance": -1,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    # The dual form is a saddle point: the momentum equation has no direct
    # velocity dependence, so the whole (u, u) diagonal is zero.  MUMPS
    # needs null-pivot detection switched on (icntl_24) to factor that;
    # without it the solve returns garbage rather than failing loudly.
    "mat_mumps_icntl_14": 200,
    "mat_mumps_icntl_24": 1,
    "mat_mumps_cntl_3": 1e-6,
}
FC = {"quadrature_degree": 4}


def build(nx=24, Lx=40e3, Ly=12e3):
    """Inclined slab: grounded inland, marine and thinning toward x = Lx."""
    mesh = firedrake.RectangleMesh(nx, max(4, nx // 3), Lx, Ly,
                                   diagonal="crossed")
    x, y = SpatialCoordinate(mesh)
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)

    # thickness tapers 900 -> 60 m; bed drops from +100 to -700 m, so the
    # downstream end floats and the grounding line sits inside the domain
    H = Function(Q, name="thickness").interpolate(
        Constant(900.0) - Constant(840.0) * x / Constant(Lx))
    b = Function(Q, name="bed").interpolate(
        Constant(100.0) - Constant(800.0) * x / Constant(Lx))
    s = Function(Q, name="surface").interpolate(
        max_value(b + H, Constant(1.0 - rho_I / rho_W) * H))

    # a plausible observed velocity: accelerating downstream
    u_obs = Function(V, name="velocity").interpolate(
        as_vector([Constant(80.0) + Constant(900.0) * x / Constant(Lx),
                   Constant(0.0)]))
    return mesh, Q, V, H, b, s, u_obs, Lx


def main():
    mesh, Q, V, H, b, s, u_obs, Lx = build()

    from multilayer.model.utilities import create_function_space, layer_thicknesses
    Z = create_function_space(mesh, len(FRACTIONS))
    h_layers = layer_thicknesses(H, len(FRACTIONS), fractions=FRACTIONS)

    He = Function(Q).interpolate(grounded_mask(H, b))
    N = Function(Q).interpolate(effective_pressure(H, s))
    print(f"mesh {mesh.num_vertices():,} verts, dual {Z.dim():,} DOFs")
    print(f"  grounded fraction He: [{He.dat.data_ro.min():.3f}, "
          f"{He.dat.data_ro.max():.3f}]  (needs both ends -> GL is interior)")
    print(f"  effective pressure N: [{N.dat.data_ro.min():.4f}, "
          f"{N.dat.data_ro.max():.3f}] MPa")
    n_afloat = int((N.dat.data_ro <= 0.0).sum())
    print(f"  nodes with N == 0 exactly (frictionless): {n_afloat}")

    C_w0 = weertman_anchor(H, s, u_obs, M_SLIDE, Q)
    print(f"  C_w0: [{C_w0.dat.data_ro.min():.4f}, {C_w0.dat.data_ro.max():.3f}]")

    theta = Function(Q, name="theta")            # zero: balanced by construction
    phi = Function(Q, name="phi")                # zero
    # m is FIXED at its target -- that is what the residual closure buys.
    # n still ramps: at M = 0 the creep term contributes nothing, so only
    # the small linear regulariser balances the strain-rate coupling and
    # the linearised viscosity is absurd.  ismip7 does the same (n_flow is
    # a mutable continuation Constant there).
    n_consts = [Constant(1.0), Constant(1.0)]
    A_layers = [Constant(A_VALS[0]),
                Constant(A_VALS[1]) * firedrake.exp(phi)]

    # COLD START: z is identically zero, including both stress blocks
    z = Function(Z)
    assert np.allclose(z.dat.data_ro[0], 0.0)

    F = multilayer_rc_residual(
        z, theta, phi, H=H, s=s, b=b, h_layers=h_layers, C_w0=C_w0,
        A_layers=A_layers, n_consts=n_consts, n_vals=N_VALS,
        m_slide=M_SLIDE, mesh=mesh, layer_fractions=FRACTIONS,
        h_visc_floor=10.0, c_w0_floor=1e-6,
    )
    bcs = [DirichletBC(Z.sub(3 * l), Constant((80.0, 0.0)), (1,))
           for l in range(len(FRACTIONS))]

    problem = NonlinearVariationalProblem(F, z, bcs=bcs,
                                          form_compiler_parameters=FC)
    solver = NonlinearVariationalSolver(problem, solver_parameters=SPARAMS)

    print(f"\nsolving from a cold start: m = {M_SLIDE} FIXED throughout, "
          f"n ramped 1 -> {N_VALS}")
    total_its = 0
    for lam in np.linspace(0.0, 1.0, N_RAMP):
        for c, target in zip(n_consts, N_VALS):
            c.assign(1.0 + lam * (target - 1.0))
        solver.solve()
        total_its += solver.snes.getIterationNumber()
        sp = np.hypot(*z.subfunctions[3].dat.data_ro.T)
        print(f"   n = {float(n_consts[0]):4.2f}/{float(n_consts[1]):4.2f}   "
              f"max speed {sp.max():8.1f} m/yr   "
              f"({solver.snes.getIterationNumber()} its)")
    print(f"  converged, {total_its} Newton iterations total")

    u_b, u_t = z.subfunctions[0], z.subfunctions[3]
    tau = z.subfunctions[2]
    sp_b = np.hypot(*u_b.dat.data_ro.T)
    sp_t = np.hypot(*u_t.dat.data_ro.T)
    tb = np.hypot(*tau.dat.data_ro.T) * 1e3

    print(f"\n  bottom speed [{sp_b.min():7.1f}, {sp_b.max():7.1f}] m/yr")
    print(f"  top    speed [{sp_t.min():7.1f}, {sp_t.max():7.1f}] m/yr")
    print(f"  shear  (top - bottom) max {np.abs(sp_t - sp_b).max():.1f} m/yr")
    print(f"  |tau_b|      [{tb.min():7.3f}, {tb.max():7.1f}] kPa")

    # The headline property: drag is exactly zero where the ice floats.
    # Check it cell-wise in DG0 -- projecting the DG0 stress to CG1 smears
    # grounded values onto adjacent floating nodes and measures the
    # projection, not the model.
    DG = FunctionSpace(mesh, "DG", 0)
    N_dg = Function(DG).interpolate(effective_pressure(H, s))
    tb_dg = Function(DG).interpolate(sqrt(inner(tau, tau)))
    afloat = N_dg.dat.data_ro <= 0.0
    print(f"\n  floating cells (N == 0 exactly): {int(afloat.sum())}")
    if afloat.any():
        worst = float(np.abs(tb_dg.dat.data_ro[afloat]).max())
        scale = float(np.abs(tb_dg.dat.data_ro).max())
        rel = worst / max(scale, 1e-300)
        print(f"  max |tau_b| there: {1e3*worst:.3e} kPa "
              f"({rel:.1e} of the grounded max -- machine zero)")
        # Compare with a phi_eff floor of 0.01, which leaves ~1% of the
        # grounded drag on every shelf node.
        assert rel < 1e-12, f"basal drag leaked onto floating ice: {rel:.2e}"
    assert np.isfinite(sp_t).all(), "non-finite velocity"
    assert sp_t.max() < 1e5, "velocity runaway"
    print("\nPASS: cold start converged with m fixed at its target throughout")


if __name__ == "__main__":
    main()
