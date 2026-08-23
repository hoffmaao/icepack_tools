r"""Does the residual form solve from a cold start?

The claim being tested: closing the basal stress as a residual makes its
Jacobian block the identity, so a two-layer composite problem on a
Coulomb bed solves

  * with the sliding exponent ``m`` FIXED at its target -- no
    m-continuation, which the potential form cannot do at all because its
    ``-ln(1 - r^(m+1))`` is singular at the Coulomb limit,
  * from a cold start ``z = 0``, no warm start,

on a geometry with both grounded ice and a floating tongue thinning to
nothing.

What removes the remaining *viscous* ``n``-continuation is the question
the 2x2 matrix below answers: (diffusion creep on/off) x (``n`` ramped
1 -> n / set directly at its target).  Without a real ``n = 1``
mechanism the creep term contributes nothing at ``M = 0``, only the tiny
``alpha`` regulariser balances the strain-rate coupling, the linearised
viscosity is absurd and the direct solve diverges -- so that divergence
is asserted, because the negative result is the evidence.  Adding
diffusion creep makes the direct solve converge, and it is checked
against the ramped one to confirm it lands on the same solution rather
than somewhere else.

    python -u multilayer_rc_test.py
"""
import numpy as np
import firedrake
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace, SpatialCoordinate,
    DirichletBC, NonlinearVariationalProblem, NonlinearVariationalSolver,
    as_vector, inner, sqrt, max_value,
)
from firedrake.exceptions import ConvergenceError

from icepack_tools.constants import ice_density as rho_I, water_density as rho_W
from icepack_tools.friction import weertman_anchor
from icepack_tools.grounding import effective_pressure, height_above_flotation
from icepack_tools.momentum import multilayer_rc_residual
from icepack_tools.viscosity import A_DIFFUSION

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


def solve_case(A_lin_layers, ramp):
    """One (diffusion on/off) x (n ramped/direct) case.

    Returns ``(its, z, mesh, H, b, s)``: the total Newton count, the solved
    mixed state and the geometry the diagnostics below need.
    """
    mesh, Q, V, H, b, s, u_obs, Lx = build()
    from multilayer.model.utilities import create_function_space, layer_thicknesses
    Z = create_function_space(mesh, len(FRACTIONS))
    h_layers = layer_thicknesses(H, len(FRACTIONS), fractions=FRACTIONS)
    C_w0 = weertman_anchor(H, s, u_obs, M_SLIDE, Q)

    theta, phi = Function(Q, name="theta"), Function(Q, name="phi")
    # NOT [Constant(1.0)] * len(N_VALS): that repeats one object, so
    # assigning per-layer exponents would set every layer to the last one.
    n_consts = ([Constant(1.0) for _ in N_VALS] if ramp
                else [Constant(v) for v in N_VALS])
    A_layers = [Constant(A_VALS[0]), Constant(A_VALS[1]) * firedrake.exp(phi)]

    z = Function(Z)                       # COLD START
    F = multilayer_rc_residual(
        z, theta, phi, H=H, s=s, b=b, h_layers=h_layers, C_w0=C_w0,
        A_layers=A_layers, n_consts=n_consts, n_vals=N_VALS,
        A_lin_layers=A_lin_layers,
        m_slide=M_SLIDE, mesh=mesh, layer_fractions=FRACTIONS,
        h_visc_floor=10.0, c_w0_floor=1e-6,
    )
    bcs = [DirichletBC(Z.sub(3 * l), Constant((80.0, 0.0)), (1,))
           for l in range(len(FRACTIONS))]
    problem = NonlinearVariationalProblem(F, z, bcs=bcs,
                                          form_compiler_parameters=FC)
    solver = NonlinearVariationalSolver(problem, solver_parameters=SPARAMS)

    its = 0
    for lam in (np.linspace(0.0, 1.0, N_RAMP) if ramp else [1.0]):
        if ramp:
            for c, target in zip(n_consts, N_VALS):
                c.assign(1.0 + lam * (target - 1.0))
        solver.solve()
        its += solver.snes.getIterationNumber()
    return its, z, mesh, H, b, s


def main():
    print("Two-layer composite (n = 4 / 1.8) on a regularised-Coulomb bed.")
    print("m is FIXED at its target in every case -- that is what the")
    print("residual closure buys.  The question is what removes the")
    print("*viscous* n-continuation.\n")
    print(f"  {'diffusion (n=1)':<18} {'n':<10} {'result':<26} {'max speed'}")
    print("  " + "-" * 68)

    off = [None for _ in N_VALS]
    on = [Constant(A_DIFFUSION) for _ in N_VALS]
    results = {}
    for A_lin_layers, tag in ((off, "off"), (on, f"{A_DIFFUSION:g}")):
        for ramp, rtag in ((True, "ramped 1->n"), (False, "direct")):
            try:
                its, z, mesh, H, b, s = solve_case(A_lin_layers, ramp)
                sp = np.hypot(*z.subfunctions[3].dat.data_ro.T).max()
                print(f"  {tag:<18} {rtag:<10} {'converged, ' + str(its) + ' its':<26} "
                      f"{sp:8.1f} m/yr")
                results[(tag, rtag)] = (its, z, mesh, H, b, s)
            except ConvergenceError as exc:
                # Deliberately narrow: the ("off", "direct") divergence
                # asserted below is this test's headline negative result,
                # so anything that is NOT a convergence failure -- a UFL
                # compilation error on the non-ramped path, a MUMPS
                # allocation failure, an OOM -- must propagate and fail
                # loudly rather than masquerade as the expected outcome.
                print(f"  {tag:<18} {rtag:<10} {'DIVERGED':<26}")
                print(f"    {' '.join(str(exc).split())[:160]}")
                results[(tag, rtag)] = None

    assert results[("off", "ramped 1->n")], "ramped path must work without diffusion"
    assert results[("off", "direct")] is None, (
        "expected the direct solve to fail without a linear mechanism")
    assert results[(f"{A_DIFFUSION:g}", "direct")], (
        "diffusion creep should remove the n-continuation")

    # both paths must reach the same solution -- the direct solve is not a
    # shortcut to somewhere else
    a = np.hypot(*results[(f"{A_DIFFUSION:g}", "ramped 1->n")][1]
                 .subfunctions[3].dat.data_ro.T)
    d = np.hypot(*results[(f"{A_DIFFUSION:g}", "direct")][1]
                 .subfunctions[3].dat.data_ro.T)
    rel = np.abs(a - d).max() / max(a.max(), 1e-300)
    print(f"\n  ramped vs direct agree to {rel:.2e} (relative, max speed)")
    assert rel < 1e-6, "ramped and direct solves disagree"

    # inspect the direct composite solution
    _, z, mesh, H, b, s = results[(f"{A_DIFFUSION:g}", "direct")]
    u_b, u_t, tau = z.subfunctions[0], z.subfunctions[3], z.subfunctions[2]
    sp_b = np.hypot(*u_b.dat.data_ro.T)
    sp_t = np.hypot(*u_t.dat.data_ro.T)
    print(f"\n  bottom speed [{sp_b.min():7.1f}, {sp_b.max():7.1f}] m/yr")
    print(f"  top    speed [{sp_t.min():7.1f}, {sp_t.max():7.1f}] m/yr")
    print(f"  vertical shear max {np.abs(sp_t - sp_b).max():.1f} m/yr")

    DG = FunctionSpace(mesh, "DG", 0)
    N_dg = Function(DG).interpolate(effective_pressure(H, s))
    haf_dg = Function(DG).interpolate(height_above_flotation(H, b))
    tb_dg = Function(DG).interpolate(sqrt(inner(tau, tau)))
    # Geometry, not N <= 0: N is the cancelling difference p_I - p_W, so on
    # a shelf it is a roundoff residue rather than 0, and an N <= 0 mask
    # drops exactly the cells a friction law gated on N > 0 can act on.
    afloat = haf_dg.dat.data_ro < -100.0
    print(f"  cells 100 m below flotation: {int(afloat.sum())} "
          f"({int((N_dg.dat.data_ro[afloat] > 0.0).sum())} of them with N > 0 "
          f"by roundoff)")
    # Guarded: .max() on an empty selection raises, and the slab geometry
    # could be retuned so the grounding line leaves the domain.  Likewise
    # floor the denominator, so a degenerate all-zero stress field trips
    # the assertion below rather than a ZeroDivisionError in this print.
    assert afloat.any(), "no floating cells -- the grounding line left the domain"
    worst = float(np.abs(tb_dg.dat.data_ro[afloat]).max())
    scale = max(float(np.abs(tb_dg.dat.data_ro).max()), 1e-300)
    rel = worst / scale
    print(f"  max |tau_b| there: {1e3*worst:.3e} kPa "
          f"({rel:.1e} of grounded max -- machine zero)")
    assert rel < 1e-12, f"basal drag leaked onto floating ice: {rel:.2e}"

    print("\nPASS: with diffusion creep the composite solves cold and direct,")
    print("      no continuation in n and none in m")


if __name__ == "__main__":
    main()
