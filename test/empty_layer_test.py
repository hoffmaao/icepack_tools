r"""An empty layer leaves no trace.

A layer of zero thickness should not be there at all: the solution with
it must be the solution without it, whatever rheology it was given.
Under the package's inherited conventions it is not.  An interface is
closed with the law of the layer *below* it, so an empty basal layer's
rheology still governs the shear above it; and ``h_visc_floor`` is
applied per layer, so an empty layer under thick ice keeps a floor's
worth of membrane, in its own rheology.  ``dual_residual(interface=
"series", h_visc_floor_on="column")`` is the consistent choice, and these
checks are what "consistent" means, each solved cold on the inclined
slab of ``multilayer_rc_test.py``:

  * two layers of one rheology: the series closure IS the old one,
  * an empty top layer: the two-layer solve is the one-layer solve,
    whatever rheology the empty layer carries,
  * an empty middle layer: the three-layer solve is the two-layer solve
    to discretisation accuracy -- the empty layer's momentum balance
    passes the stress through it, but only in the CG1 projection that
    balance is tested in, so its two interface stresses agree in
    projection and not cell by cell,
  * an empty bottom layer: its rheology does not reach the solution.  It
    is not the one-layer model, and should not be -- the sliding velocity
    under the column is its own unknown, with the upper ice's half-layer
    of shear between it and the layer above,
  * the old conventions leave a trace in each of the last three, or do not
    converge at all, so the checks discriminate,
  * a layer thinning to nothing converges to the empty-layer solve, at
    first order in its thickness for the ice above it.  The empty layer's
    own velocity, the sliding velocity under the column, does not: with
    no membrane to couple it across cells it carries cell-scale roughness
    that any real thickness damps (the thinning test measures it).

Both remaining gaps come from where the basal and interlayer stresses
live: one value per cell, closed on cell means, while the balances that
tie them are tested with vertex functions.  With those stresses at the
vertices instead (``dual_function_space(stress_family="CG")``, closures
collocated there) an empty layer's balance equates its two interface
stresses node by node and the empty middle layer is exact, and the
sliding velocity is determined node by node and converges with the ice.
The last test runs the whole benchmark on that variant.

    python -u empty_layer_test.py     (or pytest)
"""
import functools

import numpy as np
import firedrake
from firedrake import (
    Constant, DirichletBC, Function, FunctionSpace, VectorFunctionSpace,
    NonlinearVariationalProblem, NonlinearVariationalSolver,
    SpatialCoordinate, as_vector, max_value,
)

from icepack_tools.constants import ice_density as rho_I, water_density as rho_W
from icepack_tools.friction import weertman_anchor
from icepack_tools.momentum import dual_residual
from icepack_tools.spaces import dual_function_space
from icepack_tools.viscosity import A_DIFFUSION

#: (A, n) of the rheologies the layers are built from: a temperate n = 4
#: base, a grain-boundary-sliding n = 1.8 body, and Glen's n = 3 for a
#: third law that must leave no trace.
TEMPERATE, GBS, GLEN = (46.0, 4.0), (0.45, 1.8), (5.0, 3.0)
M_SLIDE = 3.0
H_VISC_FLOOR = 10.0
FC = {"quadrature_degree": 4}
# as multilayer_rc_test.py: the (u, u) diagonal of the dual system is
# zero, so MUMPS needs null-pivot detection to factor it
SOLVER = {
    "snes_type": "newtonls", "snes_linesearch_type": "nleqerr",
    "snes_max_it": 200, "snes_rtol": 1e-12, "snes_atol": 1e-11,
    "snes_stol": 0.0, "snes_divergence_tolerance": -1,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": 200,
    "mat_mumps_icntl_24": 1, "mat_mumps_cntl_3": 1e-6,
}
#: Two solves of the same problem on different mixed spaces agree to the
#: solver's tolerance; a convention that leaves a trace moves the velocity
#: by far more than this.
SAME = 1e-7
#: And what counts as a trace: a relative difference this large or more.
TRACE = 1e-3
#: What an empty middle layer costs: the velocities and basal stress, and
#: the membrane stresses, of the three-layer solve against the two-layer
#: one, on the test slab (measured 2e-5 and 9e-4).
DISCRETE_U, DISCRETE_M = 1e-3, 1e-2


def build(nx=24, Lx=40e3, Ly=12e3):
    """Inclined slab: grounded inland, marine and thinning toward x = Lx."""
    mesh = firedrake.RectangleMesh(nx, max(4, nx // 3), Lx, Ly, diagonal="crossed")
    x, y = SpatialCoordinate(mesh)
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    H = Function(Q).interpolate(Constant(900.0) - Constant(840.0) * x / Constant(Lx))
    b = Function(Q).interpolate(Constant(100.0) - Constant(800.0) * x / Constant(Lx))
    s = Function(Q).interpolate(max_value(b + H, Constant(1.0 - rho_I / rho_W) * H))
    u_obs = Function(V).interpolate(
        as_vector([Constant(80.0) + Constant(900.0) * x / Constant(Lx), Constant(0.0)]))
    return mesh, Q, H, b, s, u_obs


@functools.lru_cache(maxsize=None)
def solve(fractions, laws, interface, floor_on, tau="DG"):
    """Cold solve of a column: ``fractions`` of ``H`` per layer (0 for an
    empty layer), ``laws`` the (A, n) of each, ``tau`` where the basal and
    interlayer stresses live.  Returns the solved fields as arrays,
    ``{"u0", "M0", "S0", "u1", ...}``, bottom first."""
    mesh, Q, H, b, s, u_obs = build()
    L = len(fractions)
    Z = dual_function_space(mesh, L, stress_family=tau)
    h_layers = [Constant(f) * H for f in fractions]
    C_w0 = weertman_anchor(H, s, u_obs, M_SLIDE, Q)
    theta, phi = Function(Q), Function(Q)
    z = Function(Z)
    F = dual_residual(
        z, theta, phi, H=H, s=s, b=b, h_layers=h_layers, C_w0=C_w0,
        A_layers=[Constant(A) for A, _ in laws],
        n_consts=[Constant(n) for _, n in laws], n_vals=[n for _, n in laws],
        A_lin_layers=[Constant(A_DIFFUSION)] * L,
        m_slide=M_SLIDE, mesh=mesh, h_visc_floor=H_VISC_FLOOR, c_w0_floor=1e-6,
        interface=interface, h_visc_floor_on=floor_on,
    )
    bcs = [DirichletBC(Z.sub(3 * l), Constant((80.0, 0.0)), (1,)) for l in range(L)]
    solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(F, z, bcs=bcs, form_compiler_parameters=FC),
        solver_parameters=SOLVER)
    solver.solve()
    out = {}
    for l in range(L):
        for k, name in enumerate(("u", "M", "S")):
            out[f"{name}{l}"] = z.subfunctions[3 * l + k].dat.data_ro.copy()
    return out


def rel(a, b):
    """Relative max-norm difference of two field arrays."""
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def report(label, pairs, same, tol=SAME):
    """Print and check: every pair agrees to ``tol`` (``same``) or some
    pair shows a trace (``not same``)."""
    worst = max(rel(a, b) for a, b in pairs.values())
    detail = ", ".join(f"{k} {rel(a, b):.1e}" for k, (a, b) in pairs.items())
    print(f"  {label:<58} worst {worst:.1e}   [{detail}]")
    if same:
        assert worst < tol, f"{label}: differ by {worst:.2e}"
    else:
        assert worst > TRACE, f"{label}: expected a trace, saw {worst:.2e}"
    return worst


def old_convention(label, cases, pairs_of):
    """The inherited conventions must leave a trace -- or fail to solve at
    all, which an empty layer carrying a floor's worth of membrane with
    nothing to drive it can do.  ``cases`` are ``(fractions, laws)`` to
    solve under them; ``pairs_of`` builds the field pairs from the solves."""
    solved = []
    for fractions, laws in cases:
        try:
            solved.append(solve(fractions, laws, "below", "layer"))
        except firedrake.ConvergenceError as exc:
            print(f"  {label:<58} does not converge "
                  f"({str(exc).splitlines()[-1].strip()})")
            return
    report(label, pairs_of(*solved), False)


def test_uniform_rheology_series_is_below():
    """Two layers of one law: the series closure reproduces the old one."""
    old = solve((0.3, 0.7), (GLEN, GLEN), "below", "layer")
    new = solve((0.3, 0.7), (GLEN, GLEN), "series", "layer")
    report("uniform n = 3: series vs below", {k: (new[k], old[k]) for k in old}, True)


def test_empty_top_layer_is_one_layer():
    """[H, 0] with series + column floor is [H]; the empty layer's law
    (temperate or Glen) is irrelevant; the old conventions leave a trace."""
    one = solve((1.0,), (GBS,), "below", "layer")
    keys = {"u0": "u0", "M0": "M0", "S0": "S0"}
    for empty in (TEMPERATE, GLEN):
        two = solve((1.0, 0.0), (GBS, empty), "series", "column")
        report(f"[H, 0] empty top (A, n) = {empty} vs [H]",
               {k: (two[k], one[v]) for k, v in keys.items()}, True)
    old_convention("[H, 0] under the old conventions vs [H]",
                   [((1.0, 0.0), (GBS, TEMPERATE))],
                   lambda old: {k: (old[k], one[v]) for k, v in keys.items()})


def test_empty_middle_layer_drops_out():
    """[0.3H, 0, 0.7H] is [0.3H, 0.7H] to discretisation accuracy: both
    interfaces close on the ice beside the empty layer, and its momentum
    balance passes the stress through -- in the CG1 projection the balance
    is tested in, so the two interface stresses agree in projection and
    not cell by cell, and the velocities match to about 1e-5 on this slab
    rather than to solver precision."""
    two = solve((0.3, 0.7), (TEMPERATE, GBS), "series", "column")
    three = solve((0.3, 0.0, 0.7), (TEMPERATE, GLEN, GBS), "series", "column")
    report("[0.3H, 0, 0.7H] empty middle (Glen) vs [0.3H, 0.7H]: u, tau_b",
           {"u0": (three["u0"], two["u0"]), "u2": (three["u2"], two["u1"]),
            "S0": (three["S0"], two["S0"])}, True, tol=DISCRETE_U)
    report("the same: membrane stresses",
           {"M0": (three["M0"], two["M0"]), "M2": (three["M2"], two["M1"])},
           True, tol=DISCRETE_M)
    print(f"  the empty layer's two interface stresses differ cell by cell by "
          f"{rel(three['S2'], three['S1']):.1e} (they agree in projection)")
    old_convention("the same under the old conventions",
                   [((0.3, 0.0, 0.7), (TEMPERATE, GLEN, GBS)), ((0.3, 0.7), (TEMPERATE, GBS))],
                   lambda three_old, two_old: {"u0": (three_old["u0"], two_old["u0"]),
                                               "u2": (three_old["u2"], two_old["u1"])})


def test_empty_bottom_layer_has_no_rheology():
    """[0, H]: the empty basal layer's law does not reach the solution --
    temperate or Glen below the n = 1.8 body gives the same sliding
    velocity, layer velocity, stresses.  Under the old conventions the
    temperate law closes the interface and the floor gives it a membrane."""
    a = solve((0.0, 1.0), (TEMPERATE, GBS), "series", "column")
    b = solve((0.0, 1.0), (GLEN, GBS), "series", "column")
    report("[0, H] empty bottom: temperate vs Glen below",
           {k: (a[k], b[k]) for k in ("u0", "u1", "M1", "S0", "S1")}, True)
    old_convention("the same under the old conventions",
                   [((0.0, 1.0), (TEMPERATE, GBS)), ((0.0, 1.0), (GLEN, GBS))],
                   lambda a_old, b_old: {k: (a_old[k], b_old[k]) for k in ("u0", "u1")})
    # and it is a different model from [H]: the sliding velocity u0 sits
    # below the layer velocity u1 by the upper ice's half-layer of shear
    one = solve((1.0,), (GBS,), "below", "layer")
    shear = rel(a["u1"], a["u0"])
    print(f"  [0, H]: layer velocity above the sliding velocity by {shear:.1e} "
          f"(relative); vs [H]: u1 {rel(a['u1'], one['u0']):.1e}, "
          f"u0 {rel(a['u0'], one['u0']):.1e}")
    assert shear > TRACE


def test_thinning_layer_converges_to_empty():
    """[eps H, (1 - eps) H] -> [0, H] as eps -> 0: the ice layer's velocity
    at first order in eps.  The empty layer's own velocity -- the sliding
    velocity under the column -- does not converge: with no membrane to
    couple it across cells it is determined through cell means alone and
    carries cell-scale roughness (6 % of its magnitude here) that a layer
    of any thickness damps, since creep with n > 1 is infinitely stiff at
    zero strain rate.  The ice above sees only its projection, which is
    why u1 converges regardless.  (At eps = 1e-3 Newton does not converge
    on this slab with these settings; the limit itself solves fine.)"""
    limit = solve((0.0, 1.0), (TEMPERATE, GBS), "series", "column")
    diffs = []
    for eps in (0.1, 0.03, 0.01, 0.003):
        z = solve((eps, 1.0 - eps), (TEMPERATE, GBS), "series", "column")
        diffs.append(rel(z["u1"], limit["u1"]))
        print(f"  eps = {eps:<6g} vs [0, H]: ice velocity u1 {diffs[-1]:.2e}, "
              f"sliding velocity u0 {rel(z['u0'], limit['u0']):.2e}")
    # first order: a factor ~3 in eps buys a factor ~3
    for a, b in zip(diffs, diffs[1:]):
        assert b < 0.5 * a, diffs


def test_nodal_stresses_make_the_empty_layer_exact():
    """The benchmark again with the basal and interlayer stresses at the
    vertices: everything the cellwise stresses left to discretisation
    accuracy is now exact, and the sliding velocity converges too."""
    kw = dict(interface="series", floor_on="column", tau="CG")
    one = solve((1.0,), (GBS,), "below", "layer", tau="CG")
    top = solve((1.0, 0.0), (GBS, TEMPERATE), **kw)
    report("nodal: [H, 0] empty top vs [H]",
           {k: (top[k], one[k]) for k in ("u0", "M0", "S0")}, True)
    two = solve((0.3, 0.7), (TEMPERATE, GBS), **kw)
    three = solve((0.3, 0.0, 0.7), (TEMPERATE, GLEN, GBS), **kw)
    report("nodal: [0.3H, 0, 0.7H] empty middle vs [0.3H, 0.7H]",
           {"u0": (three["u0"], two["u0"]), "u2": (three["u2"], two["u1"]),
            "S0": (three["S0"], two["S0"]), "M0": (three["M0"], two["M0"]),
            "M2": (three["M2"], two["M1"]), "S1": (three["S1"], two["S1"]),
            "S2": (three["S2"], two["S1"])}, True)
    a = solve((0.0, 1.0), (TEMPERATE, GBS), **kw)
    b = solve((0.0, 1.0), (GLEN, GBS), **kw)
    report("nodal: [0, H] empty bottom: temperate vs Glen below",
           {k: (a[k], b[k]) for k in ("u0", "u1", "M1", "S0", "S1")}, True)
    diffs = {"u0": [], "u1": []}
    for eps in (0.1, 0.03, 0.01, 0.003):
        z = solve((eps, 1.0 - eps), (TEMPERATE, GBS), **kw)
        for k in diffs:
            diffs[k].append(rel(z[k], a[k]))
        print(f"  nodal: eps = {eps:<6g} vs [0, H]: ice velocity u1 {diffs['u1'][-1]:.2e}, "
              f"sliding velocity u0 {diffs['u0'][-1]:.2e}")
    for k, d in diffs.items():
        for p, q in zip(d, d[1:]):
            assert q < 0.5 * p, (k, d)
    try:
        z = solve((0.001, 0.999), (TEMPERATE, GBS), **kw)
        print(f"  nodal: eps = 0.001  vs [0, H]: u1 {rel(z['u1'], a['u1']):.2e}, "
              f"u0 {rel(z['u0'], a['u0']):.2e}")
    except firedrake.ConvergenceError as exc:
        print(f"  nodal: eps = 0.001  does not converge ({str(exc).splitlines()[-1].strip()})")


def main():
    print("An empty layer leaves no trace (series interface, column floor):")
    test_uniform_rheology_series_is_below()
    test_empty_top_layer_is_one_layer()
    test_empty_middle_layer_drops_out()
    test_empty_bottom_layer_has_no_rheology()
    test_thinning_layer_converges_to_empty()
    print("\nThe same with nodal basal and interlayer stresses:")
    test_nodal_stresses_make_the_empty_layer_exact()
    print("all checks passed")


if __name__ == "__main__":
    main()
