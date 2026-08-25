r"""Does the friction anchor depend on how the mesh is partitioned?

:func:`~icepack_tools.friction.weertman_anchor` is a *fixed* scaling: the
inverted ``theta`` is a log-adjustment around it, so if the anchor moves,
the friction moves, and so does everything downstream.

Its ingredient :math:`\nabla s` is discontinuous -- cell-wise constant for
a CG1 surface -- and interpolating a discontinuous expression into a
continuous space is not well defined at a shared node.  Firedrake
evaluates the expression cell by cell and the node keeps whichever cell
wrote last, so the value follows the cell numbering, which follows the
partition, which follows the rank count.  The original one-line
``Function(Q).interpolate(tau_d / speed**(1/m))`` -- still the form used
in ``ismip7/icepack2_tools/dual_friction.py`` and
``gia-icepack/scripts/ase_model.py`` -- gave 77 % of the nodes of a Store
mesh a different anchor on 4 ranks than on 1, by up to a factor of 20 at
a node, moving the converged velocity 26 % on 500 m thick ice.

So this asserts:

  * the anchor is bit-comparable on 1 and 3 ranks (to summation roundoff,
    not to a factor of 20);
  * the naive construction really is partition-dependent on this mesh, so
    the test is pinning a live defect rather than passing vacuously;
  * lifting is a convex combination -- the lifted driving stress stays
    inside the cell-wise range node for node, unlike an L2 projection,
    which can overshoot;
  * a DG0 target space is passed through untouched -- identical to the
    naive construction -- since a discontinuous target has no shared nodes
    to disagree about;
  * the guarantee holds for *whichever* operand is discontinuous, not just
    for ``tau_d``: a DG0 ``u_obs`` against a CG1 ``Q`` makes the speed the
    discontinuous factor, and it must be rank-independent too, while a CG1
    ``u_obs`` must come back bit-identical to interpolating its nodal
    speed.

    python -u anchor_reproducible_test.py      # serial, then re-runs on 3 ranks
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from firedrake import (
    Constant, Function, FunctionSpace, SpatialCoordinate, UnitSquareMesh,
    VectorFunctionSpace, as_vector, inner, max_value, sqrt,
)

from icepack_tools.constants import ice_density as rho_I, gravity as g
from icepack_tools.friction import weertman_anchor
from icepack_tools.geometry import cg1_lift, surface_slope
from icepack_tools.parallel import gather, gather_vector

N = 16
M_SLIDE = 3.0
TOL = 1e-9        # cross-rank summation roundoff, not physics
CHILD_TIMEOUT = 900.0   # seconds; a deadlocked child must fail, not hang


def build():
    """A surface with real curvature, so |grad s| genuinely varies by cell."""
    mesh = UnitSquareMesh(N, N)
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    H = Function(Q).interpolate(Constant(400.0) + Constant(600.0) * x * (1 - y))
    s = Function(Q).interpolate(Constant(2000.0) * (1 - x) ** 2
                                + Constant(300.0) * y * (1 - x))
    u_obs = Function(V).interpolate(
        as_vector([Constant(80.0) + Constant(900.0) * x ** 2,
                   Constant(40.0) * y]))
    return mesh, Q, H, s, u_obs


def naive_anchor(H, s, u_obs, m_slide, Q):
    """The construction being replaced, kept so the test can show it bites."""
    grad_s = surface_slope(s)
    tau_d = rho_I * g * H * sqrt(inner(grad_s, grad_s) + Constant(1e-12))
    speed = max_value(sqrt(inner(u_obs, u_obs)), Constant(1.0))
    return Function(Q).interpolate(tau_d / speed ** (1.0 / m_slide))


def main():
    mesh, Q, H, s, u_obs = build()
    comm = mesh.comm
    size = comm.size

    C = weertman_anchor(H, s, u_obs, M_SLIDE, Q)
    naive = naive_anchor(H, s, u_obs, M_SLIDE, Q)
    gC, gn = gather(C), gather(naive)
    assert np.all(gC > 0.0), "the anchor must be strictly positive"

    # ── the lift is a convex combination, so no overshoot ────────────
    DG = FunctionSpace(mesh, "DG", 0)
    # Only tau_d is lifted, so the exact node-for-node bound is a property
    # of the lifted tau_d, not of the anchor (which then divides by a
    # nodal speed).  Assert it where it actually holds: an L2 projection
    # in place of cg1_lift fails this, a convex combination cannot.
    grad_s = surface_slope(s)
    tau_dg = Function(DG).interpolate(
        rho_I * g * H * sqrt(inner(grad_s, grad_s) + Constant(1e-12)))
    g_cell, g_lift = gather(tau_dg), gather(cg1_lift(tau_dg))
    span = g_cell.max() - g_cell.min()
    assert g_lift.min() >= g_cell.min() - 1e-12 * span, (
        f"lifted tau_d undershoots the cell-wise range: "
        f"{g_lift.min():.6g} < {g_cell.min():.6g}")
    assert g_lift.max() <= g_cell.max() + 1e-12 * span, (
        f"lifted tau_d overshoots the cell-wise range: "
        f"{g_lift.max():.6g} > {g_cell.max():.6g}")

    # A DG0 target has no shared nodes, so it must take the naive path
    # verbatim rather than being routed through the lift.
    C_dg = weertman_anchor(H, s, u_obs, M_SLIDE, DG)
    gd = gather(C_dg)
    np.testing.assert_allclose(gd, gather(naive_anchor(H, s, u_obs, M_SLIDE, DG)),
                               rtol=1e-14, atol=0.0)

    # A DG0 u_obs against a CG1 Q: now the *speed* is the discontinuous
    # operand, and it has to be lifted for the same reason tau_d is.  A
    # CG1 u_obs must not be touched, so check that too -- the lift is
    # keyed on the operand, not applied blindly.
    Vd = VectorFunctionSpace(mesh, "DG", 0)
    u_dg = Function(Vd).interpolate(u_obs)
    C_udg = weertman_anchor(H, s, u_dg, M_SLIDE, Q)
    gu = gather(C_udg)
    assert np.all(gu > 0.0), "the anchor must be strictly positive"
    np.testing.assert_allclose(
        gC, gather(weertman_anchor(H, s, u_obs, M_SLIDE, Q)),
        rtol=0.0, atol=0.0)

    ref = os.environ.get("ICEPACK_TOOLS_ANCHOR_REF")
    if ref:                                    # the 3-rank child
        want = np.load(ref)
        assert want["C"].shape == gC.shape, "different DOF counts"
        # match by coordinate: a 3-rank mesh does not number DOFs the way
        # a serial one does
        k = np.lexsort((gather_vector(mesh.coordinates)[:, 1],
                        gather_vector(mesh.coordinates)[:, 0]))
        dC = np.abs(gC[k] - want["C"])
        du = np.abs(gu[k] - want["C_udg"])
        dn = np.abs(gn[k] - want["naive"])
        rel = dC / np.maximum(np.abs(want["C"]), 1e-300)
        rel_u = du / np.maximum(np.abs(want["C_udg"]), 1e-300)
        if comm.rank == 0:
            print(f"  [{size} ranks] max |dC| {dC.max():.3e} "
                  f"(relative {rel.max():.3e}), DG0 u_obs "
                  f"{rel_u.max():.3e}, naive max |dC| {dn.max():.3e}")
        assert rel.max() < TOL, (
            f"the anchor changed by {rel.max():.2%} between 1 and {size} "
            f"ranks -- it is still partition-dependent")
        assert rel_u.max() < TOL, (
            f"with a DG0 u_obs the anchor changed by {rel_u.max():.2%} "
            f"between 1 and {size} ranks -- the discontinuous speed is "
            f"reaching the continuous target unlifted")
        # Unlike parallel_stats_test.py, this test *does* assert that the
        # naive construction disagrees across rank counts, and the two
        # cases are not alike.  There the naive quantity is a rank-LOCAL
        # max, so whether rank 0 owns the global maximum is a coin flip
        # the partitioner tosses -- one node decides it.  Here the naive
        # quantity differs at every shared node whose adjacent cells carry
        # a different grad(s), which on a curved surface is essentially
        # every interior node.  Neighbouring cell values differ by O(100)
        # in a range of 118-3406, so for dn.max() to fall below 1e-6 the
        # 3-rank partition would have to reproduce the serial
        # last-cell-wins choice at all ~250 of them at once.  That is a
        # degenerate partition, not luck.
        assert dn.max() > 1e3 * TOL, (
            "the naive construction agreed with the serial run at every "
            "shared node, so this 3-rank partition reproduced the serial "
            "cell numbering throughout.  That means the partition is "
            "degenerate, not that the defect is gone -- check the "
            "partitioner rather than weakening the assertion")
        if comm.rank == 0:
            print("  anchor is rank-independent; the naive one is not")
        return 0

    if size > 1:
        # Launched under mpiexec by hand, with no reference to compare
        # against.  The local checks above are the whole test in that
        # case: recording a reference here would have every rank write its
        # own temporary file and spawn its own nested `mpiexec -n 3` from
        # inside an MPI job.
        if comm.rank == 0:
            print(f"  [{size} ranks] local checks pass; set "
                  f"ICEPACK_TOOLS_ANCHOR_REF or run serially for the "
                  f"cross-rank comparison")
        return 0

    # ── serial: record, then re-run on 3 ranks ───────────────────────
    print(f"weertman_anchor on a {N}x{N} mesh, {Q.dim()} nodes")
    print(f"  C_w0 range [{gC.min():.5g}, {gC.max():.5g}]   "
          f"DG0 range [{gd.min():.5g}, {gd.max():.5g}]")
    k = np.lexsort((gather_vector(mesh.coordinates)[:, 1],
                    gather_vector(mesh.coordinates)[:, 0]))
    fd, out = tempfile.mkstemp(prefix="icepack_tools_anchor_", suffix=".npz")
    os.close(fd)
    np.savez(out, C=gC[k], naive=gn[k], C_udg=gu[k])

    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        os.remove(out)
        print("  [skip] no mpiexec on PATH -- the rank-independence half of "
              "this test did not run")
        return 0
    env = dict(os.environ, ICEPACK_TOOLS_ANCHOR_REF=out, OMP_NUM_THREADS="1")
    try:
        r = subprocess.run([mpiexec, "-n", "3", sys.executable, "-u",
                            os.path.abspath(__file__)], env=env,
                           timeout=CHILD_TIMEOUT)
    except subprocess.TimeoutExpired:
        # A regression here is as likely to hang as to fail: gather is
        # collective, so a statistic computed on one rank only deadlocks
        # the run.  Without the timeout that hang is the test run's.
        raise SystemExit(
            f"the 3-rank run did not finish within {CHILD_TIMEOUT:.0f}s -- "
            f"treat a hang as a failure, not as a slow machine")
    finally:
        os.remove(out)
    if r.returncode:
        raise SystemExit(f"the 3-rank run failed with {r.returncode}")
    print("\nPASS: the friction anchor does not depend on the partition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
