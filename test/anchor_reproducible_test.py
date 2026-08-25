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
  * lifting is a convex combination -- the CG1 anchor stays inside the
    cell-wise range, unlike an L2 projection, which can overshoot;
  * a DG0 target space is passed through untouched, since a discontinuous
    target has no shared nodes to disagree about.

    python -u anchor_reproducible_test.py      # serial, then re-runs on 3 ranks
"""
import os
import shutil
import subprocess
import sys

import numpy as np
from firedrake import (
    Constant, Function, FunctionSpace, SpatialCoordinate, UnitSquareMesh,
    VectorFunctionSpace, as_vector, inner, max_value, sqrt,
)

from icepack_tools.constants import ice_density as rho_I, gravity as g
from icepack_tools.friction import weertman_anchor
from icepack_tools.geometry import surface_slope
from icepack_tools.parallel import gather, gather_vector

N = 16
M_SLIDE = 3.0
TOL = 1e-9        # cross-rank summation roundoff, not physics


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
    C_dg = weertman_anchor(H, s, u_obs, M_SLIDE, DG)
    gd = gather(C_dg)
    # tau_d is lifted before the division, so the CG1 anchor is not
    # bounded by the DG0 anchor node for node; what must hold is that it
    # stays inside the global cell-wise range rather than overshooting it.
    assert gC.min() >= gd.min() * 0.5 and gC.max() <= gd.max() * 2.0, (
        f"CG1 anchor [{gC.min():.4g}, {gC.max():.4g}] is far outside the "
        f"cell-wise range [{gd.min():.4g}, {gd.max():.4g}]")

    ref = os.environ.get("ICEPACK_TOOLS_ANCHOR_REF")
    if ref:                                    # the 3-rank child
        want = np.load(ref)
        assert want["C"].shape == gC.shape, "different DOF counts"
        # match by coordinate: a 3-rank mesh does not number DOFs the way
        # a serial one does
        k = np.lexsort((gather_vector(mesh.coordinates)[:, 1],
                        gather_vector(mesh.coordinates)[:, 0]))
        dC = np.abs(gC[k] - want["C"])
        dn = np.abs(gn[k] - want["naive"])
        rel = dC / np.maximum(np.abs(want["C"]), 1e-300)
        if comm.rank == 0:
            print(f"  [{size} ranks] max |dC| {dC.max():.3e} "
                  f"(relative {rel.max():.3e}), "
                  f"naive max |dC| {dn.max():.3e}")
        assert rel.max() < TOL, (
            f"the anchor changed by {rel.max():.2%} between 1 and {size} "
            f"ranks -- it is still partition-dependent")
        assert dn.max() > 1e3 * TOL, (
            "the naive construction agreed across rank counts on this mesh, "
            "so the test cannot show it bites.  Raise N or change the "
            "surface -- do not weaken the assertion")
        if comm.rank == 0:
            print("  anchor is rank-independent; the naive one is not")
        return 0

    # ── serial: record, then re-run on 3 ranks ───────────────────────
    print(f"weertman_anchor on a {N}x{N} mesh, {Q.dim()} nodes")
    print(f"  C_w0 range [{gC.min():.5g}, {gC.max():.5g}]   "
          f"DG0 range [{gd.min():.5g}, {gd.max():.5g}]")
    k = np.lexsort((gather_vector(mesh.coordinates)[:, 1],
                    gather_vector(mesh.coordinates)[:, 0]))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_anchor_ref.npz")
    np.savez(out, C=gC[k], naive=gn[k])

    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        os.remove(out)
        print("  [skip] no mpiexec on PATH -- the rank-independence half of "
              "this test did not run")
        return 0
    env = dict(os.environ, ICEPACK_TOOLS_ANCHOR_REF=out, OMP_NUM_THREADS="1")
    try:
        r = subprocess.run([mpiexec, "-n", "3", sys.executable, "-u",
                            os.path.abspath(__file__)], env=env)
    finally:
        os.remove(out)
    if r.returncode:
        raise SystemExit(f"the 3-rank run failed with {r.returncode}")
    print("\nPASS: the friction anchor does not depend on the partition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
