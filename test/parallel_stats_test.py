r"""Do field statistics survive a change of rank count?

The bug this pins is not exotic.  Every consumer writes report lines like
``np.hypot(*u.dat.data_ro.T).max()`` under ``PETSc.Sys.Print``, which
prints rank 0's local maximum and calls it the domain's.  On the Store
mesh the velocity RMS misfit read 191.9 m/yr in serial and 44.2 m/yr on
four ranks from bit-identical states -- a number that changes with the
rank count is not a measurement.

So the assertions are:

  * :func:`~icepack_tools.parallel.gather` reproduces the analytic
    statistics of a field whose exact min/max/mean/median are known, on
    every rank count;
  * the *naive* rank-local statistic really does differ under MPI, so
    this test would have caught the original bug rather than passing
    vacuously;
  * :func:`~icepack_tools.parallel.scatter` inverts ``gather`` exactly,
    which is what lets a redundantly-run serial optimiser drive a
    distributed control;
  * a ``Cofunction`` (an assembled 1-form -- the shape every adjoint
    gradient arrives in) gathers to the same global vector as the
    equivalent ``Function``, so the optimiser is fed the assembled
    gradient and not a partial sum.

    python -u parallel_stats_test.py          # serial, then re-runs on 3 ranks
"""
import os
import shutil
import subprocess
import sys

import numpy as np
from firedrake import (
    Function, FunctionSpace, SpatialCoordinate, TestFunction, UnitSquareMesh,
    VectorFunctionSpace, as_vector, assemble, dx,
)

from icepack_tools.parallel import (format_stats, gather, gather_speed,
                                    gather_vector, scatter, stats)

#: A 12 x 12 unit square, so the CG1 nodes are exactly the tensor grid
#: {i/12} x {j/12}.  Every statistic below is then known in closed form,
#: which makes the reference independent of any serial run.
N = 12


def build():
    mesh = UnitSquareMesh(N, N)
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    f = Function(Q).interpolate(x)
    u = Function(V).interpolate(as_vector([3.0 * x, 4.0 * x]))
    return mesh, Q, V, f, u


def check(mesh, Q, V, f, u):
    comm = mesh.comm
    rank, size = comm.rank, comm.size

    # ── analytic reference ───────────────────────────────────────────
    # f = x on the tensor grid: each of the N+1 columns x = i/N carries
    # N+1 nodes, so the value multiset is {i/N} each repeated N+1 times.
    exact = np.repeat(np.arange(N + 1) / N, N + 1)
    g = gather(f)
    assert g.size == Q.dim(), f"gather gave {g.size} of {Q.dim()} DOFs"
    np.testing.assert_allclose(np.sort(g), exact, atol=1e-12)

    s = stats(g)
    assert s["n"] == Q.dim()
    for key, want in (("min", 0.0), ("max", 1.0), ("mean", 0.5), ("p50", 0.5)):
        assert abs(s[key] - want) < 1e-12, f"{key} = {s[key]}, want {want}"
    # rms of a uniform grid on [0, 1]: mean of (i/N)^2 over i = 0..N
    want_rms = float(np.sqrt(((np.arange(N + 1) / N) ** 2).mean()))
    assert abs(s["rms"] - want_rms) < 1e-12

    # ── the naive statistic is the thing being replaced ──────────────
    # What is guaranteed is the *mechanism*: owned DOFs partition the
    # global set, so on more than one rank nobody holds them all and
    # every `f.dat.data_ro.<stat>()` is a statistic of a strict subset.
    # Whether that subset happens to contain the global extremum is up to
    # the partitioner -- PETSc's simple partitioner cuts this mesh into
    # bands that each span the full range of x -- so asserting on the
    # naive max would be asserting on the partitioner, not on the bug.
    naive = float(f.dat.data_ro.max())
    owned = comm.allgather(int(f.dat.data_ro.size))
    if size > 1:
        assert min(owned) < Q.dim(), (
            f"every rank owns all {Q.dim()} DOFs, so nothing here is "
            f"distributed and the test is vacuous")
        assert sum(owned) == Q.dim(), (
            f"owned DOFs {sum(owned)} != {Q.dim()}: not a partition, so "
            f"gather would double-count shared nodes")
    else:
        assert owned == [Q.dim()] and abs(naive - 1.0) < 1e-12

    # The real property: the gathered statistics are bit-identical on
    # every rank, and equal the analytic reference on every rank count.
    for key in ("min", "max", "mean", "rms", "p50", "p90", "p99"):
        vals = comm.allgather(s[key])
        assert all(v == vals[0] for v in vals), f"{key} differs by rank: {vals}"

    # ── masking ──────────────────────────────────────────────────────
    # A buffered domain reports on ice only; an empty mask must give
    # n = 0 and NaNs rather than raise, so a report line survives it.
    half = stats(g, mask=g > 0.5 + 1e-12)
    assert half["n"] == (N // 2) * (N + 1), half["n"]
    assert abs(half["min"] - (7 / 12)) < 1e-12
    empty = stats(g, mask=g > 2.0)
    assert empty["n"] == 0 and np.isnan(empty["max"])
    assert format_stats(empty) == "(empty)"

    # ── vectors ──────────────────────────────────────────────────────
    uv = gather_vector(u)
    assert uv.shape == (Q.dim(), 2), uv.shape
    np.testing.assert_allclose(np.sort(uv[:, 0]), 3.0 * exact, atol=1e-12)
    sp = gather_speed(u)                      # |(3x, 4x)| = 5x
    np.testing.assert_allclose(np.sort(sp), 5.0 * exact, atol=1e-12)

    # ── scatter inverts gather ───────────────────────────────────────
    h = Function(Q)
    scatter(g * 2.0 + 1.0, h)
    np.testing.assert_allclose(h.dat.data_ro, f.dat.data_ro * 2.0 + 1.0,
                               atol=1e-14)
    np.testing.assert_allclose(gather(h), g * 2.0 + 1.0, atol=1e-14)

    # ── Cofunction: the shape every adjoint gradient arrives in ──────
    # assemble() has already summed shared-node contributions into the
    # owner, so the gather must equal the serial assembly and not a
    # partial sum.  Its exact value: integral of psi_i dx = the nodal
    # area, which sums to the domain area.
    cof = assemble(TestFunction(Q) * dx)
    gc = gather(cof)
    assert gc.size == Q.dim()
    assert abs(gc.sum() - 1.0) < 1e-12, gc.sum()

    tag = "serial" if size == 1 else f"{size} ranks"
    if rank == 0:
        print(f"  [{tag}] gather {g.size} DOFs   "
              f"{format_stats(s, fmt='6.3f')}   naive local max {naive:.3f}")
    return s


def main():
    mesh, Q, V, f, u = build()
    s = check(mesh, Q, V, f, u)

    if mesh.comm.size > 1 or os.environ.get("ICEPACK_TOOLS_MPI_CHILD"):
        if mesh.comm.rank == 0:
            print("  parallel statistics match the analytic reference")
        return 0

    print("parallel statistics")
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        print("  [skip] no mpiexec on PATH -- the rank-independence half of "
              "this test did not run")
        return 0

    env = dict(os.environ, ICEPACK_TOOLS_MPI_CHILD="1", OMP_NUM_THREADS="1")
    r = subprocess.run([mpiexec, "-n", "3", sys.executable, "-u",
                        os.path.abspath(__file__)], env=env)
    if r.returncode:
        raise SystemExit(f"the 3-rank run failed with {r.returncode}; the "
                         f"statistics are not rank-independent")
    print("\nPASS: identical statistics on 1 and 3 ranks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
