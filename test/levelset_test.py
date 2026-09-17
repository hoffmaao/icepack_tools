r"""Level-set calving front: front kinematics on a synthetic shelf.

A slab of ice (h = 300 m) fills x < x0 on a rectangle; open water beyond.
The velocity is uniform (U, 0), which makes the left edge x = 0 an INFLOW
boundary for the level set: there the eikonal boundary condition, not
transport, must keep ``phi = x - x_front(t)``.  The checks are the ones
that make a front law meaningful in a forward run:

1. initialisation gives the exact signed distance on every cell and flags
   no ice cell for removal;
2. with c = 0 the front advances at U, and the inflow-boundary cells
   follow through the eikonal condition (no reinitialisation);
3. with the ``fixed`` law the front does not move at all;
4. reinitialisation restores ``|grad phi| = 1`` from a distorted field and
   leaves the zero contour where it was;
5. the von Mises rate for uniform extension is
   sqrt(3) B eps^(1/n) |u| / sigma_max;
6. a ``prescribed`` rate equal to U holds the front still (advance and
   retreat cancel), one larger than U retreats it at c - U, and a
   spatially varying rate is honoured where it is given;
7. the ``prescribed`` law refuses to run without a rate;
8. all of the above again on three ranks, with the case rotated so the
   front crosses the short axis: the partitioner's bands then put the
   whole front on one rank and the others own no interface cell, which
   is the case that deadlocked the reinitialisation before the port (a
   rank-conditional halo access inside a collective sequence).

Ported from ``ismip7/tests/test_levelset.py``; 6-8 are new.

    python -u levelset_test.py        (or pytest, serial part only)
"""
import os
import shutil
import subprocess
import sys

import numpy as np

import firedrake as fd
from firedrake import Constant, Function, FunctionSpace, VectorFunctionSpace
from firedrake.petsc import PETSc

from icepack_tools.levelset import LevelSet, LAWS, ANCHORS, _segment_distance

L, W = 100e3, 40e3
NX, NY = 50, 20
X0 = 50e3
H_ICE = 300.0
U = 1000.0  # m/yr
DX = L / NX

#: Flow axis.  0: ice fills x < X0 and flows along x (the original case).
#: 1: the same case rotated, ice in y < X0 flowing along y.  The parallel
#: run uses 1 because the default partitioner cuts the mesh into bands
#: across the long axis, so a front across the SHORT axis lands on one
#: rank and the others own no interface cell -- the case that deadlocked.
AX = int(os.environ.get("LEVELSET_TEST_AXIS", "0"))


def make_case(law="none", **kw):
    if AX == 0:
        mesh = fd.RectangleMesh(NX, NY, L, W)
    else:
        mesh = fd.RectangleMesh(NY, NX, W, L)
    Q0 = FunctionSpace(mesh, "DG", 0)
    xa = fd.SpatialCoordinate(mesh)[AX]
    h = Function(Q0, name="h").interpolate(
        fd.conditional(xa < X0, Constant(H_ICE), Constant(0.0)))
    ls = LevelSet(mesh, h, law=law, h_min=1.0, **kw)
    V = VectorFunctionSpace(mesh, "CG", 1)
    vel = [Constant(0.0), Constant(0.0)]
    vel[AX] = Constant(U)
    u = Function(V).interpolate(fd.as_vector(vel))
    b = Function(Q0).assign(-1000.0)          # deep water: floating
    A = Constant(50.0)                         # MPa^-3 yr^-1 (warm-ish ice)
    return mesh, h, ls, u, b, A


def along(ls):
    """Cell-centre coordinate along the flow axis."""
    return ls.cell_xc[:, AX]


def across(ls):
    return ls.cell_xc[:, 1 - AX]


def front_x(ls, rows=None):
    r"""x of the zero contour: for a vertical front and a distance function,
    ``x_c - phi`` on the cells nearest the front (optionally only on the
    cells selected by ``rows``).  Reduced over ranks, so it is the same
    number however the mesh is split."""
    phi = ls.phi.dat.data_ro
    xc = along(ls)
    near = np.abs(phi) < 2 * DX
    if rows is not None:
        near &= rows
    comm = ls.comm
    tot = comm.allreduce(float((xc[near] - phi[near]).sum()))
    cnt = comm.allreduce(int(near.sum()))
    return tot / cnt


def gathered(ls, arr):
    r"""Concatenate a per-cell array over ranks (order irrelevant here)."""
    return np.concatenate(ls.comm.allgather(np.asarray(arr)))


def test_segment_distance_exact():
    pts = np.array([[0.0, 1.0], [2.0, 0.0], [0.5, -3.0]])
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.0]])
    d = _segment_distance(pts, a, b)
    assert np.allclose(d, [1.0, 1.0, 3.0])


def test_initialisation_is_signed_distance():
    mesh, h, ls, *_ = make_case()
    phi = ls.phi.dat.data_ro
    xc = along(ls)
    assert np.allclose(phi, xc - X0, atol=1e-6 * L)
    ice = h.dat.data_ro > 1.0
    assert not ls.beyond_front()[ice].any()
    assert ls.beyond_front()[~ice].all()
    # drag off in the strip next to the front, on further out
    far = xc > X0 + 1.5 * DX
    assert ls.drag_mask.dat.data_ro[far].all()
    near_water = (~ice) & (xc < X0 + 0.9 * DX)
    assert not ls.drag_mask.dat.data_ro[near_water].any()
    # cell gradient of the exact distance is the unit x vector inside
    g = ls.cell_gradient()
    inner_cells = ls.chi_bnd.dat.data_ro < 0.5
    unit = [0.0, 0.0]
    unit[AX] = 1.0
    assert np.allclose(g[inner_cells], unit, atol=1e-6)
    if ls.comm.size > 1:
        # the partition this test relies on: at least one rank owns no
        # interface cell, or the deadlock it guards against is not exercised
        n_iface = ls.comm.allgather(int((np.abs(phi) < DX).sum()))
        assert min(n_iface) == 0, f"every rank owns front cells: {n_iface}"


def test_free_front_advances_and_eikonal_bc_holds_inflow_edge():
    mesh, h, ls, u, b, A = make_case(law="none", reinit_every=0)
    dt = 0.5
    for _ in range(10):
        ls.advance(dt, u, h, b)
    expected = X0 + U * dt * 10
    assert abs(front_x(ls) - expected) < 0.25 * DX
    phi = ls.phi.dat.data_ro
    xc = along(ls)
    # Distance property across the whole slab, including the inflow edge
    # x = 0, which only the eikonal boundary condition can supply.
    left = xc < 3 * DX
    assert np.allclose(phi[left], (xc - expected)[left], atol=0.5 * DX)
    inside = (xc > 3 * DX) & (xc < expected - 2 * DX)
    assert np.allclose(phi[inside], (xc - expected)[inside], atol=0.5 * DX)


def test_fixed_front_does_not_move():
    mesh, h, ls, u, b, A = make_case(law="fixed")
    for _ in range(5):
        ls.advance(1.0, u, h, b)
    assert abs(front_x(ls) - X0) < 1e-6 * L


def test_reinitialisation_restores_distance():
    mesh, h, ls, *_ = make_case(reinit_sweeps=8)
    phi0 = ls.phi.dat.data_ro.copy()
    ls.phi.dat.data[:] = 3.0 * phi0 + 0.2 * phi0 ** 2 / L
    res = ls.reinitialise()
    phi = ls.phi.dat.data_ro
    xc = along(ls)
    assert res < 0.15          # residual is judged within 10 cells of the front
    assert abs(front_x(ls) - X0) < 0.25 * DX
    band = np.abs(xc - X0) < 8 * DX
    assert np.allclose(phi[band], (xc - X0)[band], atol=0.3 * DX)
    # the same field must come out on every rank count
    assert len(gathered(ls, phi)) == NX * NY * 2


def test_von_mises_rate_uniform_extension():
    mesh, h, ls, u, b, A = make_case(law="vonmises", sigma_max_floating=0.15)
    xa = fd.SpatialCoordinate(mesh)[AX]
    eps = 1e-3   # 1/yr
    vel = [Constant(0.0), Constant(0.0)]
    vel[AX] = eps * xa
    u.interpolate(fd.as_vector(vel))
    n = 3.0
    c_expr = ls.calving_rate_expr(u, h, b, A, n)
    Q1 = FunctionSpace(mesh, "CG", 1)
    c = Function(Q1).interpolate(c_expr)
    B = float(A) ** (-1.0 / n)
    sigma = np.sqrt(3.0) * B * (np.sqrt(eps ** 2 / 2)) ** (1.0 / n)
    xn = mesh.coordinates.dat.data_ro[:, AX]
    inside = (xn > 0.2 * L) & (xn < 0.45 * L)
    expected = eps * xn[inside] * sigma / 0.15
    if inside.any():
        assert np.allclose(c.dat.data_ro[inside], expected, rtol=2e-2)


def test_prescribed_rate_balances_and_retreats():
    mesh, h, ls, u, b, A = make_case(law="prescribed", reinit_every=4)
    dt, nsteps = 0.5, 8
    for c_val, motion in ((U, 0.0), (2 * U, -U)):
        ls.initialise_from_thickness()
        ls.update_cell_fields()
        for _ in range(nsteps):
            ls.advance(dt, u, h, b, rate=Constant(c_val))
        expected = X0 + motion * dt * nsteps
        assert abs(front_x(ls) - expected) < 0.35 * DX, (c_val, front_x(ls))
    ice = h.dat.data_ro > 1.0
    n_flagged = ls.comm.allreduce(int(ls.beyond_front()[ice].sum()))
    assert n_flagged > 0, "retreat flagged no ice cell"

    # A rate given as a field is honoured where it is given: c = 2U on the
    # lower half of the front and U on the upper half, so the lower half
    # retreats at U while the upper half holds.  (The rate is lumped onto
    # CG1 nodes before extension, so it must make sense in the water cells
    # touching the front too; a field that varies *along* the front is the
    # meaningful test.)
    ls.initialise_from_thickness()
    ls.update_cell_fields()
    Q1 = FunctionSpace(mesh, "CG", 1)
    ya = fd.SpatialCoordinate(mesh)[1 - AX]
    rate = Function(Q1).interpolate(
        fd.conditional(ya < W / 2, Constant(2 * U), Constant(U)))
    for _ in range(nsteps):
        ls.advance(dt, u, h, b, rate=rate)
    yc = across(ls)
    lower = yc < W / 2 - 3 * DX
    upper = yc > W / 2 + 3 * DX
    assert abs(front_x(ls, lower) - (X0 - U * dt * nsteps)) < 0.35 * DX
    assert abs(front_x(ls, upper) - X0) < 0.35 * DX


def test_extent_anchor_is_the_distance_to_the_ice_extent():
    r"""``anchor="extent"`` solves the eikonal problem with its boundary
    condition at the ice extent: phi is the exact signed distance to the
    ice/water facets, the front length per cell sums to the domain width,
    and no ice cell is flagged for removal."""
    mesh, h, ls, u, b, A = make_case(anchor="extent")
    phi = ls.phi.dat.data_ro
    assert np.allclose(phi, along(ls) - X0, atol=1e-6 * L)
    ice = h.dat.data_ro > 1.0
    assert (phi[ice] < 0).all() and (phi[~ice] > 0).all()
    assert np.isclose(ls.comm.allreduce(float(ls.front_len.dat.data_ro.sum())),
                      W)
    assert not ls.front_len.dat.data_ro[~ice].any()
    beyond, frac = ls.calving_masks()
    assert not beyond.any() and frac is None


def test_extent_anchor_follows_the_transport():
    r"""Advance is the transport's job: the level set follows the thickness
    extent and never flags a cell while the front only grows."""
    mesh, h, ls, u, b, A = make_case(anchor="extent")
    ls.advance(0.5, u, h, b, A, 3.0)
    assert not ls.calving_masks()[0].any()
    xa = fd.SpatialCoordinate(mesh)[AX]
    h.interpolate(fd.conditional(xa < X0 + DX, Constant(H_ICE), Constant(0.0)))
    ls.advance(0.5, u, h, b, A, 3.0)
    assert np.allclose(ls.phi.dat.data_ro, along(ls) - (X0 + DX), atol=1e-6 * L)
    assert not ls.calving_masks()[0].any()


def test_extent_anchor_retreats_and_sheds_the_right_mass():
    r"""Normal-flow retreat at a constant rate moves the front by ``c dt``
    and flags the cells it passed; a sub-cell retreat sheds exactly
    ``c h L dt`` of mass instead."""
    mesh, h, ls, u, b, A = make_case(law="prescribed", anchor="extent")
    dt, c_big = 0.5, 5000.0                     # c dt = 2500 m > a cell
    ls.advance(dt, u, h, b, rate=Constant(c_big))
    xc, ice = along(ls), h.dat.data_ro > 1.0
    inside = ice & (xc > 0.2 * L)
    assert np.allclose(ls.phi.dat.data_ro[inside],
                       (xc - (X0 - c_big * dt))[inside], atol=0.35 * DX)
    beyond, frac = ls.calving_masks()
    assert beyond[ice & (xc > X0 - c_big * dt)].all()
    assert not beyond[ice & (xc < X0 - c_big * dt - DX)].any()

    mesh, h, ls, u, b, A = make_case(law="prescribed", anchor="extent")
    dt, c_small = 0.1, 100.0                    # c dt = 10 m << a cell
    ls.advance(dt, u, h, b, rate=Constant(c_small))
    beyond, frac = ls.calving_masks()
    assert not beyond.any()
    shed = ls.comm.allreduce(
        float((h.dat.data_ro * frac * ls.cell_area).sum()))
    assert np.isclose(shed, c_small * H_ICE * W * dt, rtol=1e-10)


def test_anchor_name_is_checked():
    try:
        make_case(anchor="somewhere")
    except ValueError as e:
        assert "somewhere" in str(e) and str(ANCHORS) in str(e)
    else:
        raise AssertionError("an unknown anchor should raise")


def test_prescribed_law_needs_a_rate():
    mesh, h, ls, u, b, A = make_case(law="prescribed")
    try:
        ls.advance(0.5, u, h, b)
    except ValueError as e:
        assert "rate" in str(e)
    else:
        raise AssertionError("prescribed law ran without a rate")
    try:
        LevelSet(mesh, h, law="undercut")
    except ValueError as e:
        assert str(LAWS) in str(e)
    else:
        raise AssertionError("unknown law accepted")


def main():
    tests = [
        ("segment distance", test_segment_distance_exact),
        ("signed-distance initialisation", test_initialisation_is_signed_distance),
        ("free front advances, eikonal BC at the inflow edge",
         test_free_front_advances_and_eikonal_bc_holds_inflow_edge),
        ("fixed front holds", test_fixed_front_does_not_move),
        ("reinitialisation restores the distance", test_reinitialisation_restores_distance),
        ("von Mises rate", test_von_mises_rate_uniform_extension),
        ("prescribed rate balances / retreats / varies",
         test_prescribed_rate_balances_and_retreats),
        ("prescribed law refuses to run blind", test_prescribed_law_needs_a_rate),
        ("extent anchor: distance to the ice extent",
         test_extent_anchor_is_the_distance_to_the_ice_extent),
        ("extent anchor: follows the transport",
         test_extent_anchor_follows_the_transport),
        ("extent anchor: retreat and the shed mass",
         test_extent_anchor_retreats_and_sheds_the_right_mass),
        ("anchor name is checked", test_anchor_name_is_checked),
    ]
    size = fd.COMM_WORLD.size
    for i, (label, fn) in enumerate(tests, 1):
        PETSc.Sys.Print(f"\n{i}. {label}" + (f"  [{size} ranks]" if size > 1 else ""))
        fn()
        PETSc.Sys.Print("   ok")
    if size > 1 or os.environ.get("ICEPACK_TOOLS_MPI_CHILD"):
        PETSc.Sys.Print(f"\nPASS on {size} ranks")
        return 0

    PETSc.Sys.Print("\n12. the same on three ranks, front across the short axis")
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        print("  [skip] no mpiexec on PATH -- the parallel half of this test "
              "did not run")
        return 0
    env = dict(os.environ, ICEPACK_TOOLS_MPI_CHILD="1", OMP_NUM_THREADS="1",
               LEVELSET_TEST_AXIS="1")
    r = subprocess.run([mpiexec, "-n", "3", sys.executable, "-u",
                        os.path.abspath(__file__)], env=env, timeout=600)
    if r.returncode:
        raise SystemExit(f"the 3-rank run failed with {r.returncode}")
    print("\nPASS: level-set front kinematics, serial and on 3 ranks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
