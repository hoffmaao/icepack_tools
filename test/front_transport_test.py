r"""Thickness transport at an advected calving front.

A uniform floating shelf, ``H = 300 m``, flows at a uniform ``U = 500 m/yr``
toward a straight front across a planar strip.  The flow is divergence
free, so the exact solution is known: the thickness stays uniform (plus
the accumulation) right up to the front, the front moves at ``U - c``, and
the ice leaves only through the calving flux ``H c`` per unit front length.
A small consumer below runs the step a forward runs: the front, a DG0
upwind transport through :meth:`FrontTransport.divergence`, the front
rules, the excess hand-back and the iceberg removal.

1. Free advance, ``c = 0``: no ice is booked as calving, and the ice
   behind the front keeps nearly the upstream thickness.  Zeroing every
   cell beyond the level set after a plain transport failed both: it
   drained the front cells into the water, booking 28 % of the ice that
   reached the front as calving, and the edge thinned to 0.28 of the exact
   thickness.
2. Held front, ``c = U``: everything that reaches the front calves.
3. Retreat, ``c = U + 300``: the front retreats at 300 m/yr and the calving
   flux is ``H (U + 300)``; its per-cell form, each exporting cell's
   share, sums to the total.
4. A barrier holds a free front at its t = 0 extent: what reaches it
   calves, as under ``c = U``, fast ice or slow, and nothing piles up past
   it; a retreat past the barrier is not held.
5. A pocket of open water the level set encloses in the ice (one cell
   60 km inside the front) is filled back into the ice, and no ice piles
   up in it or calves into it: under a free front, under a held one (the
   ``fixed`` law) and behind a barrier, whether the pocket is there from
   the start or opens later.
6. A grounded tongue, four cells wide, between floating ice that retreats
   on either side: the tongue survives to within a cell of where its ice
   carries it, and no cell beyond the front holds more ice than the
   thickest inside.  The excess hand-back is what holds the second.
7. A floating raft cut off from the grounded ice is removed, all of it
   and only it; a connected sheet loses nothing.
8. The same scheme on a DG1 thickness (vertex nodes), the rules reading
   the cell means and acting on whole cells, with the positivity floor a
   DG1 transport needs: a free advance calves nothing and keeps its ice, a
   held front calves what reaches it.
9. The coupling refuses an extent-anchored level set.
10. In every case the volume budget closes.
11. All of the above again on three ranks, where the floods cross
    partitions.

The cases are CalvingMIP's (its ``test/front_transport_test.py`` and
``test/iceberg_test.py``), where the scheme was built.

    python -u test/front_transport_test.py     (or pytest, serial part only)
"""
import gc
import os
import shutil
import subprocess
import sys

import numpy as np
from mpi4py import MPI

import firedrake as fd
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace, TestFunction,
    TrialFunction, assemble, dx,
)
from firedrake.petsc import PETSc

from icepack_tools.levelset import LevelSet
from icepack_tools.front_transport import FrontTransport

LENGTH, WIDTH, DX = 250e3, 40e3, 5e3
FRONT0 = 150e3
H0, U, SMB = 300.0, 500.0, 0.3
DT, STEPS = 1.0, 40
LU = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


class Strip:
    r"""A floating shelf on a planar strip, its front a straight level set at
    x = FRONT0, flowing at ``u`` [m/yr] along x.  ``land`` (a function of
    the centroids returning a boolean array) marks grounded cells, where a
    gated rate is zero.  The iceberg pass runs only with ``kill_icebergs``:
    a strip of floating ice with no grounded cell is all iceberg, and the
    shelf here stands for one fed across its upstream edge."""

    def __init__(self, u=U, barrier=False, land=None, h_init=None, kill_icebergs=False,
                 law="prescribed", pocket=None):
        mesh = fd.RectangleMesh(int(LENGTH / DX), int(WIDTH / DX), LENGTH, WIDTH)
        self.mesh, self.comm = mesh, mesh.comm
        Q0 = self.Q0 = FunctionSpace(mesh, "DG", 0)
        x = fd.SpatialCoordinate(mesh)
        self.u = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(
            fd.as_vector((Constant(u), Constant(0.0))))
        self.h = Function(Q0, name="thickness")
        if h_init is None:
            self.h.interpolate(fd.conditional(x[0] < FRONT0, Constant(H0), Constant(0.0)))
        else:
            self.h.interpolate(h_init(x))
        self.ls = LevelSet(mesh, self.h, law=law, h_min=1.0, anchor="advect")
        if h_init is None:
            self.ls.phi.interpolate(x[0] - Constant(FRONT0))
            self.ls.update_cell_fields()
        self.xc = self.ls.cell_xc
        self.pocket = None
        if pocket is not None:
            # one cell of open water at ``pocket`` (x, y), inside the ice
            d = np.hypot(self.xc[:, 0] - pocket[0], self.xc[:, 1] - pocket[1])
            best = self.comm.allreduce(d.min() if d.size else np.inf, op=MPI.MIN)
            self.pocket = d == best
            self.ls.phi.dat.data[self.pocket] = +6.0e3
            self.ls.phi0.dat.data[self.pocket] = +6.0e3
            self.ls.update_cell_fields()
        self.area = self.ls.cell_area
        self.grounded = (land(self.xc) if land is not None
                         else np.zeros(len(self.area), bool))
        self.kill_icebergs = kill_icebergs
        self.ft = FrontTransport(self.ls, self.u, barrier=barrier,
                                 phi0=self.ls.phi.copy(deepcopy=True))
        self.h_old = Function(Q0)
        self.src = Function(Q0)
        self.dt_c = Constant(1.0)
        ht, w = TrialFunction(Q0), TestFunction(Q0)
        F = ((ht - self.h_old) / self.dt_c * w * dx
             + self.ft.divergence(ht, w) - self.src * w * dx)
        self.solver = fd.LinearVariationalSolver(
            fd.LinearVariationalProblem(fd.lhs(F), fd.rhs(F), self.h),
            solver_parameters=LU)

    def total(self, a):
        return self.comm.allreduce(float((np.asarray(a) * self.area).sum()))

    def volume(self):
        return self.total(self.h.dat.data_ro)

    def step(self, dt, rate):
        r"""One step of ``dt`` with the ablation ``rate`` (UFL, m/yr); the
        volumes it moved [m3]."""
        self.h_old.assign(self.h)
        self.ft.advance(dt, rate)
        inside = self.ft.chi_in.dat.data_ro > 0.5
        self.src.dat.data[:] = np.where(inside, SMB, 0.0)
        smb = self.total(self.src.dat.data_ro) * dt
        self.dt_c.assign(dt)
        self.solver.solve()
        out = float(assemble(self.ft.outflow(self.h))) * dt
        calving = float(assemble(self.ft.calving(self.h))) * dt
        d = self.h.dat.data
        clamp = -self.total(np.minimum(d, 0.0))
        np.maximum(d, 0.0, out=d)
        keep, band, touch = self.ft.rules()
        cut = d * (1.0 - keep)
        calving += self.total(cut)
        d -= cut
        give, back, moved = self.ft.excess(self.h.dat.data_ro, band, touch)
        d = self.h.dat.data
        d[:] = d * (1.0 - give) + back
        berg_vol = 0.0
        if self.kill_icebergs:
            berg = self.ft.icebergs(self.h.dat.data_ro, self.grounded)
            d = self.h.dat.data
            berg_vol = self.total(np.where(berg, d, 0.0))
            d[berg] = 0.0
        return dict(smb=smb, calving=calving + berg_vol, berg=berg_vol,
                    outflux=out, clamp=clamp, moved=moved)

    def run(self, steps, rate):
        v0 = self.volume()
        tot = dict(smb=0.0, calving=0.0, berg=0.0, outflux=0.0, clamp=0.0, moved=0.0)
        for _ in range(steps):
            for k, v in self.step(DT, rate).items():
                tot[k] += v
        resid = (self.volume() - v0) - (tot["smb"] - tot["calving"] - tot["outflux"]
                                        + tot["clamp"])
        return tot, resid

    def front_x(self, rows=None):
        r"""Mean x of the level set's zero, from each inside cell within a
        cell of it (on all ranks)."""
        phi = self.ls.phi.dat.data_ro
        near = (phi <= 0.0) & (phi > -DX)
        if rows is not None:
            near &= rows
        s = self.comm.allreduce(float((self.xc[near, 0] - phi[near]).sum()))
        n = self.comm.allreduce(int(near.sum()))
        return s / n

    def thickness_behind(self, k, steps=STEPS):
        r"""Mean thickness of the cells between ``k - 1`` and ``k`` cell
        widths behind the front, as a fraction of the exact uniform value."""
        phi = self.ls.phi.dat.data_ro
        band = (phi <= -(k - 1) * DX) & (phi > -k * DX)
        s = self.comm.allreduce(float(self.h.dat.data_ro[band].sum()))
        n = self.comm.allreduce(int(band.sum()))
        return s / n / (H0 + SMB * DT * steps)


def test_free_advance_keeps_its_ice():
    m = Strip()
    tot, resid = m.run(STEPS, Constant(0.0))
    passing = H0 * U * WIDTH * DT * STEPS
    xf = m.front_x()
    behind = [m.thickness_behind(k) for k in (1, 2, 4)]
    PETSc.Sys.Print(
        f"    front {xf/1e3:.1f} km (exact {(FRONT0 + U*DT*STEPS)/1e3:.1f}); calving "
        f"{tot['calving']/passing:.3%} of the ice that reached the front; thickness "
        "0-5 / 5-10 / 15-20 km behind the front "
        + " ".join(f"{b:.3f}" for b in behind) + f" of exact; budget residual {resid:.2e} m3")
    assert abs(resid) < 1e-6 * passing
    assert tot["calving"] < 1e-3 * passing, tot["calving"] / passing
    assert abs(xf - (FRONT0 + U * DT * STEPS)) <= 0.2 * DX
    assert behind[0] > 0.8 and behind[1] > 0.9 and behind[2] > 0.98, behind


def test_a_held_front_calves_what_reaches_it():
    m = Strip()
    tot, resid = m.run(STEPS, Constant(U))
    flux = H0 * U * WIDTH * DT * STEPS
    xf = m.front_x()
    PETSc.Sys.Print(f"    front {xf/1e3:.1f} km (exact {FRONT0/1e3:.1f}); calving "
                    f"{tot['calving']/flux:.3f} of H U W t; thickness 2 cells behind "
                    f"{m.thickness_behind(2):.3f}")
    assert abs(resid) < 1e-6 * flux
    assert abs(xf - FRONT0) <= 0.2 * DX
    assert abs(tot["calving"] / flux - 1.0) < 0.05
    assert m.thickness_behind(2) > 0.97


def test_a_retreat_calves_at_the_calving_rate():
    w = 300.0
    m = Strip()
    tot, resid = m.run(STEPS, Constant(U + w))
    xf = m.front_x()
    # the ice passing the front, plus the ice the retreat uncovers
    expect = H0 * U * WIDTH * DT * STEPS + H0 * w * WIDTH * DT * STEPS
    PETSc.Sys.Print(f"    front {xf/1e3:.1f} km (exact {(FRONT0 - w*DT*STEPS)/1e3:.1f}); "
                    f"calving {tot['calving']/expect:.3f} of expected")
    assert abs(resid) < 1e-6 * expect
    assert abs(xf - (FRONT0 - w * DT * STEPS)) <= 0.2 * DX
    assert abs(tot["calving"] / expect - 1.0) < 0.05


def test_the_per_cell_calving_sums_to_the_total():
    m = Strip()
    m.run(5, Constant(U + 300.0))
    total = float(assemble(m.ft.calving(m.h)))
    per_cell = assemble(m.ft.calving(m.h, TestFunction(m.Q0)))
    summed = m.comm.allreduce(float(per_cell.dat.data_ro.sum()))
    PETSc.Sys.Print(f"    calving rate {total:.6e} m3/yr, per cell summed {summed:.6e}")
    assert total > 0.5 * H0 * (U + 300.0) * WIDTH, total
    assert abs(summed - total) <= 1e-10 * total, (summed, total)


def test_a_barrier_holds_an_advance_and_not_a_retreat():
    flux = H0 * U * WIDTH * DT * STEPS
    # fast ice, and slow ice, where the reinitialisation's small lifts of the
    # front used to unhold it and send a quarter of the ice past the barrier
    for u in (U, 40.0):
        m = Strip(u=u, barrier=True)
        tot, resid = m.run(STEPS, Constant(0.0))
        reach = H0 * u * WIDTH * DT * STEPS
        xf = m.front_x()
        phi = m.ls.phi.dat.data_ro
        past = m.total(np.where(phi > 0.0, m.h.dat.data_ro, 0.0))
        PETSc.Sys.Print(f"    free front at the barrier, u {u:.0f} m/yr: {xf/1e3:.1f} km "
                        f"(barrier {FRONT0/1e3:.1f}); calving {tot['calving']/reach:.3f} "
                        f"of H u W t; ice beyond the front {past/reach:.1e} of it")
        assert abs(resid) < 1e-6 * flux
        assert abs(xf - FRONT0) <= 0.2 * DX
        assert abs(tot["calving"] / reach - 1.0) < 0.05
        assert past < 1e-6 * reach
    w = 300.0
    m = Strip(barrier=True)
    tot, resid = m.run(STEPS, Constant(U + w))
    xf = m.front_x()
    PETSc.Sys.Print(f"    retreat behind the barrier: {xf/1e3:.1f} km (exact "
                    f"{(FRONT0 - w*DT*STEPS)/1e3:.1f})")
    assert abs(resid) < 1e-6 * flux
    assert abs(xf - (FRONT0 - w * DT * STEPS)) <= 0.2 * DX


def test_an_enclosed_pocket_of_water_is_filled():
    u_slow = 40.0
    passing = H0 * u_slow * WIDTH * DT * STEPS
    exact = H0 + SMB * DT * STEPS
    where = (FRONT0 - 60e3, WIDTH / 2)
    # a free front, a held one (the fixed law) and a barrier, the pocket
    # there from the start; the free front again with the pocket opening
    # after the coupling was built, so the advance has to find it
    cases = [("free", dict(), Constant(0.0), 0.0),
             ("fixed", dict(law="fixed"), None, 1.0),
             ("barrier", dict(barrier=True), Constant(0.0), 1.0),
             ("opened later", None, Constant(0.0), 0.0)]
    for label, kw, rate, calves in cases:
        if kw is None:
            m = Strip(u=u_slow)
            d = np.hypot(m.xc[:, 0] - where[0], m.xc[:, 1] - where[1])
            best = m.comm.allreduce(d.min() if d.size else np.inf, op=MPI.MIN)
            pocket = d == best
            m.ls.phi.dat.data[pocket] = +6.0e3
            m.ls.update_cell_fields()
        else:
            m = Strip(u=u_slow, pocket=where, **kw)
            pocket = m.pocket
        tot, resid = m.run(STEPS, rate)
        phi_pocket = m.comm.allreduce(float(m.ls.phi.dat.data_ro[pocket].sum()))
        h_max = m.comm.allreduce(float(m.h.dat.data_ro.max()), op=MPI.MAX)
        PETSc.Sys.Print(f"    {label}: pocket phi {phi_pocket/1e3:+.2f} km; thickest cell "
                        f"{h_max/exact:.3f} of the upstream ice; calving "
                        f"{tot['calving']/passing:.3f} of the ice reaching the front")
        assert abs(resid) < 1e-6 * passing
        assert phi_pocket < 0.0
        assert h_max < 1.02 * exact, h_max / exact
        # nothing calves into the pocket: a free front calves nothing, a held
        # one what reaches the front and no more
        assert abs(tot["calving"] / passing - calves) < (1e-3 if calves == 0.0 else 0.05)


def test_a_grounded_tongue_holds_and_piles_nothing_ahead():
    u_slow, steps = 40.0, 2 * STEPS
    land_from = FRONT0 - 50e3

    def land(xc):
        return (xc[:, 0] >= land_from) & (np.abs(xc[:, 1] - WIDTH / 2) < WIDTH / 4)

    m = Strip(u=u_slow, land=land, kill_icebergs=True)
    # a gated retreat: the floating rate on floating cells, none on land
    gate = Function(m.Q0)
    gate.dat.data[:] = np.where(m.grounded, 0.0, u_slow + 300.0)
    tot, resid = m.run(steps, gate)
    h, phi, y = m.h.dat.data_ro, m.ls.phi.dat.data_ro, m.xc[:, 1]
    mid = np.abs(y - WIDTH / 2) < WIDTH / 4 - DX       # the tongue's inner rows
    tip = m.front_x(rows=mid)
    beyond = (phi > 0.0) & (h > 0.0)
    most_out = m.comm.allreduce(float(h[beyond].max()) if beyond.any() else 0.0, op=MPI.MAX)
    most_in = m.comm.allreduce(float(h[phi <= 0.0].max()) if (phi <= 0.0).any() else 0.0,
                               op=MPI.MAX)
    exact = FRONT0 + u_slow * DT * steps
    PETSc.Sys.Print(f"    tongue tip {tip/1e3:.1f} km (exact {exact/1e3:.1f}); thickest "
                    f"cell beyond the front {most_out/most_in:.2f} of the thickest inside; "
                    f"handed back {tot['moved']:.3e} m3")
    assert abs(resid) < 1e-6 * H0 * U * WIDTH * DT * steps
    # the reinitialisation's contour reset rounds a tip four cells wide by
    # about 4 km in these 80 yr (icepack_tools.levelset, _mark_interface_cells)
    assert abs(tip - exact) <= 1.0 * DX, tip
    assert most_out <= most_in, most_out / most_in


def test_a_detached_raft_is_an_iceberg():
    core, gap, raft = 60e3, 100e3, 140e3   # ice < core grounded, water, raft

    def h_init(x):
        return fd.conditional(fd.Or(x[0] < Constant(gap - 20e3),
                                    fd.And(x[0] > Constant(gap + 20e3), x[0] < Constant(raft))),
                              Constant(H0), Constant(0.0))

    m = Strip(u=0.0, h_init=h_init, land=lambda xc: xc[:, 0] < core, kill_icebergs=True)
    x = m.xc[:, 0]
    in_raft = (x > gap + 20e3) & (x < raft)
    sheet = x < gap - 20e3
    v_raft = m.total(np.where(in_raft, m.h.dat.data_ro, 0.0))
    v_sheet = m.total(np.where(sheet, m.h.dat.data_ro, 0.0))
    tal = m.step(DT, Constant(0.0))
    left_raft = m.total(np.where(in_raft, m.h.dat.data_ro, 0.0))
    left_sheet = m.total(np.where(sheet, m.h.dat.data_ro, 0.0))
    PETSc.Sys.Print(f"    raft {v_raft:.3e} m3 -> {left_raft:.1e}; iceberg booked "
                    f"{tal['berg']:.3e} m3; sheet {v_sheet:.6e} -> {left_sheet:.6e} m3")
    # the raft leaves with the accumulation the step put on it
    with_smb = v_raft + SMB * DT * m.total(in_raft.astype(float))
    assert v_raft > 0.0 and left_raft == 0.0
    assert abs(tal["berg"] - with_smb) <= 1e-9 * with_smb
    assert np.all(m.ls.phi.dat.data_ro[in_raft] > 0.0)
    # the sheet only gains its accumulation
    assert abs(left_sheet - v_sheet - SMB * DT * m.total(sheet.astype(float))) < 1e-6 * v_sheet
    # a connected sheet is a no-op for the iceberg pass
    m2 = Strip(u=0.0, land=lambda xc: xc[:, 0] < core)
    berg = m2.ft.icebergs(m2.h.dat.data_ro, m2.grounded)
    assert m2.comm.allreduce(int(berg.sum())) == 0


class StripDG1(Strip):
    r"""The strip with a DG1 thickness on vertex nodes.  The level set reads
    the cell means (a DG0 copy refreshed every step), the rules act on whole
    cells (a fraction scales the cell, a hand-back shifts it), and the
    positivity floor scales a cell's slope until its lowest vertex sits at
    zero, which keeps its mean (Zhang and Shu 2010)."""

    def __init__(self, u=U):
        mesh = fd.RectangleMesh(int(LENGTH / DX), int(WIDTH / DX), LENGTH, WIDTH)
        self.mesh, self.comm = mesh, mesh.comm
        Q0 = self.Q0 = FunctionSpace(mesh, "DG", 0)
        Q1 = FunctionSpace(mesh, fd.FiniteElement("DG", mesh.ufl_cell(), 1,
                                                  variant="equispaced"))
        x = fd.SpatialCoordinate(mesh)
        self.u = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(
            fd.as_vector((Constant(u), Constant(0.0))))
        # the DG0 strip's initial state, cell by cell: interpolating the step
        # at the vertices would make the front column half as thick
        h0 = Function(Q0).interpolate(
            fd.conditional(x[0] < FRONT0, Constant(H0), Constant(0.0)))
        self.h1 = Function(Q1, name="thickness_dg1").interpolate(h0)
        n0 = Q0.dof_dset.size
        cell0 = Q0.cell_node_map().values[:, 0]
        cells1 = Q1.cell_node_map().values
        self.dofs = np.empty((n0, 3), dtype=cells1.dtype)
        self.dofs[cell0] = cells1[:cell0.size]
        self.h = Function(Q0, name="thickness_means")     # what the level set reads
        self._means()
        self.ls = LevelSet(mesh, self.h, law="prescribed", h_min=1.0, anchor="advect")
        self.ls.phi.interpolate(x[0] - Constant(FRONT0))
        self.ls.update_cell_fields()
        self.xc, self.area = self.ls.cell_xc, self.ls.cell_area
        self.grounded = np.zeros(len(self.area), bool)
        self.kill_icebergs = False
        self.ft = FrontTransport(self.ls, self.u)
        self.h_old = Function(Q1)
        self.src = Function(Q1)
        self.dt_c = Constant(1.0)
        ht, w = TrialFunction(Q1), TestFunction(Q1)
        F = ((ht - self.h_old) / self.dt_c * w * dx
             + self.ft.divergence(ht, w) - self.src * w * dx)
        self.solver = fd.LinearVariationalSolver(
            fd.LinearVariationalProblem(fd.lhs(F), fd.rhs(F), self.h1),
            solver_parameters=LU)

    def _means(self):
        self.h.dat.data[:] = self.h1.dat.data_ro[self.dofs].mean(axis=1)
        return self.h.dat.data_ro

    def volume(self):
        return float(assemble(self.h1 * dx))        # a 0-form: already global

    def step(self, dt, rate):
        self.h_old.assign(self.h1)
        self.ft.advance(dt, rate)
        inside = self.ft.chi_in.dat.data_ro > 0.5
        d = self.src.dat.data
        d[:] = 0.0
        d[self.dofs[inside]] = SMB
        smb = self.total(np.where(inside, SMB, 0.0)) * dt
        self.dt_c.assign(dt)
        self.solver.solve()
        out = float(assemble(self.ft.outflow(self.h1))) * dt
        calving = float(assemble(self.ft.calving(self.h1))) * dt
        # positivity: a mean below zero fills to zero (booked), a vertex below
        # zero scales the slope (moves nothing)
        v = self.h1.dat.data_ro[self.dofs]
        mean = v.mean(axis=1)
        clamp = self.total(np.maximum(-mean, 0.0))
        v = np.where((mean < 0.0)[:, None], 0.0, v)
        mean = np.maximum(mean, 0.0)
        low = v.min(axis=1)
        steep = low < 0.0
        theta = np.where(steep, mean / np.where(steep, mean - low, 1.0), 1.0)
        v = np.maximum(mean[:, None] + theta[:, None] * (v - mean[:, None]), 0.0)
        self.h1.dat.data[self.dofs] = v
        keep, band, touch = self.ft.rules()
        m = self._means()
        calving += self.total(m * (1.0 - keep))
        self.h1.dat.data[self.dofs] = self.h1.dat.data_ro[self.dofs] * keep[:, None]
        give, back, moved = self.ft.excess(self._means(), band, touch)
        v = self.h1.dat.data_ro[self.dofs] * (1.0 - give)[:, None] + back[:, None]
        self.h1.dat.data[self.dofs] = v
        self._means()
        return dict(smb=smb, calving=calving, berg=0.0, outflux=out, clamp=clamp,
                    moved=moved)


def test_a_dg1_thickness_follows_the_same_front():
    m = StripDG1()
    tot, resid = m.run(STEPS, Constant(0.0))
    passing = H0 * U * WIDTH * DT * STEPS
    xf = m.front_x()
    behind = [m.thickness_behind(k) for k in (1, 2, 4)]
    PETSc.Sys.Print(
        f"    free: front {xf/1e3:.1f} km (exact {(FRONT0 + U*DT*STEPS)/1e3:.1f}); calving "
        f"{tot['calving']/passing:.3%}; thickness behind the front "
        + " ".join(f"{b:.3f}" for b in behind) + f"; clamp {tot['clamp']:.2e} m3, "
        f"budget residual {resid:.2e} m3")
    assert abs(resid) < 1e-6 * passing
    assert tot["calving"] < 1e-3 * passing, tot["calving"] / passing
    assert abs(tot["clamp"]) < 1e-6 * passing
    assert abs(xf - (FRONT0 + U * DT * STEPS)) <= 0.2 * DX
    assert behind[0] > 0.8 and behind[1] > 0.9 and behind[2] > 0.98, behind
    m = StripDG1()
    tot, resid = m.run(STEPS, Constant(U))
    xf = m.front_x()
    PETSc.Sys.Print(f"    held: front {xf/1e3:.1f} km (exact {FRONT0/1e3:.1f}); calving "
                    f"{tot['calving']/passing:.3f} of H U W t")
    assert abs(resid) < 1e-6 * passing
    assert abs(xf - FRONT0) <= 0.2 * DX
    assert abs(tot["calving"] / passing - 1.0) < 0.05


def test_an_extent_anchor_is_refused():
    mesh = fd.RectangleMesh(10, 4, 50e3, 20e3)
    Q0 = FunctionSpace(mesh, "DG", 0)
    h = Function(Q0).assign(100.0)
    ls = LevelSet(mesh, h, law="none", anchor="extent")
    u = Function(VectorFunctionSpace(mesh, "CG", 1))
    try:
        FrontTransport(ls, u)
    except ValueError as e:
        assert "anchor" in str(e)
    else:
        raise AssertionError("an extent-anchored level set was accepted")


def main():
    tests = [
        ("a free advance keeps its ice", test_free_advance_keeps_its_ice),
        ("a held front calves what reaches it", test_a_held_front_calves_what_reaches_it),
        ("a retreat calves at the calving rate", test_a_retreat_calves_at_the_calving_rate),
        ("the per-cell calving sums to the total",
         test_the_per_cell_calving_sums_to_the_total),
        ("a barrier holds an advance and not a retreat",
         test_a_barrier_holds_an_advance_and_not_a_retreat),
        ("an enclosed pocket of water is filled", test_an_enclosed_pocket_of_water_is_filled),
        ("a grounded tongue holds and piles nothing ahead",
         test_a_grounded_tongue_holds_and_piles_nothing_ahead),
        ("a detached raft is an iceberg", test_a_detached_raft_is_an_iceberg),
        ("a DG1 thickness follows the same front", test_a_dg1_thickness_follows_the_same_front),
        ("an extent anchor is refused", test_an_extent_anchor_is_refused),
    ]
    size = fd.COMM_WORLD.size
    for i, (label, fn) in enumerate(tests, 1):
        PETSc.Sys.Print(f"\n{i}. {label}" + (f"  [{size} ranks]" if size > 1 else ""))
        fn()
        PETSc.Sys.Print("   ok")
        gc.collect()
        PETSc.garbage_cleanup(fd.COMM_WORLD)
    if size > 1 or os.environ.get("ICEPACK_TOOLS_MPI_CHILD"):
        PETSc.Sys.Print(f"\nPASS on {size} ranks")
        return 0

    PETSc.Sys.Print(f"\n{len(tests) + 1}. the same on three ranks")
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        print("  [skip] no mpiexec on PATH -- the parallel half of this test did not run")
        return 0
    env = dict(os.environ, ICEPACK_TOOLS_MPI_CHILD="1", OMP_NUM_THREADS="1")
    r = subprocess.run([mpiexec, "-n", "3", sys.executable, "-u",
                        os.path.abspath(__file__)], env=env, timeout=900)
    if r.returncode:
        raise SystemExit(f"the 3-rank run failed with {r.returncode}")
    print("\nPASS: front transport, serial and on 3 ranks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
