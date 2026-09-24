r"""Calving laws against the solver, against hand algebra and on the front.

1. The membrane-stress convention: solve icepack2's analytic unconfined
   shelf (thickness linear in x, ``eps_xx = A (rho' g h / 4)^n``) with
   ``dual_residual`` and check that the deviatoric stress read from ``M``
   is ``rho' g h / 4`` in ``xx`` and zero in ``yy``, and that the freely
   spreading shelf sits exactly at the horizontal-force-balance threshold
   and at half the zero-stress one.  This is what makes the stress laws
   read the right number.
2. The stress-based von Mises rate on a prescribed uniaxial state matches
   ``|u| sqrt(3/2) tau / sigma_max`` and picks the grounded / floating
   threshold by height above flotation.
3. The strain-rate form reproduces Morlighem's formula in uniform
   extension with the state's own fluidity, a field as well as a number,
   and refuses a state that has none.
4. The horizontal-force-balance threshold and law, with every density
   taken from the state: its seawater is the crevasse water by default.
5. The minimum-thickness law holds a front at ``Hc``, never calves
   backwards, and -- driven through the level set -- calves marine fronts
   and spares a land margin and a nunatak.
6. :class:`FrontState` is the host model's flotation test and the level
   set's current normal: a law reads the normal of the extent before the
   first advance, and the stationary-front rate holds the front still.
7. The registry builds every law, refuses a misspelt or impossible
   parameter when the law is made, parses parameters, records them at
   full precision, and takes a law registered from a file.

    python -u calving_test.py        (or pytest)
"""
import os
import tempfile

import numpy as np
import pytest

import firedrake as fd
from firedrake import (Constant, DirichletBC, Function, FunctionSpace,
                       NonlinearVariationalProblem, NonlinearVariationalSolver,
                       SpatialCoordinate, TensorFunctionSpace,
                       VectorFunctionSpace, as_vector)

from icepack_tools import calving
from icepack_tools.constants import ice_density, water_density, gravity
from icepack_tools.levelset import LevelSet
from icepack_tools.momentum import dual_residual
from icepack_tools.spaces import dual_function_space

SPARAMS = {
    "snes_type": "newtonls", "snes_max_it": 200,
    "snes_divergence_tolerance": -1,
    "snes_rtol": 1e-10, "snes_atol": 1e-12, "snes_stol": 0.0,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 200, "mat_mumps_icntl_24": 1, "mat_mumps_cntl_3": 1e-6,
}
FC = {"quadrature_degree": 4}


class Stub:
    r"""Just enough of a front state for a law to read."""

    def __init__(self, mesh, u, M, h, haf, chi_gr, nfront=None, b=None,
                 rho_i=None, rho_w=None, A=None, n=None):
        self.mesh, self.u, self.M, self.h, self.haf, self.chi_gr = mesh, u, M, h, haf, chi_gr
        self.nfront, self.b, self.A, self.n = nfront, b, A, n
        if rho_i is not None:
            self.rho_i = rho_i
        if rho_w is not None:
            self.rho_w = rho_w


def cell_values(expr, mesh):
    return Function(FunctionSpace(mesh, "DG", 0)).interpolate(expr).dat.data_ro


def test_membrane_stress_convention_on_analytic_shelf():
    Lx, Ly = 20e3, 20e3
    h0, dh, u0 = 500.0, 100.0, 100.0
    n_glen, A = 3.0, 2.9377
    mesh = fd.RectangleMesh(32, 32, Lx, Ly, diagonal="crossed")
    x, y = SpatialCoordinate(mesh)
    Q = FunctionSpace(mesh, "CG", 1)
    Q0 = FunctionSpace(mesh, "DG", 0)
    h = Function(Q).interpolate(Constant(h0) - Constant(dh) * x / Constant(Lx))
    b = Function(Q).interpolate(Constant(-3000.0))                    # afloat everywhere
    rr = float(ice_density / water_density)
    s = Function(Q).interpolate(Constant(1.0 - rr) * h)
    Z = dual_function_space(mesh, 1)
    z = Function(Z)
    rho_p = ice_density * (1 - rr)

    def exact_u(xx):
        P = rho_p * gravity * (h0 - dh * xx / Lx) / 4
        P0, dP = rho_p * gravity * h0 / 4, rho_p * gravity * dh / 4
        return u0 + Lx * A * (P0 ** (n_glen + 1) - P ** (n_glen + 1)) / ((n_glen + 1) * dP)

    z.subfunctions[0].interpolate(as_vector((exact_u(x), Constant(0.0))))
    n_c = Constant(1.0)
    theta, phi = Function(Q), Function(Q)
    F = dual_residual(z, theta, phi, H=h, s=s, b=b, h_layers=[h], C_w0=Constant(0.01),
                      A_layers=[Constant(A)], n_consts=[n_c], n_vals=[n_glen],
                      m_slide=3.0, mesh=mesh, law="budd", alpha=1e-6, h_visc_floor=0.0,
                      outflow_ids=(2,))
    bcs = [DirichletBC(Z.sub(0), Constant((u0, 0.0)), (1,)),
           DirichletBC(Z.sub(0).sub(1), 0.0, (3, 4))]
    solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(F, z, bcs=bcs, form_compiler_parameters=FC),
        solver_parameters=SPARAMS)
    for nn in (1.0, 2.0, 3.0):            # no diffusion creep here, so ramp n
        n_c.assign(nn)
        solver.solve()

    u, M, tau = z.subfunctions
    u_ex = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(as_vector((exact_u(x), 0.0)))
    err_u = fd.norm(u - u_ex) / fd.norm(u_ex)
    dev = calving.deviatoric_stress(M)
    txx = cell_values(dev[0, 0], mesh)
    tyy = cell_values(dev[1, 1], mesh)
    expect = cell_values(rho_p * gravity * h / 4, mesh)
    xc = cell_values(x, mesh)
    interior = (xc > 0.15 * Lx) & (xc < 0.85 * Lx)
    rel = np.abs(txx[interior] - expect[interior]) / expect[interior]
    assert err_u < 1e-2
    assert rel.max() < 2e-2
    assert np.abs(tyy).max() < 2e-2 * expect.mean()
    # the tensile von Mises stress is sqrt(3/2) tau_xx for uniaxial tension
    svm = cell_values(calving.tensile_von_mises_stress(M), mesh)
    assert np.allclose(svm[interior], np.sqrt(1.5) * txx[interior], rtol=2e-2)
    # and icepack2's invariant is the effective stress: |M| = tau_xx here
    M2 = (fd.inner(M, M) - fd.tr(M) ** 2 / 3) / 2
    assert np.allclose(cell_values(fd.sqrt(M2), mesh)[interior], txx[interior], rtol=2e-2)

    # Horizontal force balance: a freely spreading shelf sits exactly at the
    # HFB threshold (Buck 2023; Coffey & Lai 2025 B = 0), and at half the
    # zero-stress one, with the shelf's own densities.
    haf = Function(Q0).assign(-100.0)
    hq = Function(Q0).interpolate(h)
    R = cell_values(calving.principal_values(M)[0], mesh)
    for mode, expect_ratio in (("hfb", 1.0), ("zero_stress", 0.5)):
        Rc = cell_values(calving.hfb_critical_stress(hq, haf, 0.0, water_density, mode,
                                                     ice_density), mesh)
        ratio = R[interior] / Rc[interior]
        assert np.allclose(ratio, expect_ratio, rtol=3e-3), (mode, ratio.mean())


def _uniform(mesh, value, space="DG"):
    return Function(FunctionSpace(mesh, space, 0)).assign(value)


def test_vonmises_rate_uniaxial_and_thresholds():
    mesh = fd.UnitSquareMesh(4, 4)
    Sig = TensorFunctionSpace(mesh, "DG", 0, symmetry=True)
    V = VectorFunctionSpace(mesh, "CG", 1)
    tau = 0.2                                     # MPa deviatoric tension in x
    M = Function(Sig).interpolate(as_vector(((2 * tau, 0.0), (0.0, tau))))   # tau_h = diag(tau, 0)
    u = Function(V).interpolate(as_vector((Constant(300.0), Constant(400.0))))   # |u| = 500
    h = _uniform(mesh, 400.0)
    haf = _uniform(mesh, 0.0)
    chi_gr = _uniform(mesh, 0.0)
    law = calving.make("vonmises", sigma_max_gr=1.0, sigma_max_fl=0.15)
    for haf_val, sig_max in ((-50.0, 0.15), (+50.0, 1.0)):
        haf.assign(haf_val)
        chi_gr.assign(1.0 if haf_val > 0 else 0.0)
        c = cell_values(law.rate(Stub(mesh, u, M, h, haf, chi_gr), 0.0), mesh)
        assert np.allclose(c, 500.0 * np.sqrt(1.5) * tau / sig_max, rtol=1e-9)
    # compression calves nothing
    M.interpolate(as_vector(((-2 * tau, 0.0), (0.0, -tau))))
    c = cell_values(law.rate(Stub(mesh, u, M, h, haf, chi_gr), 0.0), mesh)
    assert np.allclose(c, 0.0, atol=1e-6)
    # pure shear: one tensile principal stress of tau
    M.interpolate(as_vector(((0.0, tau), (tau, 0.0))))
    c = cell_values(law.rate(Stub(mesh, u, M, h, haf, chi_gr), 0.0), mesh)
    assert np.allclose(c, 500.0 * np.sqrt(1.5) * tau / 1.0, rtol=1e-9)
    # the grounded gate zeroes grounded cells when asked
    law.grounded_gate = True
    c = cell_values(law.rate(Stub(mesh, u, M, h, haf, chi_gr), 0.0), mesh)
    assert np.allclose(c, 0.0)


def test_vonmises_strain_matches_morlighem():
    mesh = fd.UnitSquareMesh(4, 4)
    mesh.coordinates.dat.data[:] *= 1e4
    V = VectorFunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)
    eps, A, n = 1e-3, 2.9377, 3.0
    u = Function(V).interpolate(as_vector((eps * x[0], Constant(0.0))))
    B = A ** (-1 / n)
    expect = np.sqrt(3) * B * (np.sqrt(eps ** 2 / 2)) ** (1 / n)
    # a number and a field give the same stress
    for A_in in (A, _uniform(mesh, A)):
        sig = cell_values(calving.tensile_von_mises_strain(u, A_in, n), mesh)
        assert np.allclose(sig, expect, rtol=1e-6)
    # relation to the stress form in uniaxial extension: 2^(1/3)
    tau = B * eps ** (1 / n)          # eps_E = eps; tau_xx = B eps_E^(1/n - 1) eps
    assert np.isclose(expect / (np.sqrt(1.5) * tau), 2 ** (1 / 3), rtol=1e-6)
    # the law reads A and n off the state, and says so when they are missing
    law = calving.make("vonmises_strain", sigma_max_fl=0.15)
    haf, chi = _uniform(mesh, -10.0), _uniform(mesh, 0.0)
    c = cell_values(law.rate(Stub(mesh, u, None, None, haf, chi, A=A, n=n), 0.0), mesh)
    speed = cell_values(calving.speed(u), mesh)
    assert np.allclose(c, speed * expect / 0.15, rtol=1e-6)
    with pytest.raises(ValueError, match="fluidity"):
        law.rate(Stub(mesh, u, None, None, haf, chi), 0.0)


def test_hfb_threshold_and_law_read_their_densities_from_the_state():
    mesh = fd.UnitSquareMesh(2, 2)
    Q0 = FunctionSpace(mesh, "DG", 0)
    H, w = 500.0, 300.0
    for rho_w_si in (1024.0, 1028.0):
        rho_i, rho_w = ice_density, calving.density(rho_w_si)
        P = float(rho_i * gravity)                               # rho_i g, MPa/m
        ri = float(rho_i / rho_w)
        # grounded front: water depth w, Slater & Wagner eq. 8 with tau_b = 0
        # is the threshold itself: (1 - (rho_w/rho_i) w^2/H^2)/2
        haf = H - w / ri
        Rc = cell_values(calving.hfb_critical_stress(
            Function(Q0).assign(H), Function(Q0).assign(haf), 0.0, rho_w, "hfb", rho_i), mesh)
        assert np.allclose(Rc, P * H * (1 - (1 / ri) * (w / H) ** 2) / 2, rtol=1e-12)
        # floating with tensile strength (Slater & Wagner eq. 21 at H_ab = 0)
        sig = 0.15
        Rc = cell_values(calving.hfb_critical_stress(
            Function(Q0).assign(H), Function(Q0).assign(-50.0), sig, rho_w, "hfb", rho_i), mesh)
        st = sig / (P * H)
        expect = P * H * ((1 - ri) / 2 + (rho_w_si / (rho_w_si - 917.0)) * st ** 2 / 2)
        assert np.allclose(Rc, expect, rtol=1e-12)
        # zero stress on a shelf is twice HFB
        Rz = cell_values(calving.hfb_critical_stress(
            Function(Q0).assign(H), Function(Q0).assign(-50.0), 0.0, rho_w, "zero_stress",
            rho_i), mesh)
        assert np.allclose(Rz, P * H * (1 - ri), rtol=1e-12)

        # the law: a shelf front with M = R_crit (uniaxial: M = diag(R, R/2))
        # holds, twice that retreats at 2|u|; its default crevasse water is
        # the state's own seawater, so the same M holds under either density
        Sig = TensorFunctionSpace(mesh, "DG", 0, symmetry=True)
        V = VectorFunctionSpace(mesh, "CG", 1)
        W0 = VectorFunctionSpace(mesh, "DG", 0)
        R = P * H * (1 - ri) / 2
        M = Function(Sig).interpolate(as_vector(((R, 0.0), (0.0, R / 2))))
        u = Function(V).interpolate(as_vector((Constant(400.0), Constant(0.0))))
        n = Function(W0).interpolate(as_vector((Constant(1.0), Constant(0.0))))
        stub = Stub(mesh, u, M, Function(Q0).assign(H), Function(Q0).assign(-50.0),
                    Function(Q0), nfront=n, rho_i=rho_i, rho_w=rho_w)
        law = calving.make("hfb")
        assert np.allclose(cell_values(law.rate(stub, 0.0), mesh), 400.0, rtol=1e-9)
        # the same crevasse water named explicitly is the same law
        named = calving.make("hfb", rho_c=rho_w_si)
        assert np.allclose(cell_values(named.rate(stub, 0.0), mesh), 400.0, rtol=1e-9)
        M.interpolate(as_vector(((2 * R, 0.0), (0.0, R))))
        assert np.allclose(cell_values(law.rate(stub, 0.0), mesh), 800.0, rtol=1e-9)
        law2 = calving.make("hfb", exponent=2.0)
        assert np.allclose(cell_values(law2.rate(stub, 0.0), mesh), 1600.0, rtol=1e-9)
        # meltwater in the basal crevasse lowers the threshold on a shelf
        melt = calving.make("hfb", rho_c=1000.0)
        assert np.all(cell_values(melt.r_crit(stub), mesh)
                      < cell_values(law.r_crit(stub), mesh))

    # The default must be the FRONT-NORMAL resistive stress, the papers'
    # R_xx = 2 tau_xx + tau_yy with x normal to the front: on a state whose
    # transverse tension dominates, the two choices differ by a factor 3.
    M.interpolate(as_vector(((R, 0.0), (0.0, 3 * R))))       # n = (1, 0)
    assert np.allclose(cell_values(calving.resistive_stress(stub, "normal"), mesh), R, rtol=1e-9)
    assert np.allclose(cell_values(calving.resistive_stress(stub, "principal"), mesh), 3 * R,
                       rtol=1e-9)
    assert np.allclose(cell_values(calving.make("hfb").rate(stub, 0.0), mesh), 400.0, rtol=1e-9)
    assert np.allclose(cell_values(calving.make("hfb", stress="principal").rate(stub, 0.0), mesh),
                       1200.0, rtol=1e-9)
    assert calving.LAWS["hfb"].defaults["stress"] == "normal"
    assert {"R_xx", "R_crit", "buttressing", "crevassed_fraction", "calving_rate"} <= \
        set(calving.make("hfb").fields(stub))
    # an unbuttressed-free, fully buttressed front (no resistive stress) holds
    M.interpolate(as_vector(((0.0, 0.0), (0.0, 0.0))))
    assert np.allclose(cell_values(calving.make("hfb").rate(stub, 0.0), mesh), 0.0, atol=1e-9)


def test_hfb_ratio_cap_bounds_what_one_step_removes():
    mesh = fd.UnitSquareMesh(2, 2)
    Q0 = FunctionSpace(mesh, "DG", 0)
    Sig = TensorFunctionSpace(mesh, "DG", 0, symmetry=True)
    V = VectorFunctionSpace(mesh, "CG", 1)
    W0 = VectorFunctionSpace(mesh, "DG", 0)
    M = Function(Sig).interpolate(as_vector(((1.0, 0.0), (0.0, 0.5))))      # 1 MPa on 2 m of ice
    u = Function(V).interpolate(as_vector((Constant(100.0), Constant(0.0))))
    n = Function(W0).interpolate(as_vector((Constant(1.0), Constant(0.0))))
    stub = Stub(mesh, u, M, Function(Q0).assign(2.0), Function(Q0).assign(-1.0),
                Function(Q0), nfront=n)
    c = cell_values(calving.make("hfb", ratio_max=5.0).rate(stub, 0.0), mesh)
    assert np.allclose(c, 500.0, rtol=1e-9)


def test_thickness_law_holds_at_hc_and_never_calves_backwards():
    mesh = fd.UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, "CG", 1)
    u = Function(V).interpolate(as_vector((Constant(300.0), Constant(400.0))))   # |u| = 500
    hc = 150.0
    law = calving.make("thickness", hc=hc)
    for H, factor in ((hc, 1.0), (hc / 2, 1.5), (0.0, 2.0), (2 * hc, 0.0), (5 * hc, 0.0)):
        stub = Stub(mesh, u, None, _uniform(mesh, H), None, _uniform(mesh, 0.0),
                    b=_uniform(mesh, -500.0))
        c = cell_values(law.rate(stub, 0.0), mesh)
        assert np.allclose(c, factor * 500.0, rtol=1e-9), (H, c[0])
    with pytest.raises(ValueError, match="hc"):
        calving.make("thickness", hc=0.0)


#: One mesh, several margins. Ice fills x < 0.5 except for a nunatak hole; the
#: front at x = 0.5 is cut into bands by y, each a different margin, and every
#: band edge falls on a cell row so no front cell straddles two bands.
N = 8
HC = 150.0
U_FRONT = 0.01
_BANDS = {
    # name: (y range, thickness, bed)
    "marine_at_hc": ((0.0, 0.25), HC, -500.0),
    "marine_thin": ((0.25, 0.5), HC / 2, -500.0),
    # 100 m on a 50 m deep bed has 44 m above flotation, so it is grounded
    "grounded_cliff": ((0.5, 0.75), 100.0, -50.0),
    "land": ((0.75, 1.0), 50.0, 200.0),
}
#: The nunatak: one ice-free cell square well inside the ice, with the bed
#: above sea level under it and under the ice ringing it.
_NUNATAK_HOLE = (0.125, 0.25, 0.375, 0.5)
_NUNATAK_BED = (0.0, 0.375, 0.25, 0.625)


def _inside(box, x, y):
    x0, x1, y0, y1 = box
    return (x > x0) & (x < x1) & (y > y0) & (y < y1)


def _margins():
    mesh = fd.UnitSquareMesh(N, N)
    Q0 = FunctionSpace(mesh, "DG", 0)
    xy = Function(VectorFunctionSpace(mesh, "DG", 0)).interpolate(
        SpatialCoordinate(mesh)).dat.data_ro
    x, y = xy[:, 0], xy[:, 1]
    band = np.empty(len(x), dtype=object)
    h = Function(Q0)
    b = Function(Q0)
    for name, ((y0, y1), thickness, bed) in _BANDS.items():
        rows = (y > y0) & (y < y1)
        band[rows] = name
        h.dat.data[rows] = np.where(x[rows] < 0.5, thickness, 0.0)
        b.dat.data[rows] = bed
    hole = _inside(_NUNATAK_HOLE, x, y)
    h.dat.data[hole] = 0.0
    b.dat.data[_inside(_NUNATAK_BED, x, y)] = 20.0
    u = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(
        as_vector((Constant(U_FRONT), Constant(0.0))))
    ls = LevelSet(mesh, h, law="prescribed", h_min=1.0, anchor="extent")
    return mesh, h, b, u, ls, x, y, band


def test_thickness_law_calves_marine_fronts_and_spares_land_ones():
    r"""The bed gate, in one mesh with every kind of margin the level set
    anchors on. At ``H = Hc`` the removal is the ice's own arrival speed, so
    the front holds; thinner marine ice, a grounded cliff on a bed below sea
    level included, is removed faster than it arrives; a land margin and the
    ice ringing a nunatak, both on a bed above sea level, shed nothing."""
    mesh, h, b, u, ls, x, y, band = _margins()
    Sig = TensorFunctionSpace(mesh, "DG", 0, symmetry=True)
    state = calving.FrontState(u, Function(Sig), None, h, b, ls)
    ls.advance(1.0, u, rate=calving.make("thickness", hc=HC).rate(state, 0.0))
    beyond, frac = ls.calving_masks()
    c = ls.c_cell.dat.data_ro
    front = ls.front_len.dat.data_ro > 0.0
    ring = front & _inside(_NUNATAK_BED, x, y)
    outer = front & ~ring
    shed = h.dat.data_ro * frac * ls.cell_area
    assert not beyond.any()

    def at(name):
        cells = outer & (band == name)
        assert cells.any(), name
        return cells

    assert c[at("marine_at_hc")] == pytest.approx(U_FRONT, rel=1e-6)
    assert np.all(c[at("marine_thin")] == pytest.approx(1.5 * U_FRONT, rel=1e-6))
    assert np.all(c[at("grounded_cliff")] == pytest.approx(
        (2.0 - 100.0 / HC) * U_FRONT, rel=1e-6))
    for name in ("marine_thin", "grounded_cliff"):
        assert np.all(shed[at(name)] > 0.0), name
    assert ring.sum() >= 4
    for spared in (at("land"), ring):
        assert np.all(c[spared] == 0.0)
        assert np.all(frac[spared] == 0.0)
        assert shed[spared].sum() == 0.0
    # every calving cell is a marine one
    assert np.all(b.dat.data_ro[shed > 0.0] < 0.0)
    # and without the gate the land margin calves: the gate is doing the work
    _, h2, b2, u2, ls2, *_ = _margins()
    state2 = calving.FrontState(u2, Function(Sig), None, h2, b2, ls2)
    ls2.advance(1.0, u2, rate=calving.make("thickness", hc=HC, marine_only=False).rate(state2, 0.0))
    assert np.all(ls2.c_cell.dat.data_ro[at("land")] > 0.0)


def _slab(anchor="extent"):
    r"""Ice (300 m, afloat on a 1000 m deep bed) fills x < 0.5 of a unit
    square and flows at U along x; open water beyond."""
    mesh = fd.UnitSquareMesh(N, N)
    Q0 = FunctionSpace(mesh, "DG", 0)
    x, _ = SpatialCoordinate(mesh)
    h = Function(Q0).interpolate(fd.conditional(x < 0.5, Constant(300.0), Constant(0.0)))
    b = Function(Q0).assign(-1000.0)
    u = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(
        as_vector((Constant(U_FRONT), Constant(0.0))))
    ls = LevelSet(mesh, h, law="prescribed", h_min=1.0, anchor=anchor)
    return mesh, h, b, u, ls


def test_front_state_is_the_host_flotation_and_the_current_normal():
    mesh, h, b, u, ls = _slab()
    Sig = TensorFunctionSpace(mesh, "DG", 0, symmetry=True)
    M = Function(Sig)
    x = Function(VectorFunctionSpace(mesh, "DG", 0)).interpolate(
        SpatialCoordinate(mesh)).dat.data_ro[:, 0]
    ice = x < 0.5
    # the flotation test is the host model's: 300 m on a 250 m deep bed
    # floats under 1028 and is grounded under 917/1024's 256 m draft? no:
    # 300 - (1024/917) 250 = 20.8 m grounded, 300 - (1028/917) 250 = 19.7 m
    b.assign(-250.0)
    for rho_w_si in (1024.0, 1028.0):
        st = calving.FrontState(u, M, None, h, b, ls, rho_w=calving.density(rho_w_si))
        haf = cell_values(st.haf, mesh)
        assert np.allclose(haf[ice], 300.0 - rho_w_si / 917.0 * 250.0, rtol=1e-12)
        assert np.all(cell_values(st.chi_gr, mesh)[ice] == 1.0)
    # the normal a law reads is the level set's, current before any advance:
    # the front at x = 0.5 faces +x
    st = calving.FrontState(u, M, None, h, b, ls)
    assert st.nfront is ls.ghat
    front = ls.front_len.dat.data_ro > 0.0
    assert front.any()
    n = Function(VectorFunctionSpace(mesh, "DG", 0)).interpolate(st.nfront).dat.data_ro
    assert np.allclose(n[front], [1.0, 0.0], atol=1e-6)


def test_the_stationary_front_rate_holds_the_front_through_the_level_set():
    r"""``velocity`` (``Cr = u . n``) evaluated on the state and handed to the
    level set as its rate: what arrives calves, nothing more, so a sub-cell
    step sheds exactly the ice that crossed the front and the front holds."""
    mesh, h, b, u, ls = _slab()
    Sig = TensorFunctionSpace(mesh, "DG", 0, symmetry=True)
    st = calving.FrontState(u, Function(Sig), None, h, b, ls)
    law = calving.make("velocity")
    assert law.front_mode == "prescribed"
    dt = 1.0
    ls.advance(dt, u, rate=law.rate(st, 0.0))
    beyond, frac = ls.calving_masks()
    assert not beyond.any()
    front = ls.front_len.dat.data_ro > 0.0
    assert np.allclose(ls.c_cell.dat.data_ro[front], U_FRONT, rtol=1e-6)
    shed = float((h.dat.data_ro * frac * ls.cell_area).sum())
    assert np.isclose(shed, U_FRONT * 300.0 * 1.0 * dt, rtol=1e-6)


def test_registry_parameters_and_provenance():
    for name, cls in calving.LAWS.items():
        law = calving.make(name, **cls.defaults)
        assert law.name == name and isinstance(law.describe(), str)
        assert law.front_mode in ("prescribed", "fixed")
    assert calving.make("fixed").front_mode == "fixed"
    assert calving.make("fixed").rate(None, 0.0) is None
    with pytest.raises(ValueError, match="unknown parameter"):
        calving.make("vonmises", sigma_max_flt=0.2)
    with pytest.raises(ValueError, match="unknown calving law"):
        calving.make("vonmisses")
    # impossible values fail when the law is made, not at the first step
    for name, bad in (("vonmises", {"sigma_max_fl": 0.0}), ("hfb", {"mode": "nye"}),
                      ("hfb", {"stress": "sideways"}), ("hfb", {"rho_c": "brine"}),
                      ("velocity", {"iv": "Normal"}), ("thickness", {"hc": -1.0})):
        with pytest.raises(ValueError):
            calving.make(name, **bad)
    with pytest.raises(ValueError):
        calving.front_ice_speed(None, "sideways")
    assert set(calving.IV_CONVENTIONS) == {"normal", "speed"}
    assert calving.parse_params(["hc=300", "sigma_max_fl=0.2", "mode=zero_stress", "n=3",
                                 "rho_c=seawater", "marine_only=false", "K=3.2e9"]) == \
        {"hc": 300, "sigma_max_fl": 0.2, "mode": "zero_stress", "n": 3,
         "rho_c": "seawater", "marine_only": False, "K": 3.2e9}
    with pytest.raises(ValueError):
        calving.parse_params(["sigma_max_fl"])
    # provenance names every parameter at full precision: two thresholds
    # that differ in the seventh digit must not read the same
    a = calving.make("vonmises", sigma_max_fl=0.1234567).describe()
    b = calving.make("vonmises", sigma_max_fl=0.1234568).describe()
    assert a != b and "sigma_max_fl=0.1234567" in a and "sigma_max_gr=1.0" in a


def test_a_law_registered_from_a_file_joins_the_registry():
    source = '''
from firedrake import Constant, max_value, sym, grad
from icepack_tools import calving


@calving.register
class Eigencalving(calving.Law):
    name = "eigencalving_test"
    defaults = {"K": 3.2e9}

    def rate(self, model, t):
        e1, e2 = calving.principal_values(sym(grad(model.u)))
        return (Constant(float(self.p["K"])) * max_value(e1, Constant(0.0))
                * max_value(e2, Constant(0.0)))
'''
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "eigencalving_plugin.py")
        with open(path, "w") as f:
            f.write(source)
        calving.load_module(path)
        try:
            assert "eigencalving_test" in calving.LAWS
            mesh = fd.UnitSquareMesh(2, 2)
            mesh.coordinates.dat.data[:] *= 1e4
            x = SpatialCoordinate(mesh)
            e = 1e-3
            u = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(
                as_vector((e * x[0], 2 * e * x[1])))             # eps1 = 2e, eps2 = e
            law = calving.make("eigencalving_test", K=1e9)
            stub = Stub(mesh, u, None, None, None, None)
            assert np.allclose(cell_values(law.rate(stub, 0.0), mesh), 1e9 * 2 * e * e, rtol=1e-6)
            with pytest.raises(ValueError, match="already registered"):
                calving.load_module(path)
        finally:
            calving.LAWS.pop("eigencalving_test", None)


def main():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    for i, (name, fn) in enumerate(tests, 1):
        print(f"\n{i}. {name}")
        fn()
        print("   ok")
    print("\nPASS: calving laws")


if __name__ == "__main__":
    main()
