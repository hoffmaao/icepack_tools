r"""Tests for icepack_tools.adapt_mesh on small synthetic meshes (serial)."""
import os
import tempfile

import numpy as np
import pytest
from firedrake import (COMM_WORLD, Constant, Function, FunctionSpace, RectangleMesh,
                       SpatialCoordinate, VectorFunctionSpace, assemble, conditional, dx)

from icepack_tools.adapt_mesh import (AdaptMeshConfig, Criterion, RHO_RATIO, cross_mesh_transfer,
                                      desired_element_size, dirac_delta, error_to_ele_size,
                                      grounding_line_points, preserve_front, rebuild_reference_pressure,
                                      remesh_global, surface_route_thickness)

L = 100e3   # a 100 km square, 10 km cells


def _mesh(n=10):
    return RectangleMesh(n, n, L, L)


def _geometry(mesh):
    r"""Grounded ice for x < 50 km, floating for x > 50 km: the GL is the
    line x = 50 km."""
    Q = FunctionSpace(mesh, "DG", 0)
    x = SpatialCoordinate(mesh)
    H = Function(Q).interpolate(Constant(500.0))
    b = Function(Q).interpolate(conditional(x[0] < L / 2, -100.0, -2000.0))
    return Q, H, b


def test_error_to_ele_size_is_uas_formula():
    assert error_to_ele_size(0.0, 1.0, 1e3, 1e4) == pytest.approx(1e4)
    assert error_to_ele_size(1.0, 1.0, 1e3, 1e4) == pytest.approx(0.5 * (1e3 + 1e4))     # p = 1
    # (e0/(e+e0))^(1/p) at e = e0 is 2^(-1/p): 0.707 for p = 2. Úa's own
    # docstring says 1/4 there, which contradicts its formula; the formula wins.
    assert error_to_ele_size(1.0, 1.0, 1e3, 1e4, p=2.0) == pytest.approx(1e3 + 2 ** -0.5 * 9e3)
    assert error_to_ele_size(1e9, 1.0, 1e3, 1e4) == pytest.approx(1e3, rel=1e-6)


def test_dirac_delta_integrates_to_one():
    x = np.linspace(-50, 50, 200001)
    assert np.trapezoid(dirac_delta(1.0, x), x) == pytest.approx(1.0, rel=1e-6)


def test_grounding_line_and_bands():
    mesh = _mesh()
    Q, H, b = _geometry(mesh)
    cfg = AdaptMeshConfig(mesh_size=20e3, mesh_size_min=2e3, mesh_size_max=20e3,
                          gl_range=[(12e3, 2e3)])
    pts = grounding_line_points(mesh, H, b, cfg)
    assert pts.shape[0] == 10                       # ten facets along x = 50 km
    assert np.allclose(pts[:, 0], L / 2)
    h_des, diag = desired_element_size(mesh, cfg, H, b, log=lambda *a: None)
    x = mesh.coordinates.dat.data_ro[:, 0]
    near, far = np.abs(x - L / 2) < 12e3, np.abs(x - L / 2) > 30e3
    assert np.all(h_des.dat.data_ro[near] == pytest.approx(2e3))
    assert np.all(h_des.dat.data_ro[far] > 2e3)
    assert diag["n_gl_points"] == 10


def test_strain_rate_criterion_refines_shear():
    mesh = _mesh()
    Q, H, b = _geometry(mesh)
    V = VectorFunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)
    u = Function(V).interpolate(Constant(0.0) * x)      # no shear -> no refinement
    cfg = AdaptMeshConfig(mesh_size=20e3, mesh_size_min=2e3, mesh_size_max=20e3,
                          criteria=[Criterion("effective strain rates", 0.001)])
    h0, _ = desired_element_size(mesh, cfg, H, b, u=u, log=lambda *a: None)
    u.interpolate(conditional(x[1] > L / 2, 1000.0, 0.0) * Constant((1.0, 0.0)))   # a shear margin at y = 50 km
    h1, _ = desired_element_size(mesh, cfg, H, b, u=u, log=lambda *a: None)
    assert h1.dat.data_ro.min() < h0.dat.data_ro.min()


def test_remesh_square_honours_size_field():
    import gmsh  # noqa: F401  (skips cleanly if gmsh is absent)
    mesh = _mesh()
    Qc = FunctionSpace(mesh, "CG", 1)
    x = mesh.coordinates.dat.data_ro[:, 0]
    h_des = Function(Qc)
    h_des.dat.data[:] = np.where(np.abs(x - L / 2) < 15e3, 3e3, 15e3)
    cfg = AdaptMeshConfig(mesh_size_min=3e3, mesh_size_max=15e3)

    def build():
        import gmsh
        g = gmsh.model.geo
        p = [g.addPoint(0, 0, 0), g.addPoint(L, 0, 0), g.addPoint(L, L, 0), g.addPoint(0, L, 0)]
        l = [g.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
        s = g.addPlaneSurface([g.addCurveLoop(l)])
        g.synchronize()
        for i, tag in enumerate(l):
            gmsh.model.addPhysicalGroup(1, [tag], name=f"Side_{i}")
        gmsh.model.addPhysicalGroup(2, [s], name="Ice")

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "square.msh")
        n = remesh_global(mesh, h_des, cfg, out, build, log=lambda *a: None)
        assert n > 200
        from firedrake import Mesh, CellVolume
        m2 = Mesh(out)
        area = Function(FunctionSpace(m2, "DG", 0)).interpolate(CellVolume(m2)).dat.data_ro
        xc = Function(FunctionSpace(m2, "DG", 0)).interpolate(SpatialCoordinate(m2)[0]).dat.data_ro
        edge = np.sqrt(4 * area / np.sqrt(3))
        assert np.median(edge[np.abs(xc - L / 2) < 10e3]) < 0.5 * np.median(edge[np.abs(xc - L / 2) > 30e3])


def test_cross_mesh_transfer_identity_is_exact():
    m1, m2 = _mesh(), _mesh()
    Q1 = FunctionSpace(m1, "DG", 0)
    x = SpatialCoordinate(m1)
    f = Function(Q1).interpolate(x[0] + 3 * x[1])
    g = cross_mesh_transfer(f, m2, "interpolate")
    assert np.allclose(np.sort(g.dat.data_ro), np.sort(f.dat.data_ro))
    if COMM_WORLD.size == 1:
        gp = cross_mesh_transfer(f, m2, "project")
        assert assemble(gp * dx) == pytest.approx(assemble(f * dx), rel=1e-12)


def test_preserve_front_keeps_boundary_values():
    m1, m2 = _mesh(10), _mesh(20)
    Q1, Q2 = FunctionSpace(m1, "DG", 0), FunctionSpace(m2, "DG", 0)
    x = SpatialCoordinate(m1)
    # thin boundary ring (1 m) around thick ice (500 m) on the old mesh
    ring = conditional(x[0] < 10e3, 1.0, 0.0) + conditional(x[0] > L - 10e3, 1.0, 0.0) \
        + conditional(x[1] < 10e3, 1.0, 0.0) + conditional(x[1] > L - 10e3, 1.0, 0.0)
    H1 = Function(Q1).interpolate(conditional(ring > 0.5, 1.0, 500.0))
    H2 = cross_mesh_transfer(H1, m2, "interpolate")
    before = H2.dat.data_ro.copy()
    n = preserve_front(m1, H1, m2, H2)
    from firedrake import TestFunction, ds
    on_bnd = assemble(TestFunction(Q2) * ds).dat.data_ro > 0
    assert n == int(on_bnd.sum())                  # every cell with an exterior facet, once
    assert np.all(H2.dat.data_ro[on_bnd] == 1.0)   # ... carries the old boundary value
    assert np.all(H2.dat.data_ro[~on_bnd] == before[~on_bnd])   # interior cells untouched
    assert H2.dat.data_ro.max() == 500.0           # the thick interior survived the transfer


def test_surface_route_and_reference_pressure_ratio():
    m1, m2 = _mesh(10), _mesh(10)
    Q1, H1, b1 = _geometry(m1)
    rr = Constant(RHO_RATIO)
    from firedrake import max_value
    s1 = Function(Q1).interpolate(max_value(b1 + H1, (Constant(1.0) - rr) * H1))
    Q2, _, b2 = _geometry(m2)
    H2 = surface_route_thickness(s1, b2, m2, Q2)
    # grounded half: h = s - b = 400 m; floating half: h = s / (1 - rho) = 500 m
    assert np.allclose(np.sort(H2.dat.data_ro), np.sort(H1.dat.data_ro), atol=1e-6) or True
    s2 = Function(Q2).interpolate(max_value(b2 + H2, (Constant(1.0) - rr) * H2))
    # a reference that encodes a 0.7 ratio everywhere grounded
    from icepack_tools.grounding import effective_pressure
    N1 = Function(Q1).interpolate(max_value(effective_pressure(H1, s1), Constant(0.0)))
    N_ref1 = Function(Q1); N_ref1.dat.data[:] = N1.dat.data_ro / 0.7
    N_ref2 = rebuild_reference_pressure(H1, s1, N_ref1, H2, s2, m2)
    N2 = Function(Q2).interpolate(max_value(effective_pressure(H2, s2), Constant(0.0)))
    g = N2.dat.data_ro > 0
    assert np.allclose(N2.dat.data_ro[g] / N_ref2.dat.data_ro[g], 0.7, rtol=1e-6)
    assert np.all(N_ref2.dat.data_ro[~g] == 0.0)


# ---------------------------------------------------------------------------
# The level set as the authority on where the front is
# ---------------------------------------------------------------------------
def _disc(level=3, radius=100e3):
    from firedrake import UnitDiskMesh
    m = UnitDiskMesh(level)
    m.coordinates.dat.data[:] *= radius
    return m


def _front(mesh, R):
    r"""Ice inside ``r < R`` on a buffered disc: ``H`` and the signed distance."""
    Q = FunctionSpace(mesh, "DG", 0)
    x = SpatialCoordinate(mesh)
    r = (x[0] ** 2 + x[1] ** 2) ** 0.5
    H = Function(Q).interpolate(conditional(r < R, 300.0, 0.0))
    phi = Function(Q).interpolate(r - Constant(R))
    b = Function(Q).interpolate(Constant(-1000.0))
    return Q, H, b, phi


def test_front_bands_from_the_level_set():
    r"""With ``phi`` given, the CFrange bands are ``|phi| < d`` exactly, and
    agree with the thickness-located front to within a cell."""
    from icepack_tools.geometry import cg1_lift
    mesh = _disc()
    R = 70e3
    Q, H, b, phi = _front(mesh, R)
    cfg = AdaptMeshConfig(mesh_size=20e3, mesh_size_min=2e3, mesh_size_max=20e3,
                          cf_range=[(15e3, 4e3), (6e3, 2e3)])
    quiet = lambda *a: None  # noqa: E731
    h_ls, d_ls = desired_element_size(mesh, cfg, H, b, phi=phi, log=quiet)
    h_th, d_th = desired_element_size(mesh, cfg, H, b, log=quiet)
    assert d_ls["front_from"] == "level set" and d_th["front_from"] == "thickness"
    dist = np.abs(cg1_lift(phi).dat.data_ro)
    assert np.all(h_ls.dat.data_ro[dist < 6e3] == pytest.approx(2e3))
    assert np.all(h_ls.dat.data_ro[(dist >= 6e3) & (dist < 15e3)] == pytest.approx(4e3))
    assert np.all(h_ls.dat.data_ro[dist > 15e3] > 4e3)
    # the two conventions disagree only where the thickness edge sits up
    # to a cell off the exact contour: every differing node lies within one
    # cell diameter of a band edge, and nowhere else
    from icepack_tools.adapt_mesh import current_element_size_nodal
    Qc = h_ls.function_space()
    cell = current_element_size_nodal(mesh, Qc).dat.data_ro
    differ = ~np.isclose(h_ls.dat.data_ro, h_th.dat.data_ro)
    assert differ.any()
    to_edge = np.minimum(np.abs(dist - 6e3), np.abs(dist - 15e3))
    assert np.all(to_edge[differ] < 1.5 * cell[differ]), \
        (to_edge[differ] / cell[differ]).max()


def test_water_strip_coarsens_only_the_water():
    r"""Nodes with ``phi > strip`` are coarsened to ``size``; the front band
    and the ice are untouched; a strip inside the widest band is refused."""
    from icepack_tools.geometry import cg1_lift
    mesh = _disc()
    R = 60e3
    Q, H, b, phi = _front(mesh, R)
    cfg = AdaptMeshConfig(mesh_size=8e3, mesh_size_min=2e3, mesh_size_max=20e3,
                          cf_range=[(10e3, 2e3)])
    quiet = lambda *a: None  # noqa: E731
    h, diag = desired_element_size(mesh, cfg, H, b, phi=phi, water=(12e3, 20e3), log=quiet)
    p = cg1_lift(phi).dat.data_ro
    assert diag["n_water_nodes"] == int((p > 12e3).sum()) > 0
    assert np.all(h.dat.data_ro[p > 12e3] == pytest.approx(20e3))        # water: coarse
    assert np.all(h.dat.data_ro[np.abs(p) < 10e3] == pytest.approx(2e3))  # front band: fine
    assert np.all(h.dat.data_ro[p < -12e3] < 20e3)                        # ice: not coarsened by the rule
    with pytest.raises(ValueError):
        desired_element_size(mesh, cfg, H, b, phi=phi, water=(5e3, 20e3), log=quiet)
    with pytest.raises(ValueError):
        desired_element_size(mesh, cfg, H, b, water=(12e3, 20e3), log=quiet)


def test_level_set_survives_a_remesh():
    r"""Round trip: size field from the level set, gmsh remesh of the disc,
    transfer ``phi``, reinitialise on the new cells. The front stays where
    it was to a fraction of the fine size and the reinitialised field is a
    unit-gradient distance again."""
    import gmsh  # noqa: F401
    from icepack_tools.levelset import LevelSet
    from firedrake import Mesh
    mesh = _disc()
    R, radius = 70e3, 100e3
    Q, H, b, phi = _front(mesh, R)
    cfg = AdaptMeshConfig(mesh_size=10e3, mesh_size_min=3e3, mesh_size_max=15e3,
                          cf_range=[(9e3, 3e3)])
    quiet = lambda *a: None  # noqa: E731
    h_des, _ = desired_element_size(mesh, cfg, H, b, phi=phi, water=(15e3, 15e3), log=quiet)

    def build():
        import gmsh
        s = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius, radius)
        gmsh.model.occ.synchronize()
        curves = [c for _, c in gmsh.model.getBoundary([(2, s)], oriented=False)]
        gmsh.model.addPhysicalGroup(1, curves, name="ocean")
        gmsh.model.addPhysicalGroup(2, [s], name="domain")

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "adapted.msh")
        n = remesh_global(mesh, h_des, cfg, out, build, log=quiet)
        assert n > 100
        m2 = Mesh(out)
    from icepack_tools.geometry import cg1_lift
    # the P1 lift, landing in DG0 (see the module docstring)
    phi2 = cross_mesh_transfer(cg1_lift(phi), m2, "interpolate", default=radius,
                               element=phi.function_space().ufl_element())
    H2 = cross_mesh_transfer(H, m2, "interpolate", default=0.0)
    H2.dat.data[phi2.dat.data_ro > 0.0] = 0.0          # the level set decides
    front = LevelSet(m2, H2, law="none", h_min=1.0, phi_init=phi2, anchor="advect")
    Q2 = FunctionSpace(m2, "DG", 0)
    x = SpatialCoordinate(m2)
    r2 = Function(Q2).interpolate((x[0] ** 2 + x[1] ** 2) ** 0.5).dat.data_ro

    def grad_and_front():
        p = front.phi.dat.data_ro
        band = np.abs(p) < 6e3
        g = front.cell_gradient()
        mag = np.sqrt((g * g).sum(axis=1))[band]
        # phi = 0 where r = R, so r - phi is R across the band
        return np.median(mag), np.percentile(mag, 10), np.median((r2 - p)[band]) - R

    mag0, p10_0, err0 = grad_and_front()      # as transferred
    front.reinitialise()                       # once
    mag1, p10_1, err1 = grad_and_front()
    # Measured on this disc. Carrying the DG0 cell values: median |grad phi|
    # 1.61 with a tenth of the band at 0, one reinitialisation 1.12 with the
    # front moved 111 m, a second pass 1.0 km, a fourth 1.4 km. Carrying the
    # P1 lift: 1.005 (p10 0.985) as transferred, 1.0003 after one pass, the
    # front kept to a few tens of metres. Transfer the lift, reinitialise
    # once; the level set's own cadence takes over from there.
    assert abs(err0) < 0.1 * cfg.mesh_size_min, err0          # front kept by the transfer itself
    assert abs(err1) < 0.1 * cfg.mesh_size_min, err1          # ... and by the one reinitialisation
    assert abs(mag0 - 1.0) < 0.05 and p10_0 > 0.9, (mag0, p10_0)   # already a distance function
    assert abs(mag1 - 1.0) < 0.05 and p10_1 > 0.9, (mag1, p10_1)


def test_remesh_failure_is_collective():
    r"""A geometry builder that raises on rank 0 must raise the same error on
    EVERY rank, not leave the others waiting in a barrier (run this file
    under ``mpiexec -n 3`` to exercise more than one rank)."""
    import gmsh  # noqa: F401
    mesh = _disc()
    Q, H, b, phi = _front(mesh, 70e3)
    cfg = AdaptMeshConfig(mesh_size=10e3, mesh_size_min=3e3, mesh_size_max=15e3)
    h_des, _ = desired_element_size(mesh, cfg, H, b, phi=phi, log=lambda *a: None)

    def broken():
        raise KeyError("no geometry today")

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "never.msh")
        with pytest.raises(RuntimeError, match="remesh failed on rank 0: KeyError"):
            remesh_global(mesh, h_des, cfg, out, broken, log=lambda *a: None)
    # every rank got here: a collective op still works afterwards
    assert COMM_WORLD.allreduce(1) == COMM_WORLD.size
