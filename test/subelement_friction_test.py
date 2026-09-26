r"""Sub-element grounded friction (ISSM's SEP2, Seroussi et al. 2014).

1. The degree-2 points on the grounded part of a triangle integrate every
   polynomial of degree <= 2 over that part exactly, on random triangles
   and random linear flotation fields: area, first and second moments
   against the exact polygon integrals of the clipped part.
2. A straight grounding line swept across a mesh gives a grounded area
   exactly linear in its position -- no cell-by-cell staircase, which is
   what a cell-constant flotation test produces.
3. The friction closure built with ``subelement`` gives a basal force equal
   to the exact integral of the friction over the grounded part: with a
   linear velocity and a linear law the whole-domain force is
   ``C int_{x < s} u dx``, for grounding lines inside cells as well as on
   their edges.
4. ``update(ice=...)`` gives a cell with no ice no grounded part.

    python -u subelement_friction_test.py        (or pytest)
"""
import numpy as np

import firedrake as fd
from firedrake import Constant, Function, FunctionSpace, VectorFunctionSpace

from icepack_tools.grounding import SubelementGrounding, grounded_quadrature
from icepack_tools.momentum import _subelement_friction


def _clip(poly, a, b, c):
    r"""Sutherland-Hodgman: the part of ``poly`` where ``a x + b y + c > 0``."""
    out = []
    n = len(poly)
    for i in range(n):
        P, Q = poly[i], poly[(i + 1) % n]
        fP, fQ = a * P[0] + b * P[1] + c, a * Q[0] + b * Q[1] + c
        if fP > 0:
            out.append(P)
        if (fP > 0) != (fQ > 0):
            t = fP / (fP - fQ)
            out.append(P + t * (Q - P))
    return out


def _moments(poly):
    r"""Exact integrals of 1, x, y, x^2, xy, y^2 over a simple polygon
    (Green's theorem on each edge)."""
    m = np.zeros(6)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cr = x0 * y1 - x1 * y0
        m[0] += cr / 2
        m[1] += (x0 + x1) * cr / 6
        m[2] += (y0 + y1) * cr / 6
        m[3] += (x0 * x0 + x0 * x1 + x1 * x1) * cr / 12
        m[4] += (x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * cr / 24
        m[5] += (y0 * y0 + y0 * y1 + y1 * y1) * cr / 12
    return m


def test_the_grounded_points_integrate_quadratics_exactly():
    rng = np.random.default_rng(7)
    m = 400
    xy = rng.uniform(-1.0, 1.0, (m, 3, 2))
    area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
    xy, area = xy[area > 1e-3], area[area > 1e-3]
    abc = rng.normal(size=(len(xy), 3))
    phi = abc[:, 0:1] * xy[:, :, 0] + abc[:, 1:2] * xy[:, :, 1] + abc[:, 2:3]
    pts, w, cen, full = grounded_quadrature(phi, xy)
    partial = 0
    for k in range(len(xy)):
        poly = [xy[k, i] for i in range(3)]
        if (0.5 * ((poly[1][0] - poly[0][0]) * (poly[2][1] - poly[0][1])
                   - (poly[2][0] - poly[0][0]) * (poly[1][1] - poly[0][1]))) < 0:
            poly = poly[::-1]
        g = _clip(poly, *abc[k])
        exact = _moments(g) if len(g) >= 3 else np.zeros(6)
        if full[k]:
            assert np.isclose(exact[0], area[k], rtol=1e-10)
            assert np.allclose(w[k], 0.0)
            continue
        partial += 0 < exact[0] < area[k]
        X, Y = pts[k, :, 0], pts[k, :, 1]
        quad = area[k] * np.array([w[k].sum(), w[k] @ X, w[k] @ Y,
                                   w[k] @ X ** 2, w[k] @ (X * Y), w[k] @ Y ** 2])
        assert np.allclose(quad, exact, rtol=1e-9, atol=1e-12), (k, quad, exact)
        if exact[0] > 0:
            assert np.allclose(cen[k], exact[1:3] / exact[0], rtol=1e-9, atol=1e-12)
    assert partial > 100, partial              # the cut cells were exercised


def _strip(nx=20, ny=6, L=100e3, W=30e3):
    mesh = fd.RectangleMesh(nx, ny, L, W, diagonal="crossed")
    return mesh, L, W


def test_a_swept_grounding_line_grows_the_grounded_area_linearly():
    mesh, L, W = _strip()
    sub = SubelementGrounding(mesh)
    Q0 = FunctionSpace(mesh, "DG", 0)
    area = fd.assemble(fd.TestFunction(Q0) * fd.dx).dat.data_ro
    x = fd.SpatialCoordinate(mesh)
    comm = mesh.comm
    for s in np.linspace(3.1e3, 96.7e3, 23):
        sub.update(Constant(s) - x[0])
        ag = comm.allreduce(float((sub.fraction.dat.data_ro * area).sum()))
        assert np.isclose(ag, s * W, rtol=1e-12), (s, ag, s * W)


def test_the_basal_force_is_the_grounded_integral():
    mesh, L, W = _strip()
    sub = SubelementGrounding(mesh)
    V = VectorFunctionSpace(mesh, "CG", 1)
    T = VectorFunctionSpace(mesh, "DG", 0)
    x = fd.SpatialCoordinate(mesh)
    a, u0 = 2.0e-3, 50.0
    u = Function(V).interpolate(fd.as_vector((u0 + a * x[0], Constant(0.0))))
    C = 0.01
    theta = Function(FunctionSpace(mesh, "CG", 1))
    tau = fd.TrialFunction(T)
    sig = fd.TestFunction(T)
    t_sol = Function(T)
    for s in (17.3e3, 40.0e3, 61.9e3):          # inside cells and on an edge
        sub.update(Constant(s) - x[0])
        # a linear law (m = 1) and u_min -> 0: tau_b u/|u| = C u exactly
        F = _subelement_friction(sub, tau, sig, u, Constant(C), theta, 1.0,
                                 law="weertman", N_ref=None, nhat_floor=0.0,
                                 u_min=1e-12, c_w0_floor=0.0, u_lim=0.0, k_lim=0.0)
        fd.solve(fd.lhs(F) == fd.rhs(F), t_sol)
        force = fd.assemble(t_sol[0] * fd.dx)
        exact = -C * W * (u0 * s + a * s * s / 2)
        assert np.isclose(force, exact, rtol=1e-10), (s, force, exact)


def test_a_cell_with_no_ice_has_no_grounded_part():
    mesh, L, W = _strip()
    sub = SubelementGrounding(mesh)
    Q0 = FunctionSpace(mesh, "DG", 0)
    x = fd.SpatialCoordinate(mesh)
    ice = Function(Q0).interpolate(fd.conditional(x[0] < 50e3, 1.0, 0.0))
    sub.update(Constant(80e3) - x[0], ice=ice)
    none = ice.dat.data_ro < 0.5
    assert np.all(sub.fraction.dat.data_ro[none] == 0.0)
    assert np.all(sub.fraction.dat.data_ro[~none] == 1.0)


def main():
    tests = [
        ("the grounded points integrate quadratics exactly",
         test_the_grounded_points_integrate_quadratics_exactly),
        ("a swept grounding line grows the grounded area linearly",
         test_a_swept_grounding_line_grows_the_grounded_area_linearly),
        ("the basal force is the grounded integral", test_the_basal_force_is_the_grounded_integral),
        ("a cell with no ice has no grounded part", test_a_cell_with_no_ice_has_no_grounded_part),
    ]
    for i, (label, fn) in enumerate(tests, 1):
        print(f"{i}. {label}")
        fn()
        print("   ok")
    print("\nPASS: sub-element grounded friction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
