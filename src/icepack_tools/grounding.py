r"""Grounding-line quantities for the icepack2 dual form.

Everything here is built from the *model* thickness and surface, so the
grounded/floating transition is the hydrostatic flotation criterion itself
rather than a mask carried alongside the state.  That is what lets the
effective-pressure-capped friction laws in :mod:`icepack_tools.friction`
(``budd`` and ``regularized_coulomb``) give *exactly* zero basal drag on
floating ice.

Ported from ``ismip7/icepack2_tools/grounding.py``.
"""

import numpy as np

import firedrake as fd
from firedrake import Constant, max_value

from .constants import ice_density, water_density, gravity

#: Default grounding-zone width, in metres of height above flotation.
#: Wider gives more sub-element smoothing of the grounded/floating
#: transition that gates the friction control.
GL_WIDTH = 10.0


def height_above_flotation(H, b, rho_I=ice_density, rho_W=water_density):
    r"""Height above flotation: positive grounded, negative floating.

    .. math::  \mathrm{haf} = H - \frac{\rho_W}{\rho_I}\max(-b, 0)
    """
    h_f = Constant(rho_W / rho_I) * max_value(-b, Constant(0.0))
    return H - h_f


def smooth_heaviside(haf, kH=1.0):
    r"""Smooth grounded indicator, :math:`\tfrac12 + \tfrac12\tanh(k_H\,\mathrm{haf})`.

    ``1/kH`` is the vertical transition width in metres.
    """
    return Constant(0.5) + Constant(0.5) * fd.tanh(Constant(kH) * haf)


def grounded_mask(H, b, gl_width=GL_WIDTH, rho_I=ice_density,
                  rho_W=water_density):
    r"""Smooth grounded indicator ``He`` in [0, 1] (1 grounded, 0 floating).

    Gating the friction control by ``He`` makes the adjoint gradient
    ``dJ/dtheta`` vanish on floating ice, so an optimiser physically cannot
    place basal friction on a shelf.

    ``rho_I`` and ``rho_W`` default to icepack2's constants (917 and 1024
    kg/m^3 in MPa-m-yr units); pass the same values the caller uses for its
    own flotation surface, or the indicator and the surface disagree about
    where flotation is (MISMIP+ and CalvingMIP prescribe 1028).
    """
    haf = height_above_flotation(H, b, rho_I=rho_I, rho_W=rho_W)
    return smooth_heaviside(haf, kH=1.0 / gl_width)


def effective_pressure(H, s, rho_I=ice_density, rho_W=water_density,
                       g=gravity, h_floor=1.0):
    r"""Ocean-connected effective pressure :math:`N = \max(p_I - p_W, 0)`.

    .. math::
        p_I = \rho_I g H, \qquad p_W = \rho_W g \max(0,\, H - s)

    Built from the **model** surface, so ``N`` vanishes exactly at
    flotation.  That exactness is the point: a Coulomb cap
    :math:`\tau_c = c_0 N` then gives bit-exact zero drag on shelves, with
    no ``phi_eff`` floor and so no residual shelf drag.

    ``h_floor`` keeps ``p_I`` finite where the observed thickness is zero.
    """
    Hs = max_value(H, Constant(h_floor))
    p_I = rho_I * g * Hs
    p_W = rho_W * g * max_value(Constant(0.0), H - s)
    return max_value(p_I - p_W, Constant(0.0))


def flotation_surface(H, b, rho_I=ice_density, rho_W=water_density):
    r"""Hydrostatic surface: :math:`\max(b + H,\ (1 - \rho_I/\rho_W) H)`."""
    return max_value(b + H, Constant(1.0 - rho_I / rho_W) * H)


def blended_surface(H, b, He, rho_I=ice_density, rho_W=water_density):
    r"""Surface blended between grounded and floating by ``He``."""
    rr = Constant(rho_I / rho_W)
    return He * (b + H) + (Constant(1.0) - He) * (Constant(1.0) - rr) * H


# ── Sub-element grounded quadrature (ISSM's SEP2) ───────────────────────
#: Barycentric points of the degree-2 (three-point) triangle rule, the
#: order ISSM's ``SubelementFriction2`` asks of ``GaussTria``.
_BARY2 = np.array([[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                   [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                   [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]])


def grounded_quadrature(phi, xy):
    r"""Degree-2 Gauss points on the grounded part of each triangle.

    ``phi`` (m, 3) holds height above flotation at each cell's vertices and
    ``xy`` (m, 3, 2) their coordinates.  ``phi`` is taken as linear on the
    cell, so the grounding line inside it is a straight segment and the
    grounded part (``phi > 0``) is a triangle (one grounded vertex) or a
    quadrilateral (two), which is split into two triangles.  This is the
    construction of Seroussi et al. (2014, SEP2) and ISSM's
    ``Tria::GetGroundedPart`` / ``GaussTria(point1, f1, f2, mainlyfloating,
    2)``.

    Returns ``(points, weights, centroid, full)``:

    ``points`` (m, 6, 2)
        Quadrature points, three per sub-triangle.  Unused slots hold the
        cell centroid with weight 0.
    ``weights`` (m, 6)
        Area weights as fractions of the cell area; they sum to the
        grounded fraction.  Zero on fully grounded and on floating cells,
        which the caller integrates with its ordinary rule or not at all.
    ``centroid`` (m, 2)
        Centroid of the grounded part on a partly grounded cell, of the
        whole cell otherwise.
    ``full`` (m,) bool
        Every vertex grounded.
    """
    phi = np.asarray(phi, dtype=float)
    xy = np.asarray(xy, dtype=float)
    m = phi.shape[0]
    cell_c = xy.mean(axis=1)
    area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
    pos = phi > 0.0
    npos = pos.sum(axis=1)
    points = np.repeat(cell_c[:, None, :], 6, axis=1)
    weights = np.zeros((m, 6))
    centroid = cell_c.copy()
    full = npos == 3

    def tri_area(a, b, c):
        return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                            - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))

    def put(sel, slot0, a, b, c):
        verts = np.stack([a, b, c], axis=1)                  # (k, 3, 2)
        w = tri_area(a, b, c) / area[sel] / 3.0
        for q in range(3):
            points[sel, slot0 + q] = np.einsum("j,kjd->kd", _BARY2[q], verts)
            weights[sel, slot0 + q] = w
        return verts.mean(axis=1), 3.0 * w                  # centroid, area fraction

    rows = np.arange(m)
    # one grounded vertex i: the triangle (P_i, Q_ij, Q_ik)
    s1 = npos == 1
    if s1.any():
        r = rows[s1]
        i = np.argmax(pos[s1], axis=1)
        j, k = (i + 1) % 3, (i + 2) % 3
        pi, pj, pk = phi[r, i], phi[r, j], phi[r, k]
        Xi, Xj, Xk = xy[r, i], xy[r, j], xy[r, k]
        Qj = Xi + (pi / (pi - pj))[:, None] * (Xj - Xi)
        Qk = Xi + (pi / (pi - pk))[:, None] * (Xk - Xi)
        centroid[s1], _ = put(s1, 0, Xi, Qj, Qk)
    # two grounded vertices i, j (k floating): the quadrilateral
    # (P_i, P_j, Q_jk, Q_ik) as the triangles (P_i, P_j, Q_jk), (P_i, Q_jk, Q_ik)
    s2 = npos == 2
    if s2.any():
        r = rows[s2]
        k = np.argmin(pos[s2], axis=1)
        i, j = (k + 1) % 3, (k + 2) % 3
        pi, pj, pk = phi[r, i], phi[r, j], phi[r, k]
        Xi, Xj, Xk = xy[r, i], xy[r, j], xy[r, k]
        Qi = Xk + (pk / (pk - pi))[:, None] * (Xi - Xk)
        Qj = Xk + (pk / (pk - pj))[:, None] * (Xj - Xk)
        c1, f1 = put(s2, 0, Xi, Xj, Qj)
        c2, f2 = put(s2, 3, Xi, Qj, Qi)
        centroid[s2] = (c1 * f1[:, None] + c2 * f2[:, None]) / (f1 + f2)[:, None]
    return points, weights, centroid, full


class SubelementGrounding:
    r"""The grounded part of each cell as quadrature data on DG0 fields.

    ISSM's ``SubelementFriction2`` (Seroussi et al., 2014, SEP2; used by the
    ISSM runs of initMIP-Antarctica, ISMIP6 and Seroussi and Morlighem, 2018)
    integrates the unchanged basal friction over the grounded part of each
    triangle only, with a degree-2 Gauss rule on that part.  A cell-constant
    flotation test switches a whole cell's drag at once, so the grounding
    line migrates cell by cell (``ismip7/GEOMETRY_DISCRETIZATION.md``, "the
    grounding line is a staircase"); this moves it continuously.

    The flotation field is the caller's, given at the vertices, because the
    right vertex values are a model's own choice: an analytic or projected
    vertex bed is better than a reconstruction of the cell-mean bed, and
    the thickness should be averaged over ice cells only.  Every field here
    is a DG0 datum and a cell integrand evaluates a CG1 or DG0 field ``f``
    at a point ``X`` exactly as ``f + grad(f) . (X - x)``, since ``f`` is
    linear on the cell.  No custom quadrature is needed: the sum over the
    points is constant on the cell.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        A triangle mesh.
    """

    NPOINTS = 6

    def __init__(self, mesh):
        self.mesh = mesh
        self.Q0 = fd.FunctionSpace(mesh, "DG", 0)
        W0 = fd.VectorFunctionSpace(mesh, "DG", 0)
        vert = fd.FiniteElement("DG", mesh.ufl_cell(), 1, variant="equispaced")
        self._D1 = fd.FunctionSpace(mesh, vert)
        X1 = fd.VectorFunctionSpace(mesh, vert)
        # Vertex-based DG1: cell-blocked dof arrays are the cell vertices in
        # the DG0 cell order (the construction LevelSet uses).
        xdg = fd.Function(X1).interpolate(fd.SpatialCoordinate(mesh))
        self._xy = xdg.dat.data_ro_with_halos.reshape(-1, 3, 2).copy()
        self.points = [fd.Function(W0, name=f"grounded_point_{q}") for q in range(self.NPOINTS)]
        self.weights = [fd.Function(self.Q0, name=f"grounded_weight_{q}")
                        for q in range(self.NPOINTS)]
        self.centroid = fd.Function(W0, name="grounded_centroid")
        self.full = fd.Function(self.Q0, name="fully_grounded")
        self.fraction = fd.Function(self.Q0, name="grounded_fraction")
        self._x = fd.SpatialCoordinate(mesh)
        self._phi = fd.Function(self._D1)
        self.centroid.dat.data_with_halos[:] = self._xy.mean(axis=1)
        for X in self.points:
            X.assign(self.centroid)

    def update(self, haf_vertex, ice=None):
        r"""Recompute the quadrature from height above flotation at the
        vertices.

        ``haf_vertex`` is a CG1 Function or a UFL expression that is linear
        on each cell (it is interpolated into vertex-based DG1).  ``ice``,
        optional, is a DG0 Function: cells where it is 0 get no grounded
        part at all, whatever their vertices say -- a cell beyond a calving
        front carries no basal drag.
        """
        self._phi.interpolate(haf_vertex)
        phi = self._phi.dat.data_ro_with_halos.reshape(-1, 3)
        pts, w, cen, full = grounded_quadrature(phi, self._xy)
        if ice is not None:
            keep = ice.dat.data_ro_with_halos > 0.5
            w[~keep] = 0.0
            full = full & keep
        for q in range(self.NPOINTS):
            self.points[q].dat.data_with_halos[:] = pts[:, q]
            self.weights[q].dat.data_with_halos[:] = w[:, q]
        self.centroid.dat.data_with_halos[:] = cen
        self.full.dat.data_with_halos[:] = full
        self.fraction.dat.data_with_halos[:] = w.sum(axis=1) + full

    def at(self, f, q):
        r"""``f`` (CG1, DG0 or constant, scalar or vector) at point ``q``."""
        return f + fd.dot(fd.grad(f), self.points[q] - self._x)

    def at_centroid(self, f):
        r"""``f`` at the centroid of each cell's grounded part."""
        return f + fd.dot(fd.grad(f), self.centroid - self._x)

    def integrand(self, g):
        r"""``sum_q w_q g(lambda f: f at point q)``: the grounded-part
        integral of ``g`` on partly grounded cells, divided by the cell
        area.  ``g`` builds its expression from the evaluator it is given;
        integrate the result over ``dx`` (it is constant on each cell)."""
        total = None
        for q in range(self.NPOINTS):
            term = self.weights[q] * g(lambda f, q=q: self.at(f, q))
            total = term if total is None else total + term
        return total
