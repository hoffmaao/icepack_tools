r"""Level-set calving front for a DG0 thickness transport on a buffered mesh.

Cell-centred finite-volume level set after Hahn, Mikula & Frolkovic (2025,
arXiv:2504.05845, "Eikonal boundary condition for level set method"), with
the calving-front kinematics of Bondzio et al. (2016, The Cryosphere 10,
497-510, doi:10.5194/tc-10-497-2016).  The eikonal boundary condition is
what distinguishes this from a level set that needs Dirichlet data on the
mesh edge, and it is why the front can live on a buffered mesh:

* ``phi`` lives in DG0 on the SAME cells as the transport thickness: a
  signed distance to the ice front, **negative in ice, positive in open
  water**.  A cell is ice-free for the transport exactly when ``phi > 0``,
  so removal needs no interpolation between spaces.

Where the level set gets its boundary condition -- :data:`ANCHORS`
------------------------------------------------------------------
``anchor="advect"`` (the default) carries ``phi`` forward in time and
needs a condition on the mesh boundary, which the eikonal condition
below supplies.  ``anchor="extent"`` instead re-solves the eikonal
problem every step with its boundary condition INSIDE the mesh, at the
ice-sheet extent:

.. code::

    |grad phi| = 1,   phi = 0 on the facets between ice and ice-free cells

so ``phi`` is by construction the signed distance to the front the mass
conservation actually sees (``h > h_min``), and the mesh boundary needs
no condition at all.  Advance is then the transport's job -- the upwind
DG0 scheme fills any cell the ice flows into and the next step's extent
includes it -- and the level set only ever moves the front INWARD, by
the normal-flow retreat below.  Rebuilding ``phi`` from the extent
forgets retreat smaller than a cell, so an extent-anchored front reports
its removal in two parts through :meth:`LevelSet.calving_masks`: whole
cells the front has passed, and the fraction ``min(1, c dt L / A)`` of a
front cell's thickness (``L`` its front length, ``A`` its area), which is
exactly the mass ``c h L dt`` a front retreating at ``c`` sheds.  Use
``extent`` where the thickness is the authority on where ice is (an ice
sheet with melt and a buffered ocean); use ``advect`` where the front
must be tracked at sub-cell accuracy through a prescribed motion (an
idealised MIP, a forced retreat scenario).
* The front moves by the advective + normal-flow level-set equation
  ``phi_t + u . grad(phi) - c |grad(phi)| = 0`` (ice velocity ``u``,
  frontal ablation rate ``c`` as a normal shrinking speed).  The
  normal-flow term is linearised semi-implicitly with the previous step's
  unit gradient, ``c |grad phi| ~ c n^k . grad phi^{k+1}``,
  ``n^k = grad phi^k / |grad phi^k|`` (the treatment of Hahn et al.), so
  each step is one linear advection with velocity ``w = u_ext - c n``.
  That velocity is the one of Bondzio et al. (2016, Eqs. 4-7), whose
  boundary moves at ``w . n = u . n - c``.
* Discretisation: finite volumes on the DG0 cells, cell gradients by a
  least-squares fit over the centroid differences to the face neighbours
  (exact for linear fields on any cell with two non-collinear neighbours,
  boundary cells included, so no boundary face is ever needed: the Soner
  exclusion of the paper is automatic), linear-upwind face values (upwind
  cell value plus its gradient times the offset to the face midpoint,
  exact for linear fields; plain cell-centred upwind is inconsistent on
  diagonal faces), inflow cell values implicit and the gradient
  corrections explicit (the paper's inflow-implicit/outflow-explicit
  split), implicit Euler.
* **Eikonal boundary condition** (the paper's contribution): the
  transport equation needs a value of ``phi`` on every inflow boundary
  face, which nobody knows on a buffered mesh.  The eikonal equation
  supplies it: the face value is the cell value extrapolated along the
  previous step's unit gradient, ``phi_f = phi_p + n^k . (x_f - x_p)``,
  i.e. ``|grad phi| = 1`` across the boundary.  (The paper replaces the
  whole cell equation on boundary cells; on a coastal mesh where a third
  of the cells touch the boundary that couples pairs of boundary cells
  through nothing but their difference and the matrix goes singular, so
  here the boundary cells keep their time-derivative term and the
  condition enters through the face.)
* Reinitialisation first resets the interface cells (those sharing a face
  with an opposite-sign cell) by the sub-cell fix
  ``phi / max(|grad phi|, 1)`` (Russo and Smereka, 2000), so the front
  does not move and the anchors carry no distortion (an isolated cell
  instead takes its distance from the zero contour of the P1
  interpolant; see ``_mark_interface_cells``), then relaxes all other cells toward the
  upwind eikonal equation ``sum_q c_q (phi_p - phi_q) = sign(phi_p)``,
  ``c_q >= 0`` over the neighbours strictly closer to the interface (the
  causality of fast marching), in pseudo-time: an M-matrix system
  whatever the reconstructed gradient does at a sliver cell; a few sweeps
  with the stencil and the cosine weights updated in between.  The
  initial condition is the exact geometric signed distance to the
  ice/water facets.
* The ice velocity is needed beyond the front: a harmonic extension with
  the ice-node values as Dirichlet data.  The ablation rate is each ice
  cell's own mean of the law, read on the ice cells inside the front and
  nowhere else, and is carried into the water constant along the normal,
  ``n . grad c = 0`` (Bondzio et al., 2016, Eq. 9, after Zhao et al.,
  1996), by the reinitialisation's causal upwind stencil.  A front face
  moves with its ice cell's rate, never the extension's.

Both anchors share every piece of the discretisation below: the
least-squares cell gradients, the linear-upwind faces, the linearised
eikonal operator and its upwind (fast-marching-causal) stencil.

How the front moves (:data:`LAWS`):

``none``        ``c = 0``; the front is free to advance and never calves.
``fixed``       the level set is frozen at its t = 0 position: a
                no-advance barrier expressed through ``phi``.
``prescribed``  ``c`` is whatever the caller passes to :meth:`LevelSet.advance`
                as ``rate`` -- a Constant, a Function or a UFL expression
                on ice, in m/yr of normal face retreat.  The front then
                retreats exactly where ``c`` exceeds the ice speed into
                it, whatever the thickness of the cell it happens to be
                in -- which a thickness sink applied to front cells cannot
                do, since a thin front cell fed from a thick one refills
                within a step and the sink never reaches the thick cell
                behind.

The level set holds no calving physics.  A calving law (von Mises,
horizontal force balance, minimum thickness, ...) is a rate, defined in
:mod:`icepack_tools.calving` and evaluated by the caller on the state it
reads, front normal included: ``ghat`` is refreshed with every eikonal
solve, at the start of every advected step and at construction, so a
law reads the normal of the current front at every step, the first
included.  An imposed rate (submarine
melt undercutting a grounded tidewater front, a frontal ablation
scenario) goes in the same way.

The momentum balance needs no front term: with DG0 geometry the facet
term ``rho_I g avg(h) jump(s)`` at an ice/no-ice face IS the terminus
water-pressure force (see icepack2 ``momentum_balance``), and with a CG1
geometry on a buffered mesh the dual form resolves the front through the
zero-thickness limit.  What the level set adds on the momentum side is
only the optional ``drag_mask``: a consumer that keeps a floor-cell ocean
drag in its empty buffer can switch it OFF in the strip of water cells
adjacent to the front, so front nodes are not slowed by the drag of the
water cells they touch.

Ported from ``ismip7/icepack2_tools/levelset.py`` with the ``prescribed``
law added and two fixes to the reinitialisation's interface reset.  It
used to touch the level-set data only on ranks that owned an interface
cell, and the halo exchange that access implies is collective, so a
rank with none of the front deadlocked the next assembly.  And it kept
each cell's own sign, which made an isolated opposite-sign cell
permanent (see ``LevelSet._mark_interface_cells``).  The tests in
``test/levelset_test.py`` are that module's tests plus one for the new
law and one for the isolated cells, and they run again on three ranks so
a front owned by one rank is covered.

Units: metres, years, MPa (icepack conventions).
"""

from time import perf_counter

import numpy as np
from mpi4py import MPI as _MPI
from scipy.spatial import cKDTree

import firedrake as fd
from finat.point_set import PointSet
from finat.quadrature import QuadratureRule
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace, TestFunction,
    TrialFunction, SpatialCoordinate, CellDiameter, FacetNormal, assemble,
    dx, dS, ds, inner, grad, dot, sqrt, conditional, gt, max_value,
    min_value, avg, outer, FacetArea, TensorFunctionSpace,
)
from firedrake.petsc import PETSc

#: How :class:`LevelSet` can move a front (the calving laws themselves are
#: rates, in :mod:`icepack_tools.calving`).
LAWS = ("none", "fixed", "prescribed")

#: Where the level set takes its boundary condition (see the module
#: docstring): ``advect`` carries ``phi`` and conditions the mesh
#: boundary, ``extent`` re-solves the eikonal problem at the ice extent.
ANCHORS = ("advect", "extent")

# Regularisation of |grad phi|: phi is in metres and the unit gradient only
# ever divides by |grad phi| ~ 1, so any small delta works.
DELTA = 1e-6
# Pseudo-time of a reinitialisation sweep in cell diameters: each sweep may
# move a value by up to this many cells toward the distance function.
REINIT_TAU_CELLS = 4.0
WSUM_FLOOR = 0.05
# weight of the tie to zero in the normal extension of the rate: only a row
# with no upwind neighbour feels it
EXT_TIE = 1e-4
# relative tolerance of the velocity's harmonic extension into the water
EXT_RTOL = 1e-6
# The level set's advection and reinitialisation sweeps are upwind M-matrices
# on DG0 cells with a (pseudo-)time mass term, diagonally dominant: Krylov
# with block ILU.
LS_SOLVER = {"ksp_type": "gmres", "pc_type": "bjacobi", "sub_pc_type": "ilu",
             "ksp_rtol": 1e-10, "ksp_atol": 1e-12, "ksp_max_it": 500}
# The rate's normal extension has no mass term: a water cell whose upwind
# neighbours lie nearly across its gradient is held by little more than the
# tie, and GMRES with block ILU diverged on such a row 68 years into
# CalvingMIP experiment 4.  It is factorised.
LS_DIRECT = {"ksp_type": "preonly", "pc_type": "lu",
             "pc_factor_mat_solver_type": "mumps"}
# Relative singular-value cutoff of the least-squares gradient stencil.
LSQ_RCOND = 1e-3
# phi is a distance function, |grad phi| <= 1; the reconstruction gradient is
# limited to this magnitude so a bad stencil cannot poison the face values.
GRAD_LIMIT = 1.5
# Reinitialisation is judged within this many cell diameters of the front.
REINIT_BAND_CELLS = 10.0


def _global_count(mask, comm):
    r"""Number of True entries across ranks."""
    return int(comm.allreduce(int(np.count_nonzero(np.asarray(mask)))))


def _coord_keys(xy, scale=1e-3):
    r"""Hashable rounded-coordinate keys (mm) for vertex matching on an
    unstructured mesh without touching DMPlex numbering."""
    q = np.round(np.asarray(xy) / scale).astype(np.int64)
    return [tuple(r) for r in q]


def _segment_distance(points, seg_a, seg_b, k=8):
    r"""Distance from each of ``points`` (N,2) to the nearest of the segments
    ``seg_a[i]``-``seg_b[i]``.  A KD-tree over per-segment sample points picks
    ``k`` candidate segments per point, then the exact point-to-segment
    distance is taken over those candidates."""
    if len(seg_a) == 0:
        return np.full(len(points), np.inf)
    seg_a = np.asarray(seg_a, dtype=float)
    seg_b = np.asarray(seg_b, dtype=float)
    ns = len(seg_a)
    ts = np.linspace(0.0, 1.0, 5)
    samples = (seg_a[:, None, :] + ts[None, :, None]
               * (seg_b - seg_a)[:, None, :]).reshape(-1, 2)
    owner = np.repeat(np.arange(ns), len(ts))
    tree = cKDTree(samples)
    kq = min(k, len(samples))
    _, idx = tree.query(points, k=kq)
    idx = np.atleast_2d(idx)
    if idx.shape[0] != len(points):
        idx = idx.T
    cand = owner[idx]                                   # (N, kq)
    a = seg_a[cand]                                     # (N, kq, 2)
    d = seg_b[cand] - a
    ap = points[:, None, :] - a
    dd = np.maximum((d * d).sum(-1), 1e-30)
    t = np.clip((ap * d).sum(-1) / dd, 0.0, 1.0)
    proj = a + t[..., None] * d
    dist = np.sqrt(((points[:, None, :] - proj) ** 2).sum(-1))
    return dist.min(axis=1)


class LevelSet:
    r"""Finite-volume signed-distance calving front on the DG0 cells of ``mesh``.

    Parameters
    ----------
    mesh : firedrake.Mesh
    h_dg : Function (DG0)
        The transport thickness; read for the initial front and for the
        ice/no-ice split of the velocity extrapolation.
    law : str
        One of :data:`LAWS`: how the front moves.  A calving law runs as
        ``prescribed``, with its rate passed to :meth:`advance`.
    h_min : float
        Cells with ``h <= h_min`` [m] count as ice-free.
    reinit_every : int
        Reinitialise every this many ``advance`` calls (0 = never).
    reinit_sweeps : int
        Fixed-point sweeps of the linearised eikonal equation per
        reinitialisation.
    phi_init : Function (DG0) or None
        Restart the level set from a checkpointed field instead of the
        thickness outline.
    drag_mask : Function (DG0) or None
        The gate Function a momentum residual with floor-cell drag was
        built with; left alone by consumers without such a drag.
    """

    def __init__(self, mesh, h_dg, law="none", h_min=1.0, reinit_every=5,
                 reinit_sweeps=4, phi_init=None, drag_mask=None,
                 anchor="advect"):
        if law not in LAWS:
            raise ValueError(f"front law {law!r} not in {LAWS}")
        if anchor not in ANCHORS:
            raise ValueError(f"anchor {anchor!r} not in {ANCHORS}")
        self.mesh = mesh
        self.comm = mesh.comm
        self.law = law
        self.anchor = anchor
        self.dt = 0.0
        self.h_min = float(h_min)
        self.reinit_every = int(reinit_every)
        self.reinit_sweeps = max(1, int(reinit_sweeps))
        self.n_advance = 0

        self.Q0 = FunctionSpace(mesh, "DG", 0)
        self.W0 = VectorFunctionSpace(mesh, "DG", 0)
        self.Q1 = FunctionSpace(mesh, "CG", 1)
        self.V1 = VectorFunctionSpace(mesh, "CG", 1)
        # Vertex-based DG1 (the default variant puts the dofs at interior
        # points in this Firedrake), so cell-blocked dof arrays ARE the cell
        # vertices, in the same cell order as the DG0 dof array.
        dg1_vertices = fd.FiniteElement("DG", mesh.ufl_cell(), 1,
                                        variant="equispaced")
        self.X1 = VectorFunctionSpace(mesh, dg1_vertices)
        self.D1 = FunctionSpace(mesh, dg1_vertices)
        self.h_dg = h_dg

        xdg = Function(self.X1).interpolate(SpatialCoordinate(mesh))
        self.cell_xy = xdg.dat.data_ro_with_halos.reshape(-1, 3, 2).copy()
        self.cell_xc = Function(self.W0).interpolate(
            SpatialCoordinate(mesh)).dat.data_ro.copy()
        self.cell_area = assemble(TestFunction(self.Q0) * dx).dat.data_ro.copy()
        self.cell_diam = Function(self.Q0).interpolate(CellDiameter(mesh))
        # Edge -> cells map (halos included), the extent facets come from it.
        keys = [_coord_keys(c) for c in self.cell_xy]
        self._edge_cells = {}
        for ci, kk in enumerate(keys):
            for e in ((0, 1), (1, 2), (2, 0)):
                ek = tuple(sorted((kk[e[0]], kk[e[1]])))
                self._edge_cells.setdefault(ek, []).append(ci)
        # Boundary cells: the eikonal equation replaces transport there.
        bnd = assemble(TestFunction(self.Q0) * ds)
        self.chi_bnd = Function(self.Q0, name="boundary_cell")
        self.chi_bnd.dat.data[:] = np.where(bnd.dat.data_ro > 0.0, 1.0, 0.0)

        self.phi = Function(self.Q0, name="levelset")
        self.phi_old = Function(self.Q0)
        self.ghat = Function(self.W0, name="unit_gradient")
        self.gvec = Function(self.W0, name="cell_gradient")
        self.wsum = Function(self.Q0, name="upwind_weight_sum")
        self.chi_fb = Function(self.Q0, name="centred_fallback")
        self.xc_fn = Function(self.W0, name="centroid").interpolate(
            SpatialCoordinate(mesh))
        self.area_fn = Function(self.Q0, name="cell_area")
        self.area_fn.dat.data[:] = self.cell_area
        self.sgn = Function(self.Q0, name="sign")
        self.chi_fix = Function(self.Q0, name="interface_cell")
        self.c_cell = Function(self.Q0, name="ablation_rate_cell")
        # the cells whose rate the front reads (ice inside the front),
        # refreshed by every advect step
        self.chi_rate = Function(self.Q0, name="ice_cell_inside")
        # Extent anchoring: the ice indicator and the per-cell front length
        # (facets to ice-free neighbours), both refreshed by every
        # :meth:`solve_eikonal_from_extent`.
        self.chi = Function(self.Q0, name="ice_cell")
        self.front_len = Function(self.Q0, name="front_length")
        # A momentum residual may hold a reference to this Function, so the
        # caller passes the one it built the residual with.
        self.drag_mask = (drag_mask if drag_mask is not None
                          else Function(self.Q0, name="drag_mask"))
        self.drag_mask.assign(1.0)
        # Extended velocity beyond the front, CG1.
        self.w_ext = Function(VectorFunctionSpace(mesh, "CG", 1),
                              name="front_extension")

        if phi_init is not None:
            self.phi.assign(phi_init)
            self._refresh_extent_fields()
        elif anchor == "extent":
            self.solve_eikonal_from_extent()
        else:
            self.initialise_from_thickness()
        self._build_solvers()
        # The t=0 extent, which the ``fixed`` law holds the front at.
        self.phi0 = self.phi.copy(deepcopy=True)
        self.update_cell_fields()
        PETSc.Sys.Print(
            f"  Level-set front (FV, eikonal BC, anchor={anchor}): law={law}, "
            f"h_min={h_min:g} m"
            + (f", reinit every {self.reinit_every} steps x "
               f"{self.reinit_sweeps} sweeps" if anchor == "advect" else "")
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _ice_cells(self):
        return self.h_dg.dat.data_ro_with_halos > self.h_min

    def _gather_segments(self, seg_a, seg_b):
        a = np.asarray(seg_a, dtype=float).reshape(-1, 2)
        b = np.asarray(seg_b, dtype=float).reshape(-1, 2)
        parts_a = self.comm.allgather(a)
        parts_b = self.comm.allgather(b)
        return np.concatenate(parts_a), np.concatenate(parts_b)

    def initialise_from_thickness(self):
        r"""``phi`` = exact signed distance from each cell centroid to the
        facets between ice and ice-free cells, negative on ice cells."""
        ice = self._ice_cells()
        seg_a, seg_b = [], []
        for ek, cells in self._edge_cells.items():
            if len(cells) == 2 and ice[cells[0]] != ice[cells[1]]:
                seg_a.append(np.array(ek[0]) * 1e-3)
                seg_b.append(np.array(ek[1]) * 1e-3)
        seg_a, seg_b = self._gather_segments(seg_a, seg_b)
        n_own = len(self.phi.dat.data_ro)
        dist = _segment_distance(self.cell_xc, seg_a, seg_b)
        if not np.isfinite(dist).any():
            dist = np.full_like(dist, 1e7)
        sign = np.where(ice[:n_own], -1.0, 1.0)
        self.phi.dat.data[:] = sign * np.where(np.isfinite(dist), dist, 1e7)
        n_ice = _global_count(ice[:n_own], self.comm)
        PETSc.Sys.Print(
            f"  Level-set init: {len(seg_a)} front segments, {n_ice} ice cells"
        )

    def _refresh_extent_fields(self):
        r"""Ice indicator and per-cell front length from the current
        thickness.  Collective: every rank assembles."""
        ice = self._ice_cells()
        n_own = len(self.phi.dat.data_ro)
        self.chi.dat.data[:] = np.where(ice[:n_own], 1.0, 0.0)
        psi = TestFunction(self.Q0)
        c = self.chi
        L = assemble((conditional(gt(c('+'), c('-')), 1.0, 0.0) * psi('+')
                      + conditional(gt(c('-'), c('+')), 1.0, 0.0) * psi('-')) * dS)
        self.front_len.dat.data[:] = L.dat.data_ro

    def solve_eikonal_from_extent(self):
        r"""``phi`` := the solution of ``|grad phi| = 1`` with ``phi = 0`` on
        the ice/water facets of the current extent -- the exact signed
        distance to them, negative on ice.  This is the eikonal problem
        whose boundary condition is the ice-sheet extent inside the mesh
        (``anchor="extent"``); the mesh boundary plays no part.  Returns the
        number of front facets."""
        ice = self._ice_cells()
        seg_a, seg_b = [], []
        for ek, cells in self._edge_cells.items():
            if len(cells) == 2 and ice[cells[0]] != ice[cells[1]]:
                seg_a.append(np.array(ek[0]) * 1e-3)
                seg_b.append(np.array(ek[1]) * 1e-3)
        seg_a, seg_b = self._gather_segments(seg_a, seg_b)
        n_own = len(self.phi.dat.data_ro)
        dist = _segment_distance(self.cell_xc, seg_a, seg_b)
        dist = np.where(np.isfinite(dist), dist, 1e7)
        self.phi.dat.data[:] = np.where(ice[:n_own], -1.0, 1.0) * dist
        self._refresh_extent_fields()
        # The front normal a law reads is the normal of THIS extent.  The
        # first call comes from the constructor, before the gradient forms
        # exist; building them evaluates it.
        if hasattr(self, "_grad_form"):
            self._update_unit_gradient()
        return len(seg_a)

    def calving_masks(self):
        r"""``(beyond, fraction)`` for an extent-anchored front, to be applied
        to the thickness after the transport step.

        ``beyond`` flags cells the front has passed entirely (``phi > 0`` on
        a cell that held ice), or -- for the ``fixed`` law -- every cell
        beyond the t=0 extent.  ``fraction`` is the thickness fraction
        ``min(1, c dt L / A)`` each front cell sheds, which carries the
        sub-cell retreat that re-solving from the extent would forget; it is
        ``None`` when no law removes ice."""
        if self.anchor != "extent":
            raise RuntimeError("calving_masks() needs anchor='extent'")
        if self.law == "fixed":
            return self.phi0.dat.data_ro > 0.0, None
        beyond = (self.phi.dat.data_ro > 0.0) & (self.chi.dat.data_ro > 0.5)
        if self.law == "none":
            return beyond, None
        frac = np.clip(self.c_cell.dat.data_ro * self.dt
                       * self.front_len.dat.data_ro / self.cell_area, 0.0, 1.0)
        frac[beyond] = 0.0
        return beyond, frac

    def update_cell_fields(self):
        r"""Drag mask: drag stays on only in cells more than one cell
        diameter beyond the front, so front nodes see no water-cell drag."""
        pc = self.phi.dat.data_ro
        self.drag_mask.dat.data[:] = np.where(
            pc > self.cell_diam.dat.data_ro, 1.0, 0.0)

    def beyond_front(self):
        r"""Boolean per owned cell: outside the ice (remove it)."""
        return self.phi.dat.data_ro > 0.0

    # ------------------------------------------------------------------
    # Finite-volume operators
    # ------------------------------------------------------------------
    def _build_solvers(self):
        mesh = self.mesh
        n = FacetNormal(mesh)
        psi = TestFunction(self.Q0)
        phi = TrialFunction(self.Q0)

        # Least-squares cell gradient: G_p = M_p^-1 sum_q (phi_q - phi_p) d_pq
        # with d_pq = x_q - x_p over the face neighbours q, M_p = sum d d^T.
        # The sums are facet integrals divided by the facet length, so each
        # face counts once with unit weight.
        xc = self.xc_fn
        d_plus = xc('-') - xc('+')
        w_face = 1.0 / FacetArea(mesh)
        vt = TestFunction(TensorFunctionSpace(mesh, "DG", 0))
        M = assemble(w_face * (inner(vt('+'), outer(d_plus, d_plus))
                               + inner(vt('-'), outer(d_plus, d_plus))) * dS)
        Mc = M.dat.data_ro.reshape(-1, 2, 2)
        # A corner cell with a single neighbour has a rank-1 M: its gradient
        # cannot be reconstructed and is copied from that neighbour instead
        # (pinv would give the minimum-norm vector along the neighbour
        # direction, which is wrong whenever the front is not parallel to it).
        det = np.linalg.det(Mc)
        tr = np.trace(Mc, axis1=1, axis2=2)
        self.deficient = det < 1e-8 * tr ** 2
        # Truncated pseudo-inverse: a sliver whose neighbour centroids are
        # nearly collinear has an ill-conditioned M, and the component of the
        # gradient along the poorly-determined direction is dropped rather
        # than amplified (32 km coastal meshes have 10:1 size ratios).
        self.Minv = np.linalg.pinv(Mc, rcond=LSQ_RCOND)
        v = TestFunction(self.W0)
        self._grad_form = w_face * (
            (self.phi('-') - self.phi('+')) * inner(v('+'), d_plus)
            + (self.phi('+') - self.phi('-')) * inner(v('-'), -d_plus)
        ) * dS
        self._grad_cof = fd.Cofunction(self.W0.dual())
        self._nb_sum_form = w_face * (inner(v('+'), self.gvec('-'))
                                      + inner(v('-'), self.gvec('+'))) * dS
        # Upwind face weights for the eikonal operator: the neighbour across
        # the face is upwind for a cell when it lies toward the interface.
        po = self.phi_old
        up_p = conditional(fd.lt(abs(po('-')), abs(po('+'))), 1.0, 0.0)
        up_m = conditional(fd.lt(abs(po('+')), abs(po('-'))), 1.0, 0.0)
        self._up_plus = up_p + self.chi_fb('+') * (1 - up_p)
        self._up_minus = up_m + self.chi_fb('-') * (1 - up_m)
        dlen = sqrt(dot(d_plus, d_plus))
        cos_p = dot(self.ghat('+'), d_plus) / dlen
        cos_m = dot(self.ghat('-'), -d_plus) / dlen
        self._cos_plus, self._cos_minus, self._dlen = cos_p, cos_m, dlen
        self._wsum_form = w_face * (self._up_plus * cos_p ** 2 * psi('+')
                                    + self._up_minus * cos_m ** 2 * psi('-')) * dS
        self._up_count = w_face * (up_p * psi('+') + up_m * psi('-')) * dS

        # (1) Harmonic extension of the velocity from ice nodes into water.
        # Only ``advect`` needs it: an extent-anchored front never carries
        # phi into the water, and its rate lives on ice cells alone.
        W = self.w_ext.function_space()
        self.w_src = Function(W)
        self.ice_node = Function(self.Q1, name="ice_node")
        w = TrialFunction(W)
        vw = TestFunction(W)
        hc = CellDiameter(mesh)
        pen = Constant(1e4) / hc ** 2
        # The penalty is integrated with the vertex rule, so it is lumped:
        # it pins each ice node to its datum and touches no water node.
        # Integrated exactly, ``ice_node * w * v`` is nonzero on every cell
        # with one ice vertex and pulled the first ring of water nodes
        # toward whatever ``w_src`` held there -- which for the rate is
        # nothing, and for the velocity the momentum solve's values in
        # cells with no ice.  The extension must read ice nodes alone.
        vertex_rule = QuadratureRule(
            PointSet([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), [1.0 / 6.0] * 3)
        dxv = dx(scheme=vertex_rule)
        a_ext = (inner(grad(w), grad(vw)) * dx
                 + pen * self.ice_node * inner(w, vw) * dxv)
        L_ext = pen * self.ice_node * inner(self.w_src, vw) * dxv
        self._ext_solver = (fd.LinearVariationalSolver(
            fd.LinearVariationalProblem(a_ext, L_ext, self.w_ext),
            solver_parameters={"ksp_type": "cg", "pc_type": "gamg",
                               "ksp_rtol": EXT_RTOL, "ksp_max_it": 2000})
            if self.anchor == "advect" else None)

        # (2) Level-set step: inflow-implicit upwind advection with
        # w = u_ext - c ghat on interior cells, linearised eikonal on
        # boundary cells.
        self.dt_c = Constant(1.0)
        u_e = self.w_ext
        if self.anchor == "advect":
            # Face normal speed.  A front face (ice inside on one side
            # only) moves with the ice side's rate, as the front does in
            # Bondzio et al. (2016, Eqs. 4-7): the value extended into the
            # water is a harmonic blend of every ice node around, and
            # averaging it in handed a grounded front cell (c = 0 under a
            # gated law) part of a retreating floating neighbour's rate --
            # the cell ahead of it then took ice the front never reached.
            # Faces between two ice cells, or two water cells, average.
            w_n = -self.c_cell * self.ghat                  # DG0 vector
            ci = self.chi_rate
            s_i = ci('+') + ci('-')
            w_face = conditional(
                gt(s_i, 0.5),
                (ci('+') * w_n('+') + ci('-') * w_n('-')) / max_value(s_i, 1.0),
                avg(w_n))
            wn = dot(u_e, n('+')) + dot(w_face, n('+'))
        else:
            # Extent anchoring: normal-flow retreat only, and the face rate
            # is the ice-weighted mean of the two cells' rates, so a front
            # face carries the ice side's rate rather than half of it (the
            # water side has c = 0).
            c_face = ((self.c_cell('+') * self.chi('+')
                       + self.c_cell('-') * self.chi('-'))
                      / max_value(self.chi('+') + self.chi('-'), Constant(1.0)))
            wn = -c_face * dot(avg(self.ghat), n('+'))
        # Linear-upwind face values: phi_f = phi_up + g_up . (x_f - x_up),
        # with x_f the face midpoint (one-point facet rule is exact for the
        # linear integrand). Inflow: the neighbour's reconstruction, cell
        # value implicit; outflow: the cell's own explicit gradient term.
        x = SpatialCoordinate(mesh)
        g = self.gvec
        rec_p = dot(g('+'), x - xc('+'))      # own extrapolation to the face
        rec_m = dot(g('-'), x - xc('-'))
        adv = (
            (min_value(wn, 0.0) * (phi('-') + rec_m - phi('+'))
             + max_value(wn, 0.0) * rec_p) * psi('+')
            + (-max_value(wn, 0.0) * (phi('+') + rec_p - phi('-'))
               - min_value(wn, 0.0) * rec_m) * psi('-')
        ) * dS(degree=1)
        if self.anchor == "advect":
            # Eikonal boundary condition: face value extrapolated along the
            # unit gradient, for inflow (the unknown value) and outflow alike.
            wn_b = dot(u_e, n) - self.c_cell * dot(self.ghat, n)
            eik_b = wn_b * dot(self.ghat, x - xc) * psi * ds(degree=1)
        else:
            # No boundary term: the extent anchoring already applied the
            # eikonal condition inside the mesh, and retreat never reaches
            # the mesh edge.
            eik_b = Constant(0.0) * psi * ds
        a_ls = ((phi - self.phi_old) / self.dt_c * psi * dx + adv + eik_b)
        lu = dict(LS_SOLVER)
        self._ls_solver = fd.LinearVariationalSolver(
            fd.LinearVariationalProblem(fd.lhs(a_ls), fd.rhs(a_ls), self.phi),
            solver_parameters=lu)

        # (3) Reinitialisation: interface cells keep their value, all others
        # relax toward the linearised eikonal equation (rhs 1 on both sides;
        # the sign only steers the stencil) in pseudo-time tau ~ cell size,
        # so the system is always diagonally dominant.
        chi_f = self.chi_fix
        tau = Constant(REINIT_TAU_CELLS) * self.cell_diam
        # phi_t = -sign(phi) (|grad phi| - 1) in M-matrix form: the
        # relaxation is diagonally dominant on both sides of the front.
        eik_r = self._eikonal_terms(phi, psi, 1 - chi_f, n, rhs=self.sgn)
        a_re = (chi_f * (phi - self.phi_old) * psi * dx
                + (1 - chi_f) * (phi - self.phi_old) / tau * psi * dx + eik_r)
        self._reinit_solver = fd.LinearVariationalSolver(
            fd.LinearVariationalProblem(fd.lhs(a_re), fd.rhs(a_re), self.phi),
            solver_parameters=lu)

        # (4) The rate beyond the front, constant along the normal
        # (``n . grad c = 0``, Bondzio et al., 2016, Eq. 9): each water cell
        # takes the upwind-weighted mean of its neighbours nearer the front,
        # through the reinitialisation's own causal stencil, down to the ice
        # cells, which hold their own rate.  A weak tie to zero keeps a row
        # with no upwind neighbour determined.
        if self.anchor == "advect":
            self.c_own = Function(self.Q0, name="ablation_rate_ice")
            ci, d2 = self.chi_rate, self.cell_diam ** 2
            c_t = fd.TrialFunction(self.Q0)
            a_c = (self._eikonal_terms(c_t, psi, 1 - ci, n, rhs=Constant(0.0))
                   + ci * (c_t - self.c_own) * psi / d2 * dx
                   + Constant(EXT_TIE) * (1 - ci) * c_t * psi / d2 * dx)
            self._rate_ext_solver = fd.LinearVariationalSolver(
                fd.LinearVariationalProblem(fd.lhs(a_c), fd.rhs(a_c), self.c_cell),
                solver_parameters=dict(LS_DIRECT))

        # The unit gradient of the initial phi, so a law that reads the front
        # normal before the first advance reads this front's.
        self._update_unit_gradient()

    def _eikonal_terms(self, phi, psi, weight, n, rhs):
        r"""Cell residual of the upwind eikonal equation in M-matrix form,
        ``sum_q c_q (phi_p - phi_q) = rhs`` with ``c_q = |cos_q|/(wsum |d_pq|)
        >= 0`` over the upwind neighbours (``rhs = sign(phi_p)``: distance
        grows away from the interface on both sides), scaled by the cell
        area, applied where ``weight`` (DG0) is 1.  Only the magnitude of
        the gradient direction enters, so a noisy sign at a sliver cannot
        flip a coefficient and break diagonal dominance."""
        w_face = self.area_fn / FacetArea(self.mesh)
        # A cell whose upwind directions are nearly perpendicular to the
        # gradient carries little information; floor the weight sum so the
        # row stays weakly coupled instead of amplified.
        ws = max_value(self.wsum, Constant(WSUM_FLOOR))
        coef_p = abs(self._cos_plus) / (ws('+') * self._dlen)
        coef_m = abs(self._cos_minus) / (ws('-') * self._dlen)
        return (
            (w_face('+') * weight('+') * self._up_plus
             * (phi('+') - phi('-')) * coef_p * psi('+')
             + w_face('-') * weight('-') * self._up_minus
             * (phi('-') - phi('+')) * coef_m * psi('-')) * dS
            - weight * rhs * psi * dx
        )

    def cell_gradient(self):
        r"""Least-squares gradient of the current ``phi`` (owned cells)."""
        assemble(self._grad_form, tensor=self._grad_cof)
        r = self._grad_cof.dat.data_ro
        return np.einsum("cij,cj->ci", self.Minv, r)

    def _update_unit_gradient(self):
        r"""Unit gradient from the centred least-squares fit, then the upwind
        stencil (cells without an upwind neighbour use all neighbours) and
        the cosine-squared weight sums of the eikonal operator."""
        g = self.cell_gradient()
        self.gvec.dat.data[:] = g
        # Unconditional: assemble is collective, and which ranks own a
        # deficient cell differs between them.
        nb = assemble(self._nb_sum_form).dat.data_ro
        g = g.copy()
        g[self.deficient] = nb[self.deficient]
        mag = np.sqrt((g * g).sum(axis=1) + DELTA ** 2)
        self.ghat.dat.data[:] = g / mag[:, None]
        scale = np.minimum(1.0, GRAD_LIMIT / mag)
        self.gvec.dat.data[:] = g * scale[:, None]
        self.sgn.dat.data[:] = np.where(self.phi.dat.data_ro > 0, 1.0, -1.0)
        self.chi_fb.assign(0.0)
        cnt = assemble(self._up_count).dat.data_ro
        self.chi_fb.dat.data[:] = np.where(cnt < 0.5, 1.0, 0.0)
        self.wsum.dat.data[:] = assemble(self._wsum_form).dat.data_ro

    @staticmethod
    def _crossing_segments(v, xy):
        r"""Zero-contour segment of each P1 cell in ``v`` (m,3) with vertex
        coordinates ``xy`` (m,3,2): the two edge crossings, vectorised."""
        if len(v) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            cross = (v[:, i] > 0) != (v[:, j] > 0)
            t = np.where(cross, v[:, i] / np.where(cross, v[:, i] - v[:, j], 1.0), np.nan)
            pts.append(xy[:, i] + t[:, None] * (xy[:, j] - xy[:, i]))
        pts = np.stack(pts, axis=1)                      # (m, 3, 2), NaN where no crossing
        ok = ~np.isnan(pts[..., 0])                      # exactly two True per row
        order = np.argsort(~ok, axis=1, kind="stable")[:, :2]
        rows = np.arange(len(v))[:, None]
        two = pts[rows, order]
        return two[:, 0], two[:, 1]

    def _mark_interface_cells(self):
        r"""Cells sharing a face with an opposite-sign cell are the anchors
        of a reinitialisation.  Each is reset from its own value and its
        own least-squares gradient, ``phi / max(|grad phi|, 1)`` (the
        sub-cell fix of Russo and Smereka, 2000): the zero crossing stays
        where the cell's data puts it, a field steepened by advection is
        brought back to a distance, and a flat one is never stretched.
        Resetting to the distance from the zero contour of the lumped P1
        interpolant instead moved every convex front inward by about
        ``h^2 kappa`` per reinitialisation, since a vertex mean of a convex
        distance field over-reads it: the rounded tip of a grounded tongue
        20 km wide on a 5 km mesh went back 0.6 km at each one, sharpened,
        and lost 9 km in 80 yr while its advection alone was exact.

        An isolated cell -- no face neighbour of its own sign -- takes sign
        and distance from the interpolant instead, the sign read at its
        centroid.  Deciding by the interpolant's sign alone also caught the
        cell the front had just crossed at the tip of an advancing convex
        front, whose small value the vertex means outvote, and threw it
        back beyond the front.  An isolated cell makes no zero
        contour in the interpolant, so keeping its sign while measuring
        the distance to the interpolant's contour wrote back the distance
        to the nearest real front -- a one-cell pocket of water 6 km inside
        the ice read phi = +6.75 km and was permanent (CalvingMIP
        experiment 4 on Thule's ridges, where the rate law drives interior
        floating cells between grounded ones across zero between
        reinitialisations).  Sign and distance have to come from the same
        field or a feature one of them cannot represent survives forever.
        Taking the sign at the centroid of the interpolant is also the
        cell-wise reading of the "any vertex inside is ice" mask that a
        nodal level set uses (Bondzio et al., 2016, Sect. 2.3).  Cells the
        interpolant does not cut keep their sign."""
        sgn = Function(self.Q0)
        sgn.dat.data[:] = np.where(self.phi.dat.data_ro > 0, 1.0, -1.0)
        psi = TestFunction(self.Q0)
        cut = assemble((abs(fd.jump(sgn)) * (psi('+') + psi('-'))) * dS)
        fix = cut.dat.data_ro > 0.0
        # no face neighbour of the same sign: a one-cell pocket or island
        same = assemble(((1.0 - abs(fd.jump(sgn)) / 2.0) * (psi('+') + psi('-'))) * dS)
        isolated = same.dat.data_ro < 0.5
        self.chi_fix.dat.data[:] = np.where(fix, 1.0, 0.0)
        # P1 interpolant by mass lumping, then its zero contour.
        q = TestFunction(self.Q1)
        phi_cg = Function(self.Q1)
        phi_cg.dat.data[:] = (assemble(self.phi * q * dx).dat.data_ro
                              / assemble(q * dx).dat.data_ro)
        pc = Function(self.D1).interpolate(phi_cg).dat.data_ro_with_halos.reshape(-1, 3)
        pos = pc > 0
        cutc = pos.any(axis=1) & (~pos).any(axis=1)
        seg_a, seg_b = self._gather_segments(
            *self._crossing_segments(pc[cutc], self.cell_xy[cutc]))
        # Touch the data on EVERY rank before deciding whether there is
        # anything to reset: reading ``dat.data_ro`` may run a halo
        # exchange and writing ``dat.data`` invalidates one, both of which
        # are collective.  A rank that owns no interface cell and skipped
        # them left the others waiting at the next assembly.
        n_own = len(self.phi.dat.data_ro)
        # the interpolant at the centroid: the mean of its vertex values
        sgn_lift = np.where(pc[:n_own].mean(axis=1) > 0.0, 1.0, -1.0)
        g = self.cell_gradient()
        gmag = np.maximum(np.sqrt((g * g).sum(axis=1)), 1.0)
        phi_data = self.phi.dat.data
        own = phi_data / gmag
        if len(seg_a) and fix.any():
            dist = _segment_distance(self.cell_xc[fix], seg_a, seg_b)
            phi_data[fix] = np.where(isolated[fix], sgn_lift[fix] * dist, own[fix])

    def reinitialise(self):
        r"""Fixed-point sweeps of the linearised eikonal equation away from
        the interface cells.  Returns the final residual ``max | |G phi| - 1 |``
        over interior non-interface cells as a diagnostic."""
        self._mark_interface_cells()
        for _ in range(self.reinit_sweeps):
            self.phi_old.assign(self.phi)
            self._update_unit_gradient()
            self._reinit_solver.solve()
        g = self.cell_gradient()
        mag = np.sqrt((g * g).sum(axis=1))
        free = ((self.chi_fix.dat.data_ro < 0.5) & (self.chi_bnd.dat.data_ro < 0.5)
                & (np.abs(self.phi.dat.data_ro)
                   < REINIT_BAND_CELLS * self.cell_diam.dat.data_ro))
        loc = float(np.abs(mag[free] - 1.0).max()) if free.any() else 0.0
        return self.comm.allreduce(loc, op=_MPI.MAX)

    # ------------------------------------------------------------------
    # Ablation rate
    # ------------------------------------------------------------------
    def _ice_cell_indicator(self):
        r"""DG0 indicator of the cells the front reads its rate from: ice
        (``h > h_min``) inside the front (``phi <= 0``).  A cell beyond the
        front holds at most what the front left behind on its way out or
        carried in on its way through, not an ice column of its own, so
        neither its thickness nor a law evaluated on it means anything."""
        chi = Function(self.Q0, name="ice_cell_inside")
        chi.dat.data[:] = np.where((self.h_dg.dat.data_ro > self.h_min)
                                   & (self.phi.dat.data_ro <= 0.0), 1.0, 0.0)
        return chi

    def _ice_node_indicator(self, chi):
        r"""CG1 indicator of the vertices of the cells ``chi`` flags: the
        Dirichlet nodes of the extension into the water."""
        num = assemble(chi * TestFunction(self.Q1) * dx)
        self.ice_node.dat.data[:] = np.where(num.dat.data_ro > 0.0, 1.0, 0.0)

    def _rate_expr(self, rate):
        r"""UFL frontal ablation rate on ice (extended into the water
        afterwards): zero for ``none`` and ``fixed``, ``rate`` for
        ``prescribed``."""
        if self.law in ("none", "fixed"):
            return Constant(0.0)
        if rate is None:
            raise ValueError("the 'prescribed' law needs rate=")
        return rate

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------
    def advance(self, dt, u, rate=None):
        r"""Move the front by ``dt`` with ice velocity ``u`` (CG1) and, for
        the ``prescribed`` law, the ablation rate ``rate`` [m/yr of normal
        retreat on ice: a calving law's rate, or an imposed one].  Returns
        the mean ablation rate over front cells [m/yr] as a diagnostic."""
        self.n_advance += 1
        self.dt = float(dt)
        if self.anchor == "extent":
            return self._advance_extent(dt, u, rate)
        if self.law == "fixed":
            self.update_cell_fields()
            return 0.0
        t0 = perf_counter()
        # The unit gradient and stencil of the current phi, before the rate
        # is evaluated: a law reads ghat as the front normal, and the
        # level-set step uses the same gradient.
        self.phi_old.assign(self.phi)
        self._update_unit_gradient()
        # The rate is read on the ice cells inside the front alone, each
        # cell its own mean of the law.  A rate means nothing beyond the
        # front: a law gated on the grounded indicator is ungated there, a
        # stress law reads a dual state with no ice in it.  Reading those
        # cells too handed a front on grounded ice a good fraction of the
        # gated-off rate (CalvingMIP experiment 4 lost 17 % of its grounded
        # area where the ensemble lost 3 %), and a nodal mean over the ice
        # cells around each vertex still handed a grounded cell next to
        # floating ones a share of theirs.  Beyond the front the rate is
        # carried along the normal (solver (4)); a harmonic blend put the
        # floating ice's rate in the water ahead of a grounded tongue, that
        # water's phi fell behind the tongue's, and every reinitialisation
        # split the difference.
        chi = self._ice_cell_indicator()
        c_expr = self._rate_expr(rate)
        own = assemble(chi * c_expr * TestFunction(self.Q0) * dx(degree=2))
        self.c_own.dat.data[:] = own.dat.data_ro / self.cell_area
        self.chi_rate.assign(chi)
        # Extend the velocity from the vertices of those cells into the water.
        self._ice_node_indicator(chi)
        self.w_src.dat.data[:] = u.dat.data_ro
        ice = self.ice_node.dat.data_ro > 0.5
        bad = ~(np.isfinite(self.w_src.dat.data_ro[ice]).all(axis=1)
                & np.isfinite(self.c_own.dat.data_ro).all())
        if _global_count(bad, self.comm):
            raise FloatingPointError(
                "level set: non-finite velocity or ablation rate on ice; "
                "check the velocity and the ablation-rate inputs")
        self._ext_solver.solve()
        self._rate_ext_solver.solve()
        t1 = perf_counter()
        self.dt_c.assign(dt)
        self._ls_solver.solve()
        t2 = perf_counter()
        res = None
        if self.reinit_every > 0 and self.n_advance % self.reinit_every == 0:
            res = self.reinitialise()
        self.update_cell_fields()
        if self.n_advance in (1, max(self.reinit_every, 1)):
            PETSc.Sys.Print(
                f"    level set timing: rate+extension {t1 - t0:.1f}s, "
                f"advection {t2 - t1:.1f}s, reinit+cells {perf_counter() - t2:.1f}s"
                + (f", reinit residual {res:.2f}" if res is not None else ""))
        near = np.abs(self.phi.dat.data_ro) < self.cell_diam.dat.data_ro
        cnt = _global_count(near, self.comm)
        tot = self.comm.allreduce(float(self.c_cell.dat.data_ro[near].sum()))
        return tot / cnt if cnt else 0.0

    def _advance_extent(self, dt, u, rate=None):
        r"""One extent-anchored step: re-solve the eikonal problem at the
        current ice extent, evaluate the rate on ice cells, retreat the
        front by normal flow.  The removal itself is the caller's, through
        :meth:`calving_masks`."""
        t0 = perf_counter()
        n_seg = self.solve_eikonal_from_extent()
        if self.law == "fixed":
            self.update_cell_fields()
            return 0.0
        self.c_cell.interpolate(self.chi * self._rate_expr(rate))
        bad = ~np.isfinite(self.c_cell.dat.data_ro)
        if _global_count(bad, self.comm):
            raise FloatingPointError(
                "level set: non-finite ablation rate on the ice cells")
        t1 = perf_counter()
        # The unit gradient is already this extent's: the eikonal solve
        # refreshed it, and phi has not changed since.
        self.phi_old.assign(self.phi)
        self.dt_c.assign(dt)
        self._ls_solver.solve()
        self.update_cell_fields()
        if self.n_advance == 1:
            PETSc.Sys.Print(
                f"    level set: {n_seg} front facets; timing eikonal+rate "
                f"{t1 - t0:.1f}s, normal flow {perf_counter() - t1:.1f}s")
        front = self.front_len.dat.data_ro > 0.0
        cnt = _global_count(front, self.comm)
        tot = self.comm.allreduce(float(self.c_cell.dat.data_ro[front].sum()))
        return tot / cnt if cnt else 0.0

    def state_fields(self):
        return {"levelset": self.phi}
