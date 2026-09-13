r"""Úa-style adaptive remeshing (``AdaptMesh``) for icepack2 dual models.

A port of the mesh adaptation Úa runs between run-steps, read from UaSource
(``UaMain/AdaptMesh.m``, ``NewDesiredEleSizesAndElementsToRefineOrCoarsen2.m``,
``Error2EleSize.m``, ``GlobalRemeshing.m``, ``MapFbetweenMeshes.m``, Sep 2026).
Úa's ``GLmorphing`` mesh deformation is not ported: it has no caller in
``Ua.m``/``Ua2D.m`` and its remeshing hook is commented out as "broken anyhow".

Nothing here knows about a particular ice sheet. A project supplies

* a ``build_geometry()`` callable that builds its domain in the current gmsh
  model (curves, surface, physical groups) -- see ``remesh_global``;
* the fields of its own checkpoint, moved with the transfer helpers below.

The scheme, with Úa's names:

1. **Desired element size** at the nodes of the current mesh
   [``EleSizeDesired``]: start at ``MeshSizeMax``; for each enabled
   ``ExplicitMeshRefinementCriteria`` compute a nodal error proxy ``e`` and
   map it with ``Error2EleSize``, ``h = hMin + (e0/(e+e0))**(1/p) (hMax -
   hMin)`` (``e0`` = ``Scale``); take the minimum over criteria, ``MeshSize``
   if none fired; relax toward the current size (``W = 0.5``) and clip the
   ratio of change to ``[1/5, 5]``; THEN the absolute bands ``GLrange`` /
   ``CFrange`` and the PIG-TWG example's shelf and low-ground rules.
2. **Global remeshing** with gmsh: the nodal size field on the old mesh is the
   background view, the domain is rebuilt, the mesh regenerated, and Úa's
   element-count control rescales ``MeshSizeMin`` up to four times.
3. **Transfer** [``MapFbetweenMeshes``]: point-evaluation interpolation (Úa:
   FE shape functions, nearest outside); the geometry route ``bh-FROM-sBS``
   (Úa's default: move the surface, derive the thickness) or ``bs-FROM-hBS``.

Three things a DG0 model needs that Úa (nodal) never did, each measured on
the ISMIP7 32 km control: the DG0 front thickness smears under any
interpolation (``preserve_front``); a frozen apparent-mass-balance reference
cannot be moved (transfer ``physical_divergence`` and rebuild it); and a
frozen effective-pressure reference cannot be moved (transfer the ratio,
``rebuild_reference_pressure``).

Import this module before assembling any form: it reaches ``icepack2``
through ``icepack_tools.constants`` and Irksome refuses to load afterwards.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
from firedrake import (
    And,
    CellVolume,
    Constant,
    FacetNormal,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    TestFunction,
    assemble,
    conditional,
    dot,
    dS,
    ds,
    dx,
    grad,
    inner,
    interpolate,
    jump,
    max_value,
    project,
    sqrt,
)
from firedrake.petsc import PETSc
from mpi4py import MPI

from .constants import ice_density, water_density
from .geometry import cg1_lift
from .grounding import effective_pressure

__all__ = [
    "CRITERIA", "Criterion", "AdaptMeshConfig", "error_to_ele_size", "dirac_delta",
    "effective_strain_rate_nodal", "current_element_size_nodal",
    "facet_midpoints_and_jumps", "gather_points", "grounding_line_points",
    "calving_front_points", "nodal_distance_to", "desired_element_size",
    "physical_groups", "remesh_global", "cross_mesh_transfer", "boundary_cells",
    "preserve_front", "fv_flux_divergence", "physical_divergence",
    "surface_route_thickness", "rebuild_reference_pressure",
]


def _val(c):
    try:
        return float(c)
    except TypeError:
        return float(c.values()[0])


RHO_RATIO = _val(ice_density) / _val(water_density)      # rho_i / rho_w

CRITERIA = (
    "effective strain rates",
    "effective strain rates gradient",
    "flotation",
    "thickness gradient",
    "upper surface gradient",
    "lower surface gradient",
    "|dhdt|",
    "dhdt gradient",
)


@dataclass
class Criterion:
    r"""One entry of Úa's ``CtrlVar.ExplicitMeshRefinementCriteria``."""
    name: str
    scale: float
    ele_min: float | None = None
    ele_max: float | None = None
    p: float = 1.0
    use: bool = True


@dataclass
class AdaptMeshConfig:
    r"""Úa's ``CtrlVar`` fields that shape one adaptation, Úa's defaults."""
    mesh_size: float = 10e3            # CtrlVar.MeshSize
    mesh_size_min: float = 1e3         # CtrlVar.MeshSizeMin
    mesh_size_max: float = 10e3        # CtrlVar.MeshSizeMax
    gl_range: list = field(default_factory=list)   # MeshAdapt.GLrange rows (d, h)
    cf_range: list = field(default_factory=list)   # MeshAdapt.CFrange rows (d, h)
    criteria: list = field(default_factory=list)   # ExplicitMeshRefinementCriteria
    relaxation_w: float = 0.5          # W (hard-coded in Úa)
    max_ratio_change: float = 5.0      # MaxRatioOfChangeInEleSizeDuringAdaptMeshing
    min_ratio_change: float = 0.2      # MinRatioOfChangeInEleSizeDuringAdaptMeshing
    max_number_of_elements: int = 0    # MaxNumberOfElements (0 = no rescaling loop)
    upper_limit_factor: float = 1.3    # MaxNumberOfElementsUpperLimitFactor
    lower_limit_factor: float = 0.0    # MaxNumberOfElementsLowerLimitFactor
    thick_min: float = 1.0             # ThickMin (OutsideValue.h)
    gl_threshold: float = 0.5          # GLthreshold
    front_hmin: float = 1.0            # ice / no-ice threshold for the front
    refine_dirac_width: float = 100.0  # RefineDiracDeltaWidth [m]
    refine_dirac_offset: float = 0.0   # RefineDiracDeltaOffset
    transfer: str = "interpolate"      # "interpolate" (Úa) | "project" (conservative DG0)
    geometry: str = "bh-FROM-sBS"      # MapOldToNew.Transient.Geometry
    keep_current: bool = False         # null test: desired size = current size
    front_preserve: bool = True        # new boundary cells take the nearest old boundary cell's h
    # PIG-TWG's DefineDesiredEleSize.m rules (UaExamples)
    shelf_size: float | None = None    # floating ice gets min(h, shelf_size)
    low_surface: tuple | None = None   # (elevation [m], size): s below it gets min(h, size)

    PRESETS = {
        # Úa's own Antarctic sizes. Whole-continent Úa (Úa-FESOM coupling, GMD
        # 18, 2025): 180 km interior, 4 km high strain, 2 km at the grounding
        # line, ~250,000 elements. PIG-TWG example (UaExamples): MeshSize =
        # Max/2, MeshSizeMin = Max/20, shelves and s < 1500 m at Max/5, strain
        # Scale 0.001, AdaptMeshMaxIterations 5. MISMIP+: GLrange [20000 5000;
        # 10000 2000; 5000 500]. Inferred here: the GL bands (pan-Antarctic
        # form of the above) and 10 km shelves (4 km continent-wide would be
        # ~220k elements on its own).
        "ua": dict(mesh_size_max=180e3, mesh_size=90e3, mesh_size_min=2e3,
                   shelf_size=10e3, low_surface=(1500.0, 36e3),
                   gl_range=[(10e3, 4e3), (5e3, 2e3)],
                   max_number_of_elements=250_000,
                   criteria=[Criterion("effective strain rates", 0.001, ele_min=4e3)]),
    }

    @staticmethod
    def _rows(text):
        rows = []
        for item in (text or "").split(","):
            item = item.strip()
            if item:
                d, h = item.split(":")
                rows.append((float(d), float(h)))
        return rows

    @classmethod
    def from_env(cls, prefix="ISMIP7_ADAPT_"):
        r"""Build from ``<prefix>*`` variables (``GL_RANGE="5000:2000,1000:500"``,
        ``CRITERIA="effective strain rates:0.01,flotation:0.001:1:500:2000"``
        as ``name:scale[:p[:min:max]]``, ``PRESET=ua``, ...); a preset fills
        what is unset."""
        env = lambda k, d=None: os.environ.get(prefix + k, d)  # noqa: E731
        preset = (env("PRESET", "") or "").lower()
        if preset and preset not in cls.PRESETS:
            raise ValueError(f"{prefix}PRESET must be one of {sorted(cls.PRESETS)} or unset")
        base = dict(cls.PRESETS.get(preset, {}))
        cfg = cls(
            mesh_size=float(env("MESH_SIZE", base.get("mesh_size", 10e3))),
            mesh_size_min=float(env("MESH_SIZE_MIN", base.get("mesh_size_min", 1e3))),
            mesh_size_max=float(env("MESH_SIZE_MAX", base.get("mesh_size_max", 10e3))),
            gl_range=cls._rows(env("GL_RANGE")) if env("GL_RANGE") is not None else list(base.get("gl_range", [])),
            cf_range=cls._rows(env("CF_RANGE", "")),
            shelf_size=float(env("SHELF_SIZE")) if env("SHELF_SIZE") else base.get("shelf_size"),
            low_surface=(tuple(float(t) for t in env("LOW_SURFACE").split(":"))
                         if env("LOW_SURFACE") else base.get("low_surface")),
            relaxation_w=float(env("RELAXATION_W", 0.5)),
            max_ratio_change=float(env("MAX_RATIO_CHANGE", 5.0)),
            min_ratio_change=float(env("MIN_RATIO_CHANGE", 0.2)),
            max_number_of_elements=int(float(env("MAX_ELEMENTS", base.get("max_number_of_elements", 0)))),
            thick_min=float(env("THICK_MIN", 1.0)),
            front_hmin=float(os.environ.get("ISMIP7_FRONT_HMIN", env("FRONT_HMIN", 1.0))),
            refine_dirac_width=float(env("DIRAC_WIDTH", 100.0)),
            transfer=env("TRANSFER", "interpolate"),
            geometry=env("GEOMETRY", "bh-FROM-sBS"),
            keep_current=env("KEEP_CURRENT", "0") == "1",
            front_preserve=env("FRONT_PRESERVE", "1") == "1",
        )
        if cfg.geometry not in ("bh-FROM-sBS", "bs-FROM-hBS"):
            raise ValueError(f"{prefix}GEOMETRY must be bh-FROM-sBS or bs-FROM-hBS")
        crit_env = env("CRITERIA")
        if crit_env is None:
            cfg.criteria = list(base.get("criteria", []))
        for item in (crit_env or "").split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            name = parts[0].strip()
            if name not in CRITERIA:
                raise ValueError(f"unknown Úa refinement criterion {name!r}; one of {CRITERIA}")
            c = Criterion(name=name, scale=float(parts[1]))
            if len(parts) > 2 and parts[2]:
                c.p = float(parts[2])
            if len(parts) > 4:
                c.ele_min, c.ele_max = float(parts[3]), float(parts[4])
            cfg.criteria.append(c)
        return cfg


# ---------------------------------------------------------------------------
# Úa building blocks
# ---------------------------------------------------------------------------

def error_to_ele_size(e, e0, h_min, h_max, p=1.0):
    r"""Úa ``Error2EleSize``: ``h = h_min + (e0/(e+e0))^(1/p) (h_max - h_min)``."""
    e = np.asarray(e, dtype=float)
    return h_min + (e0 / (e + e0)) ** (1.0 / p) * (h_max - h_min)


def dirac_delta(k, x, x0=0.0):
    r"""Úa ``DiracDelta(k, x, x0) = 0.5 k sech^2(k (x - x0))``."""
    return k / (2.0 * np.cosh(np.clip(k * (np.asarray(x, dtype=float) - x0), -300, 300)) ** 2)


def _nodal(expr, Qc):
    r"""Úa ``ProjectFintOntoNodes``: L2-project a cellwise expression to CG1."""
    return Function(Qc).project(expr)


def _grad_mag_nodal(f_cg1, Qc):
    g = grad(f_cg1)
    return _nodal(sqrt(inner(g, g) + Constant(1e-30)), Qc)


def effective_strain_rate_nodal(u, Qc):
    r"""Úa ``CalcHorizontalNodalStrainRates``: ``sqrt(exx^2 + eyy^2 + exx eyy
    + exy^2)`` at the nodes."""
    exx = _nodal(grad(u)[0, 0], Qc)
    eyy = _nodal(grad(u)[1, 1], Qc)
    exy = _nodal(0.5 * (grad(u)[0, 1] + grad(u)[1, 0]), Qc)
    return Function(Qc).interpolate(sqrt(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2 + Constant(1e-30)))


def current_element_size_nodal(mesh, Qc):
    r"""Úa ``EleSizeCurrent = sqrt(M * EleArea)``: the square root of the mean
    area of the elements around each node (0.66 of an equilateral edge; Úa
    mixes this with edge-length targets and so does this port)."""
    Q0 = FunctionSpace(mesh, "DG", 0)
    mean_area = cg1_lift(Function(Q0).interpolate(CellVolume(mesh)))
    out = Function(Qc)
    out.dat.data[:] = np.sqrt(np.maximum(mean_area.dat.data_ro, 1e-30))
    return out


def facet_midpoints_and_jumps(mesh, flags):
    r"""Per-facet midpoints and the jump of each DG0 flag across interior
    facets, via an HDiv-trace test function (one dof per facet). Exterior
    facets carry the flag's own value. Returns ``(xm, ym, jumps, is_exterior)``
    for this rank's owned facets."""
    T = FunctionSpace(mesh, "HDiv Trace", 0)
    vt = TestFunction(T)
    vp = vt("+")                      # a trace is single-valued but dS wants a restriction
    x = SpatialCoordinate(mesh)
    length = assemble(vp * dS + vt * ds).dat.data_ro
    xm = assemble(x[0] * vp * dS + x[0] * vt * ds).dat.data_ro / length
    ym = assemble(x[1] * vp * dS + x[1] * vt * ds).dat.data_ro / length
    ext = assemble(vt * ds).dat.data_ro > 0.5 * length
    jumps = {name: assemble(jump(f) * vp * dS + f * vt * ds).dat.data_ro / length
             for name, f in flags.items()}
    return xm, ym, jumps, ext


def gather_points(comm, xs, ys):
    pts = np.column_stack([xs, ys]) if len(xs) else np.zeros((0, 2))
    allp = comm.allgather(pts)
    return np.vstack(allp) if allp else np.zeros((0, 2))


def grounding_line_points(mesh, H, b, cfg):
    r"""Midpoints of the facets between grounded and floating ice cells (Úa's
    GL at the ``GLthreshold`` crossing), gathered globally."""
    Q0 = FunctionSpace(mesh, "DG", 0)
    rr = Constant(RHO_RATIO)
    ice = conditional(H > cfg.front_hmin, 1.0, 0.0)
    grounded = Function(Q0).interpolate(conditional(rr * H + b > 0.0, 1.0, 0.0) * ice)
    floating = Function(Q0).interpolate(conditional(rr * H + b <= 0.0, 1.0, 0.0) * ice)
    xm, ym, jumps, ext = facet_midpoints_and_jumps(mesh, {"g": grounded, "f": floating})
    sel = (~ext) & (np.abs(jumps["g"]) > 0.5) & (np.abs(jumps["f"]) > 0.5)
    return gather_points(mesh.comm, xm[sel], ym[sel])


def calving_front_points(mesh, H, cfg):
    r"""Midpoints of the facets between ice and no-ice cells, plus exterior
    facets of ice cells, gathered globally."""
    Q0 = FunctionSpace(mesh, "DG", 0)
    ice = Function(Q0).interpolate(conditional(H > cfg.front_hmin, 1.0, 0.0))
    xm, ym, jumps, ext = facet_midpoints_and_jumps(mesh, {"i": ice})
    sel = ((~ext) & (np.abs(jumps["i"]) > 0.5)) | (ext & (jumps["i"] > 0.5))
    return gather_points(mesh.comm, xm[sel], ym[sel])


def nodal_distance_to(mesh, Qc, points):
    r"""Distance from every CG1 node to the nearest of ``points``."""
    from scipy.spatial import cKDTree
    X = mesh.coordinates.dat.data_ro[:, :2]
    d = Function(Qc)
    d.dat.data[:] = np.inf if points.shape[0] == 0 else cKDTree(points).query(X, k=1)[0]
    return d


# ---------------------------------------------------------------------------
# Step 1: desired element size
# ---------------------------------------------------------------------------

def desired_element_size(mesh, cfg, H, b, u=None, dhdt=None, weight=None, log=PETSc.Sys.Print):
    r"""Úa ``NewDesiredEleSizesAndElementsToRefineOrCoarsen2`` for the
    ``explicit:global`` method: the CG1 ``EleSizeDesired`` on ``mesh`` and a
    dict of diagnostics. ``weight`` (CG1, 0..1) multiplies every relative
    criterion's error proxy, e.g. an observation mask."""
    Qc = FunctionSpace(mesh, "CG", 1)
    comm = mesh.comm
    n_nodes = Qc.dof_dset.size
    h_des = Function(Qc)
    h_des.dat.data[:] = cfg.mesh_size_max
    fired = False
    rr = Constant(RHO_RATIO)

    s = Function(H.function_space()).interpolate(max_value(b + H, (Constant(1.0) - rr) * H))
    H_cg, b_cg, s_cg = cg1_lift(H), cg1_lift(b), cg1_lift(s)

    for c in cfg.criteria:
        if not c.use:
            continue
        h_min = cfg.mesh_size_min if c.ele_min is None else c.ele_min
        h_max = cfg.mesh_size_max if c.ele_max is None else c.ele_max
        name = c.name
        if name in ("effective strain rates", "effective strain rates gradient"):
            if u is None:
                log(f"  adapt: criterion {name!r} skipped (no velocity given)")
                continue
            e = effective_strain_rate_nodal(u, Qc)
            if name.endswith("gradient"):
                e = _grad_mag_nodal(e, Qc)
        elif name == "flotation":
            hf = Function(Qc).interpolate(Constant(1.0 / RHO_RATIO) * max_value(-b_cg, Constant(0.0)))
            e = Function(Qc)
            e.dat.data[:] = dirac_delta(1.0 / cfg.refine_dirac_width,
                                        H_cg.dat.data_ro - hf.dat.data_ro, cfg.refine_dirac_offset)
        elif name == "thickness gradient":
            e = _grad_mag_nodal(H_cg, Qc)
        elif name == "upper surface gradient":
            e = _grad_mag_nodal(s_cg, Qc)
        elif name == "lower surface gradient":
            e = _grad_mag_nodal(b_cg, Qc)
        elif name in ("|dhdt|", "dhdt gradient"):
            if dhdt is None:
                log(f"  adapt: criterion {name!r} skipped (no dhdt given)")
                continue
            dh = dhdt if dhdt.function_space() == Qc else cg1_lift(dhdt)
            e = Function(Qc).interpolate(abs(dh)) if name == "|dhdt|" else _grad_mag_nodal(dh, Qc)
            emax = comm.allreduce(float(np.abs(e.dat.data_ro).max()) if n_nodes else 0.0, op=MPI.MAX)
            if emax < 1e-5:
                log(f"  adapt: criterion {name!r} too small to be of use, discarded")
                continue
        else:
            raise ValueError(name)
        e_vals = e.dat.data_ro if weight is None else e.dat.data_ro * weight.dat.data_ro
        h_c = error_to_ele_size(e_vals, c.scale, h_min, h_max, c.p)
        h_des.dat.data[:] = np.minimum(h_des.dat.data_ro, h_c)
        fired = True
        lo = comm.allreduce(float(h_c.min()) if n_nodes else np.inf, op=MPI.MIN)
        log(f"  adapt: criterion {name!r} scale={c.scale:g} -> min desired size {lo:.0f} m")

    all_max = comm.allreduce(bool(np.all(h_des.dat.data_ro >= cfg.mesh_size_max - 1e-9)), op=MPI.LAND)
    if not fired or all_max:
        h_des.dat.data[:] = cfg.mesh_size
        log(f"  adapt: no relative criterion fired; desired size = MeshSize {cfg.mesh_size:g} m")

    # Relaxation toward the current size, then the ratio-of-change limits.
    h_cur = current_element_size_nodal(mesh, Qc)
    if cfg.keep_current:
        h_des.dat.data[:] = h_cur.dat.data_ro * (4.0 / np.sqrt(3.0)) ** 0.5   # sqrt(area) -> edge
        log("  adapt: keep_current: desired size = current element size")
    W = cfg.relaxation_w if not cfg.keep_current else 1.0
    h_des.dat.data[:] = W * h_des.dat.data_ro + (1.0 - W) * h_cur.dat.data_ro
    ratio = h_des.dat.data_ro / h_cur.dat.data_ro
    hi, lo_ = ratio > cfg.max_ratio_change, ratio < cfg.min_ratio_change
    h_des.dat.data[hi] = cfg.max_ratio_change * h_cur.dat.data_ro[hi]
    h_des.dat.data[lo_] = cfg.min_ratio_change * h_cur.dat.data_ro[lo_]

    # Absolute bands from the grounding line and the calving front.
    diag = {}
    if cfg.gl_range:
        gl = grounding_line_points(mesh, H, b, cfg)
        d_gl = nodal_distance_to(mesh, Qc, gl)
        for dist, size in cfg.gl_range:
            if size < cfg.mesh_size_min:
                log(f"  adapt: GLrange size {size:g} < MeshSizeMin {cfg.mesh_size_min:g}, using the latter")
            sel = d_gl.dat.data_ro < dist
            h_des.dat.data[sel] = np.minimum(h_des.dat.data_ro[sel], max(size, cfg.mesh_size_min))
        diag["n_gl_points"] = int(gl.shape[0])
    if cfg.cf_range:
        cf = calving_front_points(mesh, H, cfg)
        d_cf = nodal_distance_to(mesh, Qc, cf)
        for dist, size in cfg.cf_range:
            sel = d_cf.dat.data_ro < dist
            h_des.dat.data[sel] = np.minimum(h_des.dat.data_ro[sel], max(size, cfg.mesh_size_min))
        diag["n_cf_points"] = int(cf.shape[0])

    # Úa's user hook (DefineDesiredEleSize) runs after the bands; PIG-TWG's is
    # two rules: floating ice and low ground get a fixed size.
    if cfg.shelf_size is not None:
        fl = Function(H.function_space()).interpolate(
            conditional(And(rr * H + b <= 0.0, H > cfg.front_hmin), 1.0, 0.0))
        fl_n = cg1_lift(fl).dat.data_ro > 0.9                   # Úa: GF.node < 0.1
        h_des.dat.data[fl_n] = np.minimum(h_des.dat.data_ro[fl_n], cfg.shelf_size)
    if cfg.low_surface is not None:
        elev, size = cfg.low_surface
        low = s_cg.dat.data_ro < elev
        h_des.dat.data[low] = np.minimum(h_des.dat.data_ro[low], size)

    h_des.dat.data[:] = np.clip(h_des.dat.data_ro, cfg.mesh_size_min, cfg.mesh_size_max)
    lo = comm.allreduce(float(h_des.dat.data_ro.min()) if n_nodes else np.inf, op=MPI.MIN)
    hi = comm.allreduce(float(h_des.dat.data_ro.max()) if n_nodes else -np.inf, op=MPI.MAX)
    diag.update(size_min=lo, size_max=hi)
    log(f"  adapt: desired element size in [{lo:.0f}, {hi:.0f}] m"
        + (f", GL points {diag['n_gl_points']}" if "n_gl_points" in diag else "")
        + (f", front points {diag['n_cf_points']}" if "n_cf_points" in diag else ""))
    return h_des, diag


# ---------------------------------------------------------------------------
# Step 2: global remeshing with gmsh (rank 0)
# ---------------------------------------------------------------------------

def _gather_nodal_field(mesh, f):
    comm = mesh.comm
    allX = comm.gather(np.asarray(mesh.coordinates.dat.data_ro[:, :2]), root=0)
    allv = comm.gather(np.asarray(f.dat.data_ro), root=0)
    return (np.vstack(allX), np.concatenate(allv)) if comm.rank == 0 else (None, None)


def physical_groups(msh_path):
    r"""``{name: tag}`` of the 1-D physical groups of a .msh file."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.open(msh_path)
    out = {gmsh.model.getPhysicalName(d, t): t for d, t in gmsh.model.getPhysicalGroups(1)}
    gmsh.finalize()
    return out


def remesh_global(mesh, h_des, cfg, out_msh, build_geometry, *, reference_msh=None,
                  sidecar_in=None, sidecar_out=None, log=PETSc.Sys.Print):
    r"""Úa ``GlobalRemeshing`` with gmsh. Collective; gmsh runs on rank 0.

    ``build_geometry()`` is called after ``gmsh.model.add`` and must create the
    domain's curves, surface and physical groups with the gmsh API. The nodal
    size field of the old mesh is the background view (a Delaunay
    triangulation of the old nodes), and Úa's element-count control rescales
    ``MeshSizeMin`` up to four times. If ``reference_msh`` is given the new
    mesh must expose the same physical groups (or the boundary conditions land
    on the wrong facets), and ``sidecar_in`` is then copied to ``sidecar_out``.
    Returns the element count.
    """
    comm = mesh.comm
    X, v = _gather_nodal_field(mesh, h_des)
    n_ele = None
    if comm.rank == 0:
        import gmsh
        from scipy.spatial import Delaunay
        tri = Delaunay(X).simplices
        sizes = v.copy()
        size_min = cfg.mesh_size_min

        def generate(sizes):
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 1)
            gmsh.model.add("adapt")
            build_geometry()
            view = gmsh.view.add("ua size field")
            P, S = X[tri], sizes[tri]
            data = np.column_stack([P[:, :, 0], P[:, :, 1], np.zeros_like(S), S]).ravel()
            gmsh.view.addListData(view, "ST", tri.shape[0], data.tolist())
            bg = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(bg, "ViewTag", view)
            gmsh.model.mesh.field.setAsBackgroundMesh(bg)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.model.mesh.generate(2)
            tags, _ = gmsh.model.mesh.getElementsByType(2)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(out_msh)
            groups = {gmsh.model.getPhysicalName(d, t): t for d, t in gmsh.model.getPhysicalGroups(1)}
            gmsh.finalize()
            return len(tags), groups

        n_ele, groups = generate(sizes)
        log(f"  adapt: remeshed -> {n_ele} elements")
        if cfg.max_number_of_elements > 0:
            N, it = cfg.max_number_of_elements, 0
            while (n_ele > cfg.upper_limit_factor * N or n_ele < cfg.lower_limit_factor * N) and it < 4:
                scaling = np.sqrt(n_ele / N)
                max_e, min_e = float(sizes.max()), float(sizes.min())
                new_min = min(min_e * scaling, 0.9 * cfg.mesh_size_max)
                sizes = (new_min + (cfg.mesh_size_max - new_min) * (sizes - min_e) / (max_e - min_e)
                         if max_e != min_e else np.full_like(sizes, new_min))
                it += 1
                if new_min != size_min:
                    log(f"  adapt: MeshSizeMin rescaled {size_min:.0f} -> {new_min:.0f} m "
                        f"(Nele {n_ele} vs MaxNumberOfElements {N})")
                size_min = new_min
                n_ele, groups = generate(sizes)
                log(f"  adapt: remeshed -> {n_ele} elements")
        if reference_msh is not None:
            old_groups = physical_groups(reference_msh)
            if groups != old_groups:
                raise RuntimeError("remesh produced different physical groups than the reference mesh: "
                                   f"{len(groups)} vs {len(old_groups)}")
        if sidecar_in and sidecar_out:
            with open(sidecar_in) as f:
                side = json.load(f)
            with open(sidecar_out, "w") as f:
                json.dump(side, f)
    comm.barrier()
    return comm.bcast(n_ele, root=0)


# ---------------------------------------------------------------------------
# Step 3: transfer helpers (MapFbetweenMeshes and the DG0 additions)
# ---------------------------------------------------------------------------

def cross_mesh_transfer(f_old, mesh_new, how="interpolate", default=0.0):
    r"""Move ``f_old`` onto ``mesh_new`` in a space of the same element.
    ``interpolate`` is Úa's point evaluation (``default`` outside the old
    mesh); ``project`` is the conservative supermesh projection, which
    Firedrake refuses between independently partitioned meshes in parallel,
    so it needs one rank."""
    V_new = FunctionSpace(mesh_new, f_old.function_space().ufl_element())
    if how == "project":
        if mesh_new.comm.size > 1:
            raise RuntimeError("transfer='project' needs the adapt step on ONE rank (mpiexec -n 1)")
        return project(f_old, V_new)
    return assemble(interpolate(f_old, V_new, allow_missing_dofs=True, default_missing_val=default))


def boundary_cells(mesh, Q0):
    r"""Owned DG0 dofs of the cells with an exterior facet, and their centroids."""
    x = SpatialCoordinate(mesh)
    on_bnd = assemble(TestFunction(Q0) * ds).dat.data_ro > 0.0
    cells = np.flatnonzero(on_bnd)
    xc = Function(Q0).interpolate(x[0]).dat.data_ro[cells]
    yc = Function(Q0).interpolate(x[1]).dat.data_ro[cells]
    return cells, xc, yc


def preserve_front(mesh_old, H_old, mesh_new, H_new):
    r"""Overwrite every new boundary cell's value with that of the nearest old
    boundary cell (by centroid, globally): the DG0 analogue of Úa's boundary
    nodes interpolating from the old boundary edge alone. Returns the count."""
    from scipy.spatial import cKDTree
    comm = mesh_new.comm
    oc, oxc, oyc = boundary_cells(mesh_old, H_old.function_space())
    old_pts = gather_points(comm, oxc, oyc)
    old_vals = np.concatenate(comm.allgather(np.asarray(H_old.dat.data_ro[oc])))
    if old_pts.shape[0] == 0:
        return 0
    nc, nxc, nyc = boundary_cells(mesh_new, H_new.function_space())
    if nc.size:
        _, j = cKDTree(old_pts).query(np.column_stack([nxc, nyc]), k=1)
        H_new.dat.data[nc] = old_vals[j]
    return int(comm.allreduce(nc.size))


def fv_flux_divergence(mesh, h_dg, u):
    r"""The DG0 upwind facet-flux divergence per unit area: the discrete
    ``div(h u)`` of a first-order finite-volume transport."""
    Q0 = h_dg.function_space()
    phi = TestFunction(Q0)
    n = FacetNormal(mesh)
    un = dot(u, n)
    unp = (un + abs(un)) / 2
    flux = assemble((unp("+") * h_dg("+") - unp("-") * h_dg("-")) * jump(phi) * dS + unp * h_dg * phi * ds)
    area = Function(Q0).interpolate(CellVolume(mesh))
    out = Function(Q0, name="flux_div")
    out.dat.data[:] = flux.dat.data_ro / area.dat.data_ro
    return out


def physical_divergence(mesh, h_dg, u, a_ref):
    r"""``P = flux/area - a_ref``: the flux divergence net of a frozen
    apparent-mass-balance correction. The correction cancels the OLD mesh's
    discrete divergence spike by spike and cannot be moved; ``P`` can, and the
    new mesh rebuilds ``a_ref = flux_new/area - P`` with its own operator."""
    d = fv_flux_divergence(mesh, h_dg, u)
    out = Function(h_dg.function_space(), name="phys_div")
    out.dat.data[:] = d.dat.data_ro - a_ref.dat.data_ro
    return out


def surface_route_thickness(s_old, b_new, mesh_new, Q_g):
    r"""Úa's default ``bh-FROM-sBS``: move the surface, derive the thickness
    from it and the (re-sampled) bed, ``h = min(s - b, s rho_w/(rho_w - rho_i))``.
    For a DG0 surface the moved field is its volume-preserving CG1 lift, the
    analogue of Úa's nodal ``s``."""
    s_lift = cg1_lift(s_old) if s_old.function_space().ufl_element().degree() == 0 else s_old
    s_c = Function(Q_g).interpolate(cross_mesh_transfer(s_lift, mesh_new, "interpolate", 0.0))
    h_ground = s_c.dat.data_ro - b_new.dat.data_ro
    h_float = s_c.dat.data_ro / (1.0 - RHO_RATIO)
    H = Function(Q_g)
    H.dat.data[:] = np.where(s_c.dat.data_ro > 0.0, np.maximum(np.minimum(h_ground, h_float), 0.0), 0.0)
    return H


def rebuild_reference_pressure(H_old, s_old, N_ref_old, H_new, s_new, mesh_new, nhat_cap=3.0):
    r"""A frozen effective-pressure reference ``N_ref`` (the t=0 value, read as
    ``N_hat = N_eff/N_ref`` capped at ``nhat_cap``) cannot be moved: re-sampling
    the bed changes ``N_eff`` cell by cell while an interpolated reference does
    not follow. Measured on a same-resolution remesh the grounded-cell ratio
    spread went from 0.91-1.48 to 0.36-2.97 with 10% of cells losing their
    reference. What transfers is the RATIO; this rebuilds
    ``N_ref = N_eff_new / ratio`` on the new geometry."""
    Q_old = H_old.function_space()
    N_old = Function(Q_old).interpolate(max_value(effective_pressure(H_old, s_old), Constant(0.0)))
    ratio_old = Function(Q_old, name="nhat")
    nr = N_ref_old.dat.data_ro
    ratio_old.dat.data[:] = np.where(nr > 1e-6, np.minimum(N_old.dat.data_ro / np.maximum(nr, 1e-6), nhat_cap), nhat_cap)
    ratio_new = cross_mesh_transfer(ratio_old, mesh_new, "interpolate", default=1.0)
    Q_new = H_new.function_space()
    N_new = Function(Q_new).interpolate(max_value(effective_pressure(H_new, s_new), Constant(0.0)))
    rn = np.clip(ratio_new.dat.data_ro, 1e-3, nhat_cap)
    out = Function(Q_new, name="N_ref")
    out.dat.data[:] = np.where(N_new.dat.data_ro > 0.0, N_new.dat.data_ro / rn, 0.0)
    return out
