r"""Calving laws: the frontal ablation rate a level-set front retreats at.

A law turns a front state into a **calving rate** ``c`` on the ice cells:
metres per year of retreat along the outward front normal, so the front
moves at ``u . n - c``.  :class:`icepack_tools.levelset.LevelSet` carries
the front and knows no calving physics; it takes the rate as its
``prescribed`` input.  Every law is defined here and nowhere else, and
every project that runs a front (an idealised MIP, an ice-sheet forward,
a law tuned against observed fronts) selects from the one registry
:data:`LAWS`, so a parameter tuned in one of them means the same thing in
all of them.  Adding a law is adding a class; a project-specific one (an
experiment's imposed front motion, say) registers itself from its own
module with :func:`register`, or from a file with :func:`load_module`.

What a law reads -- the front state
-----------------------------------
A law reads its inputs off a *front state*: any object with the
attributes below.  :class:`FrontState` builds one from a dual solution, a
thickness, a bed and a level set; a model can also be its own state.

``u``            velocity (CG1)
``M``            membrane stress from the dual solve (DG0 tensor)
``tau``          basal stress from the dual solve
``h``            thickness (DG0, the transport field)
``b``            bed elevation
``haf``          height above flotation, ``h - (rho_w / rho_i) max(-b, 0)``
``chi_gr``       grounded indicator, ``haf > 0``
``nfront``       outward unit normal of the front (the level set's unit
                 gradient)
``A``, ``n``     fluidity and Glen exponent; read only by the strain-rate
                 form of the von Mises law
``rho_i``,       the densities ``haf`` was computed with, in model units;
``rho_w``        optional, icepack2's by default.  A law that needs a
                 density takes it from here, so its threshold and the
                 state's flotation test are one pair of numbers

``M`` and ``tau`` are the point of doing this in the dual form.  A primal
SSA model has only the velocity, and a stress-based law must differentiate
it and push the strain rate back through the flow law; here the stress is
a solved field, cell by cell, including on the front cells where the
terminus force balance holds.

icepack2's membrane stress convention
-------------------------------------
Pinned by icepack2's own floating-shelf test: the analytic unconfined
shelf has ``eps_xx = A (rho' g h / 4)^n``, and carrying that through
``ice_shelf_momentum_balance`` (``h M_xx = rho' g h^2 / 2``) and
``flow_law`` gives

.. code::

    M = tau_h + tr(tau_h) I           (tau_h: horizontal deviatoric stress)
    tau_h = M - tr(M) / 3 * I
    tau_zz = -tr(tau_h) = -tr(M) / 3
    |M|^2 = (M:M - tr(M)^2 / 3) / 2 = tau_E^2   (the effective stress)

so ``M`` is the SSA membrane (resistive) stress ``2 tau_xx + tau_yy`` on
the diagonal, and the deviatoric stress is its 3-D-traceless part.
``test/calving_test.py`` checks this against a solved shelf, not just the
algebra.

Von Mises (Morlighem et al. 2016, GRL 43)
-----------------------------------------
``c = |u| sigma~ / sigma_max`` with a tensile von Mises stress and separate
thresholds for grounded and floating ice.  Morlighem defines it from the
strain rate, ``sigma~ = sqrt(3) B eps~^(1/n)``,
``eps~^2 = (max(eps1, 0)^2 + max(eps2, 0)^2) / 2`` over the principal
horizontal strain rates, ``B = A^(-1/n)``; that is ``vonmises_strain``,
which reads ``A`` and ``n`` off the state.  Its stress counterpart,
``vonmises``, is the same combination of the principal horizontal
deviatoric stresses,

.. code::

    sigma~ = sqrt(3) tau~,   tau~^2 = (max(tau1, 0)^2 + max(tau2, 0)^2) / 2

read straight from ``M``.  The von Mises law uses the tensile principal
stresses because Morlighem's law is defined that way; the force-balance
law below uses the front-normal component instead because that is what
*its* papers define (see :func:`resistive_stress`).  The two forms agree
wherever all strain is tensile (``eps~ = eps_E``) and otherwise differ by
``(eps~ / eps_E)^(1/n - 1)``, a factor ``2^(1/3) = 1.26`` in uniaxial
extension, so a ``sigma_max`` tuned for one is not a ``sigma_max`` for the
other.  The thresholds default to the values in common use, 1 MPa
grounded and 150 kPa floating; Morlighem et al. (2016) calibrate them per
basin.

Horizontal force balance (Buck 2023; Coffey et al. 2024; Coffey & Lai 2025;
Slater & Wagner 2025)
--------------------------------------------------------------------------
The crevasse-depth family made self-consistent: the horizontal force on
the ice between a surface and a basal crevasse must equal the force in the
unfractured ice, which lets the crevasses concentrate stress in the
intact ligament and penetrate further than Nye's zero-stress estimate.
Full-thickness penetration -- calving -- happens when the depth-averaged
resistive stress ``R_xx`` (the membrane stress normal to the front, i.e.
icepack2's ``M`` itself) reaches a threshold that depends only on the
geometry and the densities.  Slater & Wagner (2025, eq. 21, with a
tensile strength ``sigma_max`` and a basal-crevasse water density
``rho_c``) give it as, with ``a = max(H_ab, 0) / H`` the height above
buoyancy fraction,

.. code::

    R_crit / (rho_i g H) = (1 - (rho_i / rho_c) (1 - a)^2) / 2
                           + rho_c sigma~^2 / (2 (rho_c - rho_i)),   sigma~ = sigma_max / (rho_i g H)

For ``sigma_max = 0`` and seawater in the crevasse this is exactly Coffey
& Lai's ``B = 0`` (their eq. 11 with ``lambda = 1 - a``): an
*unbuttressed* front is at the calving threshold, so a freely spreading
shelf calves at its front and a grounded front with no basal drag calves
at flotation.  The zero-stress (Nye / Nick) counterpart, kept for
comparison, needs twice the stress on a shelf:

.. code::

    R_crit^ZS / (rho_i g H) = 1 - (rho_i / rho_c) (1 - a)

Both are position criteria.  In this rate framework they are used the way
the thickness law and the von Mises law are:
``c = |u| (R_xx / R_crit)^p`` -- the front holds where the stress is at
the threshold, retreats where crevasses would penetrate fully and
advances where they would not.  Bassis et al. (2026, "Nye was right!")
argue from numerical experiments that the HFB depths hold for widely
spaced crevasses only; the two variants are provided so that question can
be asked.  Meltwater-filled surface crevasses (Coffey & Lai's MS
configurations, threshold reduced by their ``B*``) are not implemented.

Minimum thickness (CalvingMIP experiment 5)
-------------------------------------------
``c = max(0, 1 + (Hc - H) / Hc) |u|``: at ``H = Hc`` the rate is exactly
the speed the ice arrives with, so ``Hc`` is the thickness the front
settles at, not a cutoff.  The level set anchors on every ice/no-ice
facet, so on an ice sheet the rule would also erode land-terminating
margins and nunataks thinner than ``Hc`` and book the loss as calving;
by default it acts only where the bed is below sea level, which still
calves a marine grounded cliff.

Units: MPa, m, yr (icepack conventions).
"""

import importlib.util
import numbers
from pathlib import Path

from firedrake import (Constant, Identity, conditional, dot, grad, gt, lt,
                       max_value, min_value, sqrt, sym, tr)

from .constants import ice_density, water_density, gravity, year
from .grounding import height_above_flotation

__all__ = [
    "LAWS", "IV_CONVENTIONS", "FrontState", "Law", "register", "make",
    "parse_params", "load_module", "density", "densities",
    "deviatoric_stress", "principal_values", "tensile_von_mises_stress",
    "tensile_von_mises_strain", "speed", "front_ice_speed",
    "resistive_stress", "hfb_critical_stress",
]

EPS = 1e-30


def density(rho_si):
    r"""A density quoted in kg/m3, which is how the papers quote it, in
    icepack's MPa, m, yr units: ``density(1024)`` is icepack2's seawater."""
    return float(rho_si) / year ** 2 * 1e-6


def densities(state):
    r"""``(rho_i, rho_w)`` in model units: the state's own if it names them,
    else icepack2's."""
    return (float(getattr(state, "rho_i", ice_density)),
            float(getattr(state, "rho_w", water_density)))


# ── the state a law reads ───────────────────────────────────────────────
class FrontState:
    r"""A front state (see the module docstring) built from a dual solution.

    Parameters
    ----------
    u, M, tau : Function
        The dual solution's velocity, membrane stress and basal stress.
    h : Function (DG0)
        The transport thickness.
    b : Function or UFL
        Bed elevation.
    levelset : :class:`icepack_tools.levelset.LevelSet`
        The front; ``nfront`` is its unit gradient and ``front_len`` its
        per-cell front length, both refreshed by the level set itself.
    A, n : Function or float, float
        Fluidity and Glen exponent, for the strain-rate von Mises form.
    rho_i, rho_w : float
        The densities of the host model's flotation test, in model units.

    ``haf`` and ``chi_gr`` are UFL on the cells, so they follow ``h`` and
    ``b`` without an update.
    """

    def __init__(self, u, M, tau, h, b, levelset, A=None, n=None,
                 rho_i=ice_density, rho_w=water_density):
        self.u, self.M, self.tau = u, M, tau
        self.h, self.b = h, b
        self.A, self.n = A, n
        self.rho_i, self.rho_w = float(rho_i), float(rho_w)
        self.haf = height_above_flotation(h, b, rho_I=self.rho_i,
                                          rho_W=self.rho_w)
        self.chi_gr = conditional(gt(self.haf, Constant(0.0)),
                                  Constant(1.0), Constant(0.0))
        self.levelset = levelset
        self.nfront = levelset.ghat
        self.front_len = levelset.front_len
        self.Q0 = levelset.Q0

    @classmethod
    def from_dual(cls, z, h, b, levelset, **kwargs):
        r"""The state of a dual solution ``z = (u, M, tau)``."""
        u, M, tau = z.subfunctions
        return cls(u, M, tau, h, b, levelset, **kwargs)


# ── stress algebra on the dual state ─────────────────────────────────────
def deviatoric_stress(M):
    r"""Horizontal deviatoric stress ``tau_h = M - tr(M)/3 I`` (MPa)."""
    return M - tr(M) / 3 * Identity(2)


def principal_values(T):
    r"""Eigenvalues of a symmetric 2x2 tensor, larger first."""
    mean = tr(T) / 2
    rad = sqrt(((T[0, 0] - T[1, 1]) / 2) ** 2 + T[0, 1] ** 2 + Constant(EPS))
    return mean + rad, mean - rad


def tensile_von_mises_stress(M):
    r"""``sqrt(3) tau~``, ``tau~^2 = (tau1+^2 + tau2+^2)/2`` from ``M`` (MPa)."""
    t1, t2 = principal_values(deviatoric_stress(M))
    t1p, t2p = max_value(t1, Constant(0.0)), max_value(t2, Constant(0.0))
    return sqrt(Constant(1.5) * (t1p ** 2 + t2p ** 2) + Constant(EPS))


def _ufl(value):
    return Constant(float(value)) if isinstance(value, numbers.Real) else value


def tensile_von_mises_strain(u, A, n):
    r"""Morlighem's ``sqrt(3) B eps~^(1/n)`` from the velocity (MPa), with
    ``B = A^(-1/n)``; ``A`` may be a number or a field."""
    e1, e2 = principal_values(sym(grad(u)))
    e1p, e2p = max_value(e1, Constant(0.0)), max_value(e2, Constant(0.0))
    e_tilde = sqrt((e1p ** 2 + e2p ** 2) / 2 + Constant(EPS))
    B = _ufl(A) ** (-1.0 / float(n))
    return Constant(3.0 ** 0.5) * B * e_tilde ** (1.0 / float(n))


def speed(u):
    return sqrt(dot(u, u) + Constant(1e-12))


# ── the registry ─────────────────────────────────────────────────────────
#: name -> Law subclass.  Filled by :func:`register`; a driver chooses from
#: it and :func:`load_module` can add to it at run time.
LAWS = {}

#: The two readings of "the ice velocity at the calving front", ``Iv``.
IV_CONVENTIONS = ("normal", "speed")


def front_ice_speed(model, iv):
    r"""``Iv`` for a rate ``Cr = Iv - Wv``: ``"normal"`` is ``u . n``,
    ``"speed"`` is ``|u|`` (the ISSM-based CalvingMIP submissions).
    Anything else is an error, not a silent fall-through to one of them."""
    if iv == "normal":
        return dot(model.u, model.nfront)
    if iv == "speed":
        return speed(model.u)
    raise ValueError(f"iv must be one of {IV_CONVENTIONS}, got {iv!r}")


def register(cls):
    r"""Class decorator: make a :class:`Law` selectable by its ``name``."""
    if not cls.name or cls.name in LAWS:
        raise ValueError(f"law name {cls.name!r} missing or already registered")
    LAWS[cls.name] = cls
    return cls


def _format(value):
    r"""A parameter as provenance records it: a float at full precision, so
    two runs that differ in the seventh digit do not read the same."""
    return repr(value) if isinstance(value, float) else str(value)


class Law:
    r"""Base class for a calving law.

    Subclass, set ``name`` and the ``defaults`` of your parameters, decorate
    with :func:`register`, and implement :meth:`rate`.  Everything else is
    optional:

    ``rate(model, t)``
        UFL calving rate on ice [m/yr of retreat along the front normal].
        Read any of the state fields listed in the module docstring;
        ``self.p`` holds the parameters.
    ``check()``
        Raise ``ValueError`` for a parameter value the law cannot run with,
        so a mistyped configuration fails when the law is made, not at the
        first step of a run.
    ``fields(model)``
        ``{name: UFL}`` cell-wise diagnostics a run may save (a stress, the
        rate itself, a damage field).
    ``tracers(model)``
        ``{name: (initial, source)}`` prognostic DG0 fields the model must
        carry with the ice: ``initial`` a float or UFL expression,
        ``source(model) -> UFL`` its rate of change on ice per year (or
        ``None``).  A model that supports tracers advects each one
        conservatively as ``h q`` with the thickness transport's flux.

    ``front_mode`` is the :class:`~icepack_tools.levelset.LevelSet` law the
    front runs under: ``"prescribed"`` (the level set retreats at
    :meth:`rate`) for every law but :class:`FixedFront`.

    Parameters arrive as keyword arguments, are checked against
    ``defaults`` so a typo fails loudly, and are exposed as ``self.p[key]``.
    ``grounded_gate`` zeroes the rate on grounded cells; apply it through
    :meth:`gated`.
    """
    name = ""
    #: parameter name -> default value; anything else is refused
    defaults = {}
    #: zero the rate on grounded cells
    grounded_gate = False
    #: does the rate depend on time explicitly
    time_dependent = False
    #: the LevelSet law the front runs under
    front_mode = "prescribed"

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}; "
                             f"it takes {sorted(self.defaults)}")
        self.p = dict(self.defaults, **params)
        self.check()

    @property
    def params(self):
        return dict(self.p)

    def check(self):
        pass

    def rate(self, model, t):
        raise NotImplementedError

    def gated(self, model, c):
        return c * (Constant(1.0) - model.chi_gr) if self.grounded_gate else c

    def fields(self, model):
        return {}

    def tracers(self, model):
        return {}

    def describe(self):
        p = ", ".join(f"{k}={_format(v)}" for k, v in self.p.items())
        return f"{self.name}({p})" + (" [grounded ice does not calve]" if self.grounded_gate else "")


@register
class FixedFront(Law):
    r"""The front is frozen at its initial position: no calving, no advance."""
    name = "fixed"
    front_mode = "fixed"

    def rate(self, model, t):
        return None


@register
class VelocityRate(Law):
    r"""``Cr = Iv``: the rate that holds a front stationary.

    The ISSM-based CalvingMIP submissions hold a front the same way but
    with ``Cr = |u|`` directed along the *velocity*, so that
    ``w = u - |u| u_hat`` vanishes identically.  Both give an exactly
    stationary front; ``iv="normal"`` is the same statement resolved on the
    front normal."""
    name = "velocity"
    defaults = {"iv": "normal"}

    def check(self):
        if self.p["iv"] not in IV_CONVENTIONS:
            raise ValueError(f"iv must be one of {IV_CONVENTIONS}, got {self.p['iv']!r}")

    def rate(self, model, t):
        return self.gated(model, front_ice_speed(model, self.p["iv"]))


@register
class ThicknessLaw(Law):
    r"""``Cr = max(0, 1 + (Hc - H)/Hc) |u|`` (CalvingMIP experiment 5).

    ``hc`` [m] is where the front settles; 375 m is the value in the
    CalvingMIP protocol's example results, which the protocol does not fix.
    ``marine_only`` (default on) confines the rule to cells whose bed is
    below sea level, so a land margin or the ice ringing a nunatak never
    calves while a marine grounded cliff still does."""
    name = "thickness"
    defaults = {"hc": 375.0, "marine_only": True}

    def check(self):
        if not float(self.p["hc"]) > 0.0:
            raise ValueError(f"thickness: hc must be a positive thickness, got {self.p['hc']!r}")

    def rate(self, model, t):
        hc = Constant(float(self.p["hc"]))
        thickness = max_value(model.h, Constant(0.0))
        c = max_value(Constant(1.0) + (hc - thickness) / hc, Constant(0.0)) * speed(model.u)
        if self.p["marine_only"]:
            c = conditional(lt(model.b, Constant(0.0)), c, Constant(0.0))
        return self.gated(model, c)

    def fields(self, model):
        return {"calving_rate": self.rate(model, 0.0)}


@register
class VonMises(Law):
    r"""``c = |u| sigma~ / sigma_max`` with ``sigma~`` from the dual membrane
    stress (see the module docstring); ``sigma_max_gr`` on grounded cells,
    ``sigma_max_fl`` on floating ones [MPa]."""
    name = "vonmises"
    defaults = {"sigma_max_gr": 1.0, "sigma_max_fl": 0.15}

    def check(self):
        for key in ("sigma_max_gr", "sigma_max_fl"):
            if not float(self.p[key]) > 0.0:
                raise ValueError(f"{self.name}: {key} must be positive, got {self.p[key]!r}")

    def sigma_tilde(self, model):
        return tensile_von_mises_stress(model.M)

    def sigma_max(self, model):
        return conditional(gt(model.haf, Constant(0.0)),
                           Constant(float(self.p["sigma_max_gr"])),
                           Constant(float(self.p["sigma_max_fl"])))

    def rate(self, model, t):
        c = speed(model.u) * self.sigma_tilde(model) / self.sigma_max(model)
        return self.gated(model, c)

    def fields(self, model):
        return {"sigma_vm": self.sigma_tilde(model), "calving_rate": self.rate(model, 0.0)}


@register
class VonMisesStrain(VonMises):
    r"""Morlighem's strain-rate form, ``sigma~ = sqrt(3) B eps~^(1/n)``, with
    the state's own fluidity ``A`` and exponent ``n``: what ISSM, Ua and
    Kori run."""
    name = "vonmises_strain"

    def sigma_tilde(self, model):
        A, n = getattr(model, "A", None), getattr(model, "n", None)
        if A is None or n is None:
            raise ValueError(
                "vonmises_strain reads the fluidity A and the exponent n off "
                "the state, and this state has none; build it with A= and n=")
        return tensile_von_mises_strain(model.u, A, n)


def resistive_stress(model, which="normal"):
    r"""Depth-averaged resistive stress ``R_xx`` [MPa] for a front law.

    ``"normal"`` (the default) is ``n . M n`` with ``n`` the outward front
    normal, and it is the quantity the horizontal-force-balance papers
    mean.  They write the resistive stress as ``R_xx = 2 tau_xx + tau_yy``
    with ``x`` along flow, normal to the front; icepack2's membrane stress
    is ``M = tau_h + tr(tau_h) I``, so in front-aligned coordinates
    ``M_nn = 2 tau_nn + tau_tt`` is exactly their ``R_xx``.

    ``"principal"`` takes the largest principal value of ``M`` instead.
    That is never smaller, and on the CalvingMIP Thule steady state it is
    9 % larger at the front (149 against 136 kPa), which is enough to move
    the front from the critical buttressing the theory predicts (ratio
    0.98) to 9 % over it (1.09) and turn a front that should hold into one
    retreating at ~900 m/yr.  Kept for that comparison, not as a default.
    """
    if which == "normal":
        n = model.nfront
        return dot(n, dot(model.M, n))
    if which == "principal":
        return principal_values(model.M)[0]
    raise ValueError(f"stress must be 'normal' or 'principal', got {which!r}")


#: The thresholds :func:`hfb_critical_stress` knows.
HFB_MODES = ("hfb", "zero_stress")


def hfb_critical_stress(h, haf, sigma_max=0.0, rho_c=water_density,
                        mode="hfb", rho_i=ice_density):
    r"""``R_crit`` [MPa] of the horizontal-force-balance calving criterion
    (Slater & Wagner 2025 eq. 21; Coffey & Lai 2025 ``B = 0`` when
    ``sigma_max = 0`` and the crevasse water is seawater), or the
    zero-stress one (``mode="zero_stress"``).  ``rho_c`` is the density of
    the water in a basal crevasse and ``rho_i`` the ice's, both in model
    units (:func:`density` converts from kg/m3)."""
    if mode not in HFB_MODES:
        raise ValueError(f"mode must be one of {HFB_MODES}, got {mode!r}")
    P = rho_i * gravity * max_value(h, Constant(1.0))          # rho_i g H
    a = max_value(haf, Constant(0.0)) / max_value(h, Constant(1.0))
    ri = Constant(float(rho_i / rho_c))
    if mode == "zero_stress":
        return P * (Constant(1.0) - ri * (Constant(1.0) - a))
    s_tilde = Constant(float(sigma_max)) / P
    strength = Constant(float(rho_c / (rho_c - rho_i))) * s_tilde ** 2 / 2
    return P * ((Constant(1.0) - ri * (Constant(1.0) - a) ** 2) / 2 + strength)


@register
class HorizontalForceBalance(Law):
    r"""``c = |u| (R_xx / R_crit)^exponent`` with the horizontal-force-balance
    threshold (see the module docstring).

    ``R_xx`` is the front-normal resistive stress by default, which is the
    papers' own quantity; see :func:`resistive_stress` for what the
    ``principal`` alternative costs.  ``sigma_max`` [MPa] is the ice
    tensile strength -- 0 is the Buck / Coffey & Lai limit, in which an
    unbuttressed shelf is exactly at the calving threshold and, as those
    authors note, no unbuttressed shelf should exist; Slater & Wagner's
    150 kPa with basal friction is the combination they find consistent
    with observed fronts.  ``rho_c`` is the basal-crevasse water density
    in kg/m3, ``"seawater"`` (the default) being the state's own ``rho_w``
    and 1000 a meltwater-filled crevasse, and ``mode`` selects ``hfb`` or
    the ``zero_stress`` (Nye) threshold, which is twice as large on a
    shelf.

    ``ratio_max`` caps ``R_xx / R_crit``: the threshold scales with ``H``,
    so on cells thinning to nothing the ratio is unbounded and one step
    could carry the front across many cells; ``ratio_max |u|`` is the most
    a step may remove.  Where the ice is thick it never binds.
    """
    name = "hfb"
    defaults = {"sigma_max": 0.0, "rho_c": "seawater", "mode": "hfb",
                "stress": "normal", "exponent": 1.0, "ratio_max": 5.0}

    def check(self):
        if self.p["mode"] not in HFB_MODES:
            raise ValueError(f"hfb: mode must be one of {HFB_MODES}, got {self.p['mode']!r}")
        if self.p["stress"] not in ("normal", "principal"):
            raise ValueError(f"hfb: stress must be 'normal' or 'principal', got {self.p['stress']!r}")
        if self.p["rho_c"] != "seawater" and not isinstance(self.p["rho_c"], numbers.Real):
            raise ValueError(f"hfb: rho_c must be 'seawater' or a density in kg/m3, "
                             f"got {self.p['rho_c']!r}")

    def rho_c(self, model):
        r"""The crevasse water density in model units."""
        if self.p["rho_c"] == "seawater":
            return densities(model)[1]
        return density(self.p["rho_c"])

    def r_xx(self, model):
        return max_value(resistive_stress(model, self.p["stress"]), Constant(0.0))

    def r_crit(self, model):
        return hfb_critical_stress(model.h, model.haf, float(self.p["sigma_max"]),
                                   self.rho_c(model), self.p["mode"],
                                   densities(model)[0])

    def ratio(self, model):
        return min_value(self.r_xx(model) / self.r_crit(model),
                         Constant(float(self.p["ratio_max"])))

    def rate(self, model, t):
        c = speed(model.u) * self.ratio(model) ** float(self.p["exponent"])
        return self.gated(model, c)

    def fields(self, model):
        ratio = self.ratio(model)
        # Coffey & Lai's buttressing number and HFB crevassed fraction
        # 1 - sqrt(B), meaningful for the dry-surface / seawater-basal case
        B = max_value(Constant(1.0) - ratio, Constant(0.0))
        return {"R_xx": self.r_xx(model), "R_crit": self.r_crit(model),
                "buttressing": B, "crevassed_fraction": Constant(1.0) - sqrt(B),
                "calving_rate": self.rate(model, 0.0)}


# ── making a law ─────────────────────────────────────────────────────────
def _parse_value(val):
    low = val.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(val) if val.strip().lstrip("-").isdigit() else float(val)
    except ValueError:
        return val


def parse_params(items):
    r"""``["hc=300", "sigma_max_fl=0.2", "mode=zero_stress"]`` -> a dict,
    numbers and ``true``/``false`` parsed, anything else kept as text."""
    out = {}
    for item in items or ():
        key, sep, val = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"law parameter {item!r} is not key=value")
        out[key] = _parse_value(val)
    return out


def load_module(path):
    r"""Import a Python file whose :func:`register` classes join
    :data:`LAWS`.  Returns the module."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make(name, **params):
    r"""Instantiate law ``name`` with ``params``, validated against the law's
    ``defaults`` and its :meth:`Law.check`."""
    if name not in LAWS:
        raise ValueError(f"unknown calving law {name!r}; registered: {sorted(LAWS)}")
    return LAWS[name](**params)
