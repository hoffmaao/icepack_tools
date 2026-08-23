r"""Reusable building blocks for icepack2 dual-form ice-flow models.

Everything here is written in **residual** form rather than as a
dissipation potential, because the dual system's stress blocks are
rank-deficient at zero stress under a power-law potential and singular at
the Coulomb limit under a regularised-Coulomb one.  Closing the stresses
directly restores a well-conditioned Jacobian and removes the need to
continue in the sliding exponent.

Modules
-------
constants   physical constants, re-exported from icepack2
geometry    CG1 lifting and a slope that works for CG1 or DG0 surfaces
grounding   height above flotation, smooth grounded mask, effective pressure
friction    regularised-Coulomb basal stress and its residual closure
viscosity   composite membrane and interlayer closures
momentum    per-layer momentum balance and full single/multi-layer residuals
"""

from . import constants, geometry, grounding, friction, viscosity, momentum  # noqa: F401

__all__ = [
    "constants", "geometry", "grounding", "friction", "viscosity", "momentum",
]
