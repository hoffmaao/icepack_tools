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
spaces      mixed dual function spaces, L >= 1 (L = 1 is single-layer)
geometry    CG1 lifting and a slope that works for CG1 or DG0 surfaces
grounding   height above flotation, smooth grounded mask, effective pressure
friction    Weertman/Budd/regularised-Coulomb basal stress, closed as
            a residual
viscosity   composite membrane and interlayer closures
momentum    per-layer momentum balance and full single/multi-layer residuals
parallel    rank-independent field statistics and control-vector gathers
"""

from . import (constants, geometry, grounding, spaces, friction,  # noqa: F401
               viscosity, momentum, parallel)

__all__ = [
    "constants", "geometry", "grounding", "spaces", "friction", "viscosity",
    "momentum", "parallel",
]
