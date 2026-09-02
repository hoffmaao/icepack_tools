r"""Grounding-line quantities for the icepack2 dual form.

Everything here is built from the *model* thickness and surface, so the
grounded/floating transition is the hydrostatic flotation criterion itself
rather than a mask carried alongside the state.  That is what lets the
effective-pressure-capped friction laws in :mod:`icepack_tools.friction`
(``budd`` and ``regularized_coulomb``) give *exactly* zero basal drag on
floating ice.

Ported from ``ismip7/icepack2_tools/grounding.py``.
"""

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
