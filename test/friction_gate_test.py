r"""The Budd shelf gate must not be a sign test on roundoff.

On a floating cell the surface IS the flotation branch, so the effective
pressure ``N = max(p_I - p_W, 0)`` is a roundoff residue of either sign. The
old gate ``conditional(N > 0, N_hat, 0)`` let a positive residue through, and
with ``N_ref`` equally tiny the PISM floor ``nhat_floor * p_I / N_ref`` lifted
it to ``nhat_cap``: triple Weertman friction on 418 of 3791 floating cells of
the ISMIP7 32 km MAP, flipped wholesale by a 2e-13 m change in the surface.
Gating on ``He`` alone was not enough either: ``He`` is smooth in height
above flotation, so a cell floating by a few metres sits inside the He band
and still got ``He * nhat_cap``. The gate is now HAF > 0 itself.

These cases give ``N`` a POSITIVE roundoff-scale residue on purpose (the
surface 1e-9 m above flotation) and require exactly zero drag:
  * a cell floating by 600 m (far shelf)
  * a cell floating by 3 m (inside the He band, He ~ 0.35)
and a grounded cell 50 m above flotation must still carry drag.

    python -m pytest -q friction_gate_test.py
"""
import numpy as np
from firedrake import (Constant, Function, FunctionSpace, RectangleMesh,
                       VectorFunctionSpace, as_vector)

from icepack_tools.friction import basal_stress, ice_density as rho_I, water_density as rho_W
from icepack_tools.grounding import effective_pressure


def _tau(H_val, haf, s_offset, grounded=False):
    mesh = RectangleMesh(4, 4, 1e4, 1e4)
    Q = FunctionSpace(mesh, "DG", 0)
    V = VectorFunctionSpace(mesh, "CG", 1)
    H = Function(Q).assign(H_val)
    b = Function(Q).assign(-(H_val - haf) * float(rho_I) / float(rho_W))   # bed at the requested HAF
    s_val = (b.dat.data_ro[0] + H_val) if grounded else H_val * (1.0 - float(rho_I) / float(rho_W))
    s = Function(Q).assign(s_val + s_offset)
    u = Function(V).interpolate(as_vector((100.0, 0.0)))
    C_w0 = Function(Q).assign(0.1)
    theta = Function(Q).assign(0.0)
    # the PISM floor is only accepted against a frozen N_ref; freeze it at
    # this geometry, exactly what a forward does at t=0 (on the floating
    # cells N_ref is then the same roundoff residue as N)
    N_ref = Function(Q).interpolate(effective_pressure(H, s))
    tau = basal_stress(u, C_w0, theta, H, s, b, Constant(3.0), law="budd",
                       N_ref=N_ref, nhat_floor=0.02, nhat_cap=3.0)
    return Function(Q).interpolate(tau).dat.data_ro


def test_far_shelf_with_positive_roundoff_has_no_drag():
    tau = _tau(500.0, haf=-600.0, s_offset=1e-9)
    assert np.all(tau == 0.0), tau.max()


def test_cell_floating_by_metres_inside_the_He_band_has_no_drag():
    tau = _tau(500.0, haf=-3.0, s_offset=1e-9)
    assert np.all(tau == 0.0), tau.max()


def test_grounded_cell_keeps_its_drag():
    tau = _tau(500.0, haf=50.0, s_offset=0.0, grounded=True)
    assert np.all(tau > 0.0), tau.min()
