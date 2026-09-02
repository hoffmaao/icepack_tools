r"""Density and gravity parameters thread through to flotation.

MISMIP+ and CalvingMIP prescribe 1028 kg/m^3 seawater against icepack2's
1024.  A consumer builds its hydrostatic surface with its own value, so the
grounded indicator, the effective pressure and the driving stress in the
dual residual must accept the same value or they place flotation a few
metres away from where the surface does.

    python -u density_test.py       (or pytest)
"""
import numpy as np
import firedrake as fd
from firedrake import Constant, Function, FunctionSpace, assemble, dx

from icepack_tools.constants import ice_density, water_density, gravity
from icepack_tools.grounding import grounded_mask, effective_pressure
from icepack_tools.momentum import dual_residual
from icepack_tools.spaces import dual_function_space

RHO_W_1028 = water_density * 1028.0 / 1024.0


def _scalar(expr, mesh):
    return float(assemble(expr * dx(domain=mesh)))


def test_grounded_mask_takes_densities():
    mesh = fd.UnitIntervalMesh(1)
    b = Constant(-1000.0)
    # exactly at flotation for 1028: haf = 0, He = 1/2
    H = Constant(1028.0 / 917.0 * 1000.0)
    he_1028 = _scalar(grounded_mask(H, b, rho_W=RHO_W_1028), mesh)
    he_1024 = _scalar(grounded_mask(H, b), mesh)
    assert abs(he_1028 - 0.5) < 1e-12
    assert he_1024 > 0.55, he_1024      # 1024 calls the same column grounded


def test_effective_pressure_vanishes_at_its_own_flotation():
    mesh = fd.UnitIntervalMesh(1)
    H = Constant(1000.0)
    rr = 917.0 / 1028.0
    s = (1 - rr) * H                     # hydrostatic surface for 1028
    N_1028 = _scalar(effective_pressure(H, s, rho_W=RHO_W_1028), mesh)
    N_1024 = _scalar(effective_pressure(H, s), mesh)
    assert abs(N_1028) < 1e-9 * float(ice_density * gravity * 1000.0)
    assert N_1024 > 1e-3 * float(ice_density * gravity * 1000.0)


def test_dual_residual_accepts_densities():
    mesh = fd.UnitSquareMesh(4, 4)
    mesh.coordinates.dat.data[:] *= 20e3
    Q0 = FunctionSpace(mesh, "DG", 0)
    Q1 = FunctionSpace(mesh, "CG", 1)
    x = fd.SpatialCoordinate(mesh)
    b = Function(Q0).interpolate(Constant(-600.0) + x[0] / 20.0)
    H = Function(Q0).interpolate(Constant(500.0))
    rr = 917.0 / 1028.0
    s = Function(Q0).interpolate(fd.max_value(b + H, Constant(1 - rr) * H))
    Z = dual_function_space(mesh, 1)
    z = Function(Z)
    z.subfunctions[0].interpolate(fd.as_vector((Constant(100.0), Constant(0.0))))
    theta, phi = Function(Q1), Function(Q1)
    F = dual_residual(z, theta, phi, H=H, s=s, b=b, h_layers=[H], C_w0=Constant(0.01),
                      A_layers=[Constant(2.94)], n_consts=[Constant(3.0)], n_vals=[3.0],
                      m_slide=3.0, mesh=mesh, law="budd",
                      rho_I=ice_density, rho_W=RHO_W_1028, g=gravity)
    r = assemble(F)
    with r.dat.vec_ro as v:
        assert np.isfinite(v.norm()) and v.norm() > 0
    # the same residual with the default seawater density differs: the
    # densities are actually used, not swallowed
    F0 = dual_residual(z, theta, phi, H=H, s=s, b=b, h_layers=[H], C_w0=Constant(0.01),
                       A_layers=[Constant(2.94)], n_consts=[Constant(3.0)], n_vals=[3.0],
                       m_slide=3.0, mesh=mesh, law="budd")
    r0 = assemble(F0)
    with r.dat.vec_ro as v, r0.dat.vec_ro as v0:
        assert (v - v0).norm() > 1e-12 * v.norm()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("PASS: densities")
