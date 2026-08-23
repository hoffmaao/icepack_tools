r"""Is the diffusion-creep term wired into the layers it claims to be?

``multilayer_rc_test.py`` shows what diffusion creep *buys* -- a cold,
direct solve with no continuation.  This one checks the plumbing that
claim rests on, with assembly only and no solves:

  * ``A_lin_layers=None`` (the default) is byte-identical to spelling out
    ``[None, ...]``, so callers written before diffusion creep existed
    get exactly the residual they got before,
  * a wrong-length list is rejected rather than silently truncated,
  * layer ``l``'s membrane closure takes ``A_lin_layers[l]`` and the
    closure on interface ``l`` takes ``A_lin_layers[l - 1]`` -- the layer
    *below* -- and no other block moves.  This is what distinguishes a
    genuinely per-layer prefactor from one broadcast to every layer,
  * the diffusion contribution does not move with the log-fluidity
    ``phi``: it is prescribed physics, deliberately outside the control.

    python -u diffusion_creep_test.py
"""
import numpy as np
import firedrake
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace, SpatialCoordinate,
    as_vector, assemble, max_value,
)

from icepack_tools.constants import ice_density as rho_I, water_density as rho_W
from icepack_tools.friction import weertman_anchor
from icepack_tools.momentum import multilayer_rc_residual
from icepack_tools.viscosity import A_DIFFUSION

FRACTIONS = [0.15, 0.85]
N_VALS = [4.0, 1.8]
A_VALS = [46.0, 0.45]
M_SLIDE = 3.0
FC = {"quadrature_degree": 4}

# block index of each unknown: layer l is (u, M, S) at 3l, 3l+1, 3l+2
BLOCKS = {0: "u0", 1: "M0", 2: "tau_b", 3: "u1", 4: "M1", 5: "S1"}


def build(nx=8, Lx=40e3, Ly=12e3):
    mesh = firedrake.RectangleMesh(nx, 4, Lx, Ly, diagonal="crossed")
    x, y = SpatialCoordinate(mesh)
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    H = Function(Q).interpolate(Constant(900.0) - Constant(840.0) * x / Constant(Lx))
    b = Function(Q).interpolate(Constant(100.0) - Constant(800.0) * x / Constant(Lx))
    s = Function(Q).interpolate(
        max_value(b + H, Constant(1.0 - rho_I / rho_W) * H))
    u_obs = Function(V).interpolate(
        as_vector([Constant(80.0) + Constant(900.0) * x / Constant(Lx),
                   Constant(0.0)]))
    return mesh, Q, V, H, b, s, u_obs


def residual(A_lin_layers, phi_val=0.0, omit=False):
    """Assemble the multilayer residual at a fixed, non-zero state."""
    from multilayer.model.utilities import create_function_space, layer_thicknesses
    mesh, Q, V, H, b, s, u_obs = build()
    Z = create_function_space(mesh, len(FRACTIONS))
    h_layers = layer_thicknesses(H, len(FRACTIONS), fractions=FRACTIONS)
    C_w0 = weertman_anchor(H, s, u_obs, M_SLIDE, Q)
    theta = Function(Q)
    phi = Function(Q).interpolate(Constant(phi_val))

    # A non-zero state, so every |M|^(n-1) and |S|^(n-1) is exercised away
    # from the origin and the diffusion term has something to act on.  The
    # stresses are symmetric-tensor / vector DG0, which does not take an
    # interpolated UFL expression, so their dofs are set directly; the seed
    # is fixed so every call below assembles at the *same* state.
    x, y = SpatialCoordinate(mesh)
    rng = np.random.default_rng(0)
    z = Function(Z)
    for l in range(len(FRACTIONS)):
        z.subfunctions[3 * l].interpolate(
            as_vector([Constant(100.0 + 40.0 * l) + Constant(300.0) * x / Constant(40e3),
                       Constant(5.0) * y / Constant(12e3)]))
        for i in (3 * l + 1, 3 * l + 2):
            dat = z.subfunctions[i].dat
            dat.data[:] = 0.05 + 0.02 * rng.standard_normal(dat.data.shape)

    kwargs = dict(
        H=H, s=s, b=b, h_layers=h_layers, C_w0=C_w0,
        A_layers=[Constant(A_VALS[0]), Constant(A_VALS[1]) * firedrake.exp(phi)],
        n_consts=[Constant(v) for v in N_VALS], n_vals=N_VALS,
        m_slide=M_SLIDE, mesh=mesh, layer_fractions=FRACTIONS,
        h_visc_floor=10.0, c_w0_floor=1e-6,
    )
    if not omit:
        kwargs["A_lin_layers"] = A_lin_layers
    F = multilayer_rc_residual(z, theta, phi, **kwargs)
    return [assemble(F, form_compiler_parameters=FC)
            .subfunctions[i].dat.data_ro.copy()
            for i in range(len(BLOCKS))]


def moved(a, bb):
    """Which blocks changed, and by how much relative to the block norm."""
    out = {}
    for i in range(len(BLOCKS)):
        num = np.abs(a[i] - bb[i]).max()
        den = max(np.abs(a[i]).max(), 1e-300)
        out[BLOCKS[i]] = num / den
    return out


def main():
    A = A_DIFFUSION
    print("Diffusion-creep plumbing, two layers (n = 4 bottom / 1.8 top).")
    print("Blocks per layer l: u = 3l, M = 3l+1, S = 3l+2.\n")

    base_omitted = residual(None, omit=True)
    base_none = residual(None)
    base_list = residual([None, None])

    for tag, other in (("A_lin_layers=None", base_none),
                       ("A_lin_layers=[None, None]", base_list)):
        d = max(np.abs(base_omitted[i] - other[i]).max()
                for i in range(len(BLOCKS)))
        print(f"  omitted vs {tag:<26} max block diff {d:.3e}")
        assert d == 0.0, f"{tag} is not identical to omitting the argument"
    print("  -> pre-diffusion callers are bit-for-bit unaffected\n")

    for bad in ([Constant(A)], [Constant(A)] * 3):
        try:
            residual(bad)
        except ValueError as exc:
            print(f"  {len(bad)} entries for 2 layers -> ValueError: {exc}")
        else:
            raise AssertionError(f"{len(bad)} entries should have been rejected")
    print()

    lower_only = residual([Constant(A), None])
    upper_only = residual([None, Constant(A)])

    print("  A_lin on the LOWER layer only (index 0):")
    d0 = moved(base_omitted, lower_only)
    for k, v in d0.items():
        print(f"    {k:<6} {v:.3e}" + ("   <- moved" if v > 1e-12 else ""))
    print("  A_lin on the UPPER layer only (index 1):")
    d1 = moved(base_omitted, upper_only)
    for k, v in d1.items():
        print(f"    {k:<6} {v:.3e}" + ("   <- moved" if v > 1e-12 else ""))

    # index 0 -> layer 0's membrane block, and interface 1 (the layer BELOW
    # the interface); index 1 -> layer 1's membrane block and nothing else
    assert {k for k, v in d0.items() if v > 1e-12} == {"M0", "S1"}, d0
    assert {k for k, v in d1.items() if v > 1e-12} == {"M1"}, d1
    print("  -> per-layer, with the documented interface convention\n")

    # not aliased: one Constant per layer, distinct effects
    assert np.abs(lower_only[1] - upper_only[1]).max() > 0.0
    assert np.abs(lower_only[4] - upper_only[4]).max() > 0.0

    # phi scales A (dislocation creep) but must not touch diffusion creep.
    # The contribution is a difference of two residuals that both carry the
    # (large, phi-scaled) creep term, so it cancels to roundoff rather than
    # to zero -- hence 1e-10 and not an exact comparison.
    worst = 0.0
    for phi_val in (0.0, 0.5):
        off = residual([None, None], phi_val=phi_val)
        on = residual([None, Constant(A)], phi_val=phi_val)
        contrib = on[4] - off[4]                       # layer 1 membrane block
        if phi_val == 0.0:
            ref = contrib
        rel = np.abs(contrib - ref).max() / max(np.abs(ref).max(), 1e-300)
        worst = max(worst, rel)
        print(f"  phi = {phi_val}: |diffusion contribution - reference| "
              f"= {rel:.3e} (relative)")
    assert worst < 1e-10, "diffusion creep moved with the log-fluidity control"
    # and phi really does move the dislocation part, so the check has teeth
    moved_by_phi = np.abs(residual([None, None], phi_val=0.5)[4]
                          - base_omitted[4]).max()
    assert moved_by_phi > 0.0, "phi did not change the creep term at all"
    print(f"  (phi does move the creep term: {moved_by_phi:.3e})")

    print("\nPASS: default off, length checked, per-layer as documented,")
    print("      and diffusion creep is outside the phi control")


if __name__ == "__main__":
    main()
