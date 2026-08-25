# icepack_tools

Residual-form building blocks for [icepack2](https://github.com/icepack/icepack2)
dual ice-flow models, single- or multi-layer.

## Why residual form

The dual formulation carries stress as an unknown alongside velocity.
Closing a stress through a **dissipation potential** -- the usual
`icepack2.model.minimization.friction_power` and `viscous_power` -- gives
a Jacobian block that scales like `|tau|^(m-1)` or `|M|^(n-1)`, and is
therefore **rank-deficient at zero stress** for any exponent above 1.  A
cold start has no descent direction, which is why potential-form dual
models need a continuation that ramps every exponent up from 1.

Regularised Coulomb is worse.  Its complementary energy is

```
P = integral  u_0 beta^2 / (m+1) * [-ln(1 - r^(m+1))] dx,    r = |tau| / beta^2
```

which is **singular at the Coulomb limit** `r -> 1`.  For a fast tidewater
glacier that limit is the operating point, not a corner case: with
`u_0 = 300` m/yr a trunk moving 2000 m/yr sits at `r ~ 0.97`, and Newton
cannot reach it from `tau = 0` at all.  (Observed on Store Gletscher: a
potential-form continuation diverges at the very first `n = m = 1` solve.)

Closing the stress directly instead,

```
tau_W   = C_w0 exp(theta He) |u|_reg^(1/m)          Weertman branch
tau_cap = c0 N                                       Coulomb cap
tau_b   = tau_W tau_cap / (tau_W + tau_cap)          harmonic blend
F      += inner(tau + tau_b u / |u|_reg, sigma) dx   identity tau-block
```

makes that block the identity -- perfectly conditioned at `tau = 0` -- and
the blend is smooth everywhere, so **no continuation in `m` is needed**.

The flow-law exponents `n` need one *unless* the composite carries a real
linear mechanism.  At `M = 0` the creep term contributes nothing, so if
the only other term is the small `alpha` regulariser (prefactor
`A_reg = A tau_c^(n-1)`, at a *constant* reference thickness) the
linearised viscosity is absurd and Newton diverges.  Adding **diffusion
creep** (`n = 1`, `A_lin_layers ~ 1e-3` MPa⁻¹ yr⁻¹, one entry per layer)
in parallel with dislocation creep fixes it -- and it is a physical
mechanism, not a numerical device: Goldsby & Kohlstedt's composite makes
Glen's `n = 3` an effective average of several mechanisms, and the
Thwaites multilayer runs carry this same term.  It is per-layer because
its prefactor depends on temperature and grain size, it carries the
*layer* thickness so it vanishes with the ice, and it is deliberately
**not** scaled by the inverted log-fluidity `phi` -- diffusion creep is
prescribed physics, and `phi` controls only the dislocation-creep
component.  The honest consequence: at low deviatoric stress diffusion
carries a large share of the effective fluidity and `phi` cannot adjust
that share.  `dual_residual`'s `A_lin_layers` docstring gives that share
layer by layer and stress by stress.

`test/multilayer_rc_test.py` isolates what each ingredient buys.  Two-layer
`n = 4 / 1.8` composite on a Coulomb bed, cold start `z = 0`, `m = 3`
fixed at its target in every case:

| diffusion (n=1) | n | result | max speed |
|---|---|---|---|
| off | ramped 1 -> n | converged, 42 its | 1147.3 m/yr |
| off | **direct** | **diverges** | -- |
| 1e-3 | ramped 1 -> n | converged, 41 its | 1151.0 m/yr |
| 1e-3 | **direct** | **converged, 47 its** | 1151.0 m/yr |

So: the residual friction closure removes the `m`-continuation, and
diffusion creep removes the `n`-continuation.  Together **the whole
continuation apparatus disappears** -- one cold solve replaces a staged
ramp.  The ramped and direct paths agree to 1.98e-16, so the direct solve
is not converging somewhere else.

The test also checks the headline property directly: on the 280 cells
lying 100 m or more below flotation, `|tau_b|` is below 1e-12 kPa, i.e.
below 1e-14 of the grounded maximum -- machine zero, against the ~1 % of
grounded drag that a `phi_eff` floor of 0.01 leaves on every shelf node.
Floating cells are picked out by height above flotation rather than by
`N <= 0`: `N` is the cancelling difference `p_I - p_W`, so on a shelf it
is a roundoff residue rather than 0, and an `N <= 0` mask would drop
exactly the cells a law gated on `N > 0` still acts on.

Afloat, `tau_b` is *analytically* zero rather than small: `N` is clamped
to `max(p_I - p_W, 0)`, so `tau_cap = max(c0 N, eps_tauc)` is exactly 0
at the default `eps_tauc = 0`, and the blend `tau_W tau_cap /
max(tau_W + tau_cap, 1e-15)` is then 0 whatever `tau_W` is -- the point
of `N` entering as a *factor* rather than through a conditional, since
there is no threshold to sit on the wrong side of.  What the solve
returns is therefore roundoff, not residual physics: `tau` is a
solved-for unknown, so the factorisation reaches that zero only to the
precision of the system it sits in.  Those digits belong to the linear
solve and move whenever anything upstream perturbs it -- this branch's
anchor change among them -- so machine-zero drag is quoted here and in
the friction-law table below as a *bound* rather than as a measured
value.  The bound is the property actually being asserted, and it holds
for any anchor for the reason above: `N` enters `tau_b` as a
multiplicative factor and is exactly zero afloat.  Every bound quoted is
the one its test asserts (`SHELF_DRAG_MAX_KPA` and `SHELF_DRAG_MAX_REL`
in `multilayer_rc_test.py` and `dual_forms_test.py`), so a drift that
falsifies this readme fails a test rather than going unnoticed; each
carries two to five decades of headroom over what the solves currently
print, sized to how far that law's residue has actually been seen to
move.  The cell count is not a bound but an exact number -- it is pure
geometry, how many cells lie 100 m or more below flotation.

## What this buys

- **Exactly zero drag on floating ice.**  `N = max(p_I - p_W, 0)` is built
  from the *model* surface, so it vanishes at precisely the hydrostatic
  flotation criterion, and `budd` additionally gates on the grounded
  indicator `He` so that the roundoff residue of that cancelling
  difference cannot be amplified back into shelf drag.  No `phi_eff`
  floor, so no residual shelf drag.
- **Grounded-only friction inference.**  `theta` is carried as
  `exp(theta * He)` with `He` a smooth grounded indicator, so `dJ/dtheta`
  is identically zero afloat: an optimiser physically cannot place basal
  friction on a shelf.
- **A balanced starting control.**  The anchor `C_w0 = tau_d / |u_obs|^(1/m)`
  makes the Weertman branch return the *lifted* `tau_d` at
  `u = u_obs, theta = 0` -- the patch-averaged driving stress rather than
  the pointwise one, since `grad(s)` is discontinuous and cannot be
  interpolated into a continuous space without the answer following the
  mesh partition.  The balance is therefore approximate, measurably so at
  the domain boundary where the lift's stencil is one-sided, but `theta`
  is still an O(1) log-adjustment rather than carrying the whole friction
  magnitude.
- **Positive-definite membrane block as `h -> 0`.**  The composite flow law
  adds a small linear term at a *constant* reference thickness, so calving
  fronts, nunataks and ocean buffers stay well posed.

## Thickness floors

A floor on the membrane coupling is sometimes needed for coercivity at
ice-free nodes.  It must **not** be applied to the driving stress:
clamping `H` there fabricates a spurious `rho g H_floor grad(s)` and blows
the buffer velocity up.  `momentum_residual` takes `h_floor` and applies
it only to the `-h M : eps(v)` term.

The interlayer closure needs a second, separate floor.  It normalises the
velocity jump by `h_above + h_below` -- the summed thickness of the two
layers meeting at that interface, `2H/L` for `L` uniform layers -- which
is exactly zero at ice-free nodes.  Any domain buffered seaward past the
calving front, which is the configuration this readme recommends, has
them, and the unguarded division makes the residual NaN there: the solve
dies with `DIVERGED_FUNCTION_NANORINF` before Newton's first step.
`dual_residual` takes `h_jump_floor` (default `viscosity.H_JUMP_FLOOR`,
1 m) and applies it **only** to that denominator -- to no physics term,
not the driving stress, not the effective pressure, not the membrane
coupling.  There is no ice to shear at those nodes, so any finite value
serves; the floor exists only to keep the residual finite.  It engages
once the summed pair drops below it, i.e. below a column thickness of
`L * h_jump_floor / 2` for uniform layers, so a 10-layer column is
floored below 5 m rather than below 1 m.

## Single-layer and multilayer are the same builder

`L = 1` **is** the ordinary single-layer icepack2 dual model: the state
reduces to `(u, M, tau)` on `V x Sigma x T` and the interlayer loop is
empty.  So `dual_residual` covers both, and there is no separate
single-layer code path to keep in step.

`spaces.dual_function_space(mesh, num_layers=1)` builds that space, and
`test/dual_forms_test.py` checks element-for-element that it reproduces
what the single-layer consumers write by hand (`Z = V * Sigma * T` in
`ismip7/antarctica/scripts/diagnostic_solve.py`) -- so adopting the helper
is not a silent change of discretisation.  Providing it here also means a
single-layer consumer never has to depend on the multilayer package.

```python
from icepack_tools.spaces import dual_function_space, layer_thicknesses
from icepack_tools.friction import weertman_anchor
from icepack_tools.momentum import dual_residual

Z = dual_function_space(mesh)                    # single layer
z = Function(Z)                                  # cold start is fine
F = dual_residual(z, theta, phi, H=H, s=s, b=b,
                  h_layers=layer_thicknesses(H, 1),
                  C_w0=weertman_anchor(H, s, u_obs, m, Q),
                  A_layers=[A], n_consts=[n], n_vals=[3.0],
                  m_slide=m, mesh=mesh, law="budd")
```

## Friction laws

Three, selected with `law=`, named to match ismip7's `fric_law`.  All
share the Weertman branch `tau_W = C_w0 exp(theta He) |u|^(1/m)` and
differ only in how the bed's strength is capped:

| `law` | `tau_b` | zero afloat |
|---|---|---|
| `weertman` | `tau_W` | **no** -- there is no cap |
| `budd` | `tau_W * He * N_hat`, `N_hat = N/N_ref` normalised | yes -- the `He` gate |
| `regularized_coulomb` | `tau_W tau_cap / (tau_W + tau_cap)`, `tau_cap = c0 N` | yes -- `N` is a factor |

Budd's zero afloat is enforced by the grounded indicator `He`, not by the
sign of `N`.  `N` is the cancelling difference `p_I - p_W`, which on a
shelf is a roundoff residue rather than 0; with a frozen `N_ref` the
`nhat_floor` term `nhat_floor * p_I / N_ref` would otherwise amplify that
residue all the way to `nhat_cap`.  `He` is a function of height above
flotation, so it is zero hundreds of metres below flotation whatever the
last bits of `N` do -- and it is continuous, which a gate on `N > 0` is
not.  `regularized_coulomb` needs no such gate: `N` enters as a factor,
so the residue passes straight through instead of being amplified.

Measured on the test slab (`L = 1`, `n = 3`, cold start, `theta = 0`),
maximum `|tau_b|` on cells 100 m or more below flotation.  The two capped
laws are bounds rather than values, for the reason given above: what the
solve prints there is its own roundoff on an analytically zero quantity.

| law | its | max speed | shelf drag |
|---|---|---|---|
| `regularized_coulomb` | 23 | 2081.3 m/yr | < 1e-13 kPa (< 1e-14 of grounded max) |
| `budd` | 23 | 868.6 m/yr | < 1e-15 kPa (< 1e-18) |
| `weertman` | 22 | 429.2 m/yr | 6.2e+00 kPa (1.9e-02) |

`budd` keeps its own, much tighter bound because it gates on the grounded
indicator `He` as well as capping on `N`, so what is left afloat is the
`tau` solve's own roundoff rather than the cap's -- a single bound loose
enough to cover both laws would stop testing that.  Its residue is also
the more volatile: it moved six decades under this branch's anchor
change, where `regularized_coulomb`'s moved a factor of 1.3, which is why
it carries the wider margin over what it prints.

Weertman's nonzero shelf drag is correct, not a bug -- it has no
effective-pressure cap.  The test asserts it *is* nonzero (above 1e-6 of
the grounded maximum), so that the machine-zero assertions on the other
two are known to be discriminating rather than vacuously true.

## Modules

| module | contents |
|---|---|
| `constants` | physical constants, re-exported from icepack2 |
| `spaces` | `dual_function_space` (L >= 1), `layer_thicknesses`, `split_layers` |
| `geometry` | `cg1_lift`, `surface_slope` (CG1 *or* DG0 safe) |
| `grounding` | `height_above_flotation`, `grounded_mask`, `effective_pressure` |
| `friction` | `weertman_anchor` (partition-independent), `basal_stress` (3 laws), `friction_residual`, `speed_limiter` |
| `viscosity` | `membrane_residual`, `interlayer_residual` (composite, regularised) |
| `momentum` | `momentum_residual`, `calving_terminus`, `dual_residual` (L >= 1) |
| `parallel` | `gather`/`scatter`, `gather_vector`, `gather_speed`, `stats`, `format_stats` |

## Calving fronts

`calving_terminus` is needed only when the calving front *is* the domain
boundary.  If the domain is buffered past the front so the terminus is
interior, leave it out: the dual form is well posed at zero thickness and
the front's stress balance comes out of the thickness gradient itself.
Adding a boundary back-pressure there double-counts it.

## Install

```bash
pip install --no-deps -e /media/andrew/wd1/projects/icepack_tools
```

Depends on `firedrake` and `icepack2`.

## Provenance

Ported and generalised from `ismip7/icepack2_tools/dual_friction.py`,
`grounding.py` and `geometry.py`, which in turn derive from
`gia-icepack/scripts/ase_model.py:build_F_rc`.  The multilayer assembly is
new, and follows the layer ordering of
[multilayer](https://github.com/hoffmaao/multi-layer).
