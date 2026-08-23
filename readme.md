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
carries a large share of the effective fluidity (98 % at 10 kPa, 33 % at
50 kPa, 6 % at 100 kPa for the `n = 4`, `A = 46` layer) and `phi` cannot
adjust that share.

`test/multilayer_rc_test.py` isolates what each ingredient buys.  Two-layer
`n = 4 / 1.8` composite on a Coulomb bed, cold start `z = 0`, `m = 3`
fixed at its target in every case:

| diffusion (n=1) | n | result | max speed |
|---|---|---|---|
| off | ramped 1 -> n | converged, 42 its | 1132.5 m/yr |
| off | **direct** | **diverges** | -- |
| 1e-3 | ramped 1 -> n | converged, 41 its | 1136.1 m/yr |
| 1e-3 | **direct** | **converged, 46 its** | 1136.1 m/yr |

So: the residual friction closure removes the `m`-continuation, and
diffusion creep removes the `n`-continuation.  Together **the whole
continuation apparatus disappears** -- one cold solve replaces a staged
ramp.  The ramped and direct paths agree to 7.9e-12, so the direct solve
is not converging somewhere else.

The test also checks the headline property directly: on the 296 cells
where `N == 0` exactly, `|tau_b|` is 5.3e-15 kPa, i.e. 1.8e-17 of the
grounded maximum -- machine zero, against the ~1 % of grounded drag that a
`phi_eff` floor of 0.01 leaves on every shelf node.

## What this buys

- **Exactly zero drag on floating ice.**  `N = max(p_I - p_W, 0)` is built
  from the *model* surface, so it vanishes at precisely the hydrostatic
  flotation criterion.  No `phi_eff` floor, so no residual shelf drag.
- **Grounded-only friction inference.**  `theta` is carried as
  `exp(theta * He)` with `He` a smooth grounded indicator, so `dJ/dtheta`
  is identically zero afloat: an optimiser physically cannot place basal
  friction on a shelf.
- **A balanced starting control.**  The anchor `C_w0 = tau_d / |u_obs|^(1/m)`
  makes the Weertman branch return exactly `tau_d` at `u = u_obs, theta = 0`,
  so `theta` is an O(1) log-adjustment rather than carrying the whole
  friction magnitude.
- **Positive-definite membrane block as `h -> 0`.**  The composite flow law
  adds a small linear term at a *constant* reference thickness, so calving
  fronts, nunataks and ocean buffers stay well posed.

## Thickness floors

A floor on the membrane coupling is sometimes needed for coercivity at
ice-free nodes.  It must **not** be applied to the driving stress:
clamping `H` there fabricates a spurious `rho g H_floor grad(s)` and blows
the buffer velocity up.  `momentum_residual` takes `h_floor` and applies
it only to the `-h M : eps(v)` term.

## Modules

| module | contents |
|---|---|
| `constants` | physical constants, re-exported from icepack2 |
| `geometry` | `cg1_lift`, `surface_slope` (CG1 *or* DG0 safe) |
| `grounding` | `height_above_flotation`, `grounded_mask`, `effective_pressure` |
| `friction` | `weertman_anchor`, `basal_stress`, `friction_residual`, floor-cell drags |
| `viscosity` | `membrane_residual`, `interlayer_residual` (composite, regularised) |
| `momentum` | `momentum_residual`, `calving_terminus`, `multilayer_rc_residual` |

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
