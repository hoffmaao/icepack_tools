# Level set as the primary front

> **Historical record.** This plan records what was decided on 5 September
> 2026 and is kept as it was written. Its references to ISSM source files and
> defaults are observations of one ISSM revision on that date, not citations.
> For the method's citations see the `src/icepack_tools/levelset.py` module
> docstring: Hahn, Mikula and Frolkovic (2025) for the eikonal boundary
> condition, and Bondzio et al. (2016) for the front kinematics.

Plan of record, 5 September 2026. Companion to `src/icepack_tools/levelset.py`.
A phased plan to move the ice-front treatment from a thickness-anchored level
set to the convention ISSM uses, where the level set carries the front,
everything outside it is deactivated, and thickness follows. Each phase has a
gate that must pass before the next starts. The ISMIP7 control simulation's
pinned front is preserved throughout.

Decided 5 Sep (Andrew): the level-set developments live here, in the
CalvingMIP toolbox, whose `LevelSet` both icepack2/ISMIP7 and CalvingMIP run.
Phases 1, 2 (the mask fields), 3 and 4 are toolbox work with their own tests
and pull requests. The consuming model's part is the residual keyword, the
budget columns, the protocol wiring for the control, and the Antarctic
validation in Phase 5.

## The decision in one paragraph

Today the front is the boundary of the ice extent: each step the level set is
rebuilt as the signed distance to the cells holding more than one metre of
ice, so it can only describe where the thickness already is. That is why
zeroing thickness or resistance "outside the front" pins advance, and why the
review rounds of 4-5 Sep went into rules that keep sub-threshold inflow alive.
ISSM inverts the dependency. Its level set is state, advected each step with
an extrapolated velocity minus the calving rate along the normal; elements
with no ice under that level set are not assembled at all; thickness there is
held at a floor. Masking by the front is then consistent, because the front
moves on its own. This package already contains most of that machinery under
its shelved `advect` anchor. This plan promotes it, one gated phase at a time.

## What ISSM does, from its source

Read on 5 Sep 2026 from `ISSMteam/ISSM` main: `cores/movingfront_core.cpp`,
`modules/SetActiveNodesLSMx`, `modules/KillIcebergsx`,
`analyses/{Extrapolation,Levelset,Masstransport,Stressbalance}Analysis.cpp`,
`classes/Elements/Element.cpp`, `classes/FemModel.cpp`, and the Python
parameter classes `levelset.py`, `masstransport.py`, `calving.py`.

- **Classification.** An element has ice if the minimum of its vertex level
  set is negative (`IsIceInElement`); it is ice-only if the maximum is
  non-positive. Any ice at any vertex activates the whole element, so front
  elements are assembled in full and the front pressure is applied where the
  level set crosses zero inside them.
- **Deactivation, not zeroing.** Every analysis begins
  `if(!element->IsIceInElement()) return NULL;`. Ice-free elements contribute
  nothing to the stress balance or to mass transport, and nodes belonging
  only to such elements are removed from the system through the
  node-activation mask. There is no friction, viscosity or drag "in the
  water" because there is no equation there.
- **Thickness outside.** After each transport solve, nodal thickness below
  `md.masstransport.min_thickness` is clamped to it and the clamped amount
  stored as a residual field. The default the setup routine installs is 1 m.
  The floor is not conserved and not booked as calving.
- **Front motion.** Per step: extrapolate velocity and calving rate beyond
  the front by a diffusion-based extension along the normal; advect the level
  set with velocity minus calving rate along the normal, capped by
  `migration_max`; every `reinit_frequency` steps (default 10) recompute it as
  a signed distance; optionally kill disconnected ice by setting its level set
  positive and reinitialising.

## Where we are

| | |
|---|---|
| front state | DG0 signed distance, `anchor="extent"`: rebuilt from the thickness every step, `phi = 0` on ice/water facets of cells with `h > h_min` (1 m). |
| advance | Owned by the DG0 upwind transport; sub-threshold inflow into water cells is kept so cells can cross the threshold and join the extent (icepack2 `front.retreat_slivers`, 5 Sep). |
| removal | `calving_masks()`: whole cells the front has passed, plus a sub-cell fraction `c dt L / A` shed from front cells; booked to the calving column. |
| resistance in water | Ocean drag gated off within one cell of the front (`drag_mask`), on beyond; basal friction still active because the consuming model floors the overburden at 1 m; a viscous floor and a grounding-line-gated viscous collar for coercivity. |
| shelved | `anchor="advect"`: harmonic extension of `(u, c)` from ice nodes, inflow-implicit upwind advection with `w = u_ext - c ghat`, fixed-point eikonal reinitialisation every `reinit_every` steps. Set aside on 2 Sep because reinitialisation anchors drifted and advance was made the transport's job. |
| protocol (ISMIP7) | Control: calving "set constant to end of 2014 conditions", implemented as a pinned front with apparent mass balance. Projections: a physically based calving law, never silently pinned. |

Two conventions on a strip of four DG0 cells at the margin:

```
A. extent anchor (today): thickness is primary
   | h = 320 m | h = 140 m || h = 0.05 m | h = 0     |     front at the 1 m crossing
   phi is recomputed from h each step. Masking h by phi > 0 would erase
   the 0.05 m and freeze the front.

B. advected anchor (ISSM, this plan): the level set is primary
   | h = 320 m | h = 140 m | h = 12 m  |  | deactivated |   front inside a cell
   phi advects with u_ext - c n and is reinitialised. Cells with phi > 0 are
   not assembled; h there is held at the floor.
```

## Target design in this framework

The finite-volume level set stays DG0 and stays in this package; consuming
models use it through their shims. What changes is which quantity is primary
and where the masks are applied.

### State and motion

- `phi` is checkpointed state (`state_fields()` already returns it) and is the
  front. Per step: extension of `(u_x, u_y, c)` from ice nodes, advection with
  `w = u_ext - c ghat`, reinitialisation every `reinit_every` steps. Our
  default is 5; ISSM's is 10. Keep ours until Phase 1 measures drift.
- The ice indicator is the level set's own `chi` (cell has `phi < 0`), not
  `h > 1 m`. Front cells are those with `|phi| < cell_diam`.

### Momentum

- The level set exposes an `ice_mask` DG0 gate alongside the existing
  `drag_mask`. The consuming residual multiplies basal friction by it (no bed
  contact in water). This transfers to today's convention on its own and is
  Phase 0.
- Full deactivation of water cells is not available in Firedrake's assembly
  the way ISSM removes elements, so it is emulated: `ice_mask` also multiplies
  the driving stress and the membrane terms, while the coercivity terms that
  must survive there (viscous floor, collar, far-field ocean drag) are left
  unmasked so the system stays non-singular. Water cells then hold a damped,
  near-rigid null solution that transmits no stress into the ice.
- The front pressure term is unchanged: with DG0 geometry the facet term
  `rho g avg(h) jump(s)` at an ice/water face already is the terminus load.

### Thickness

- Cells with `phi > 0` are held at a floor after each transport step, as ISSM
  does, but the clamped mass is booked: the part that came from ice cells the
  front has passed goes to calving; the rest to a clamp column, which becomes
  a reported residual rather than a silent one. The floor is the front
  threshold `h_min`.
- Advance is the level set's. When `phi` turns negative in a cell that held
  only the floor, the cell activates and the transport fills it from
  upstream, exactly as ISSM's newly activated element receives mass. The
  consuming model clears its apparent-mass-balance reference on the t = 0
  water mask for every law, as it does now.
- The sub-cell shed and the retreat-sliver rule are retired in advected mode;
  they exist to compensate a front that is rebuilt from thickness.

### Laws and the control

- `fixed` keeps its meaning and its implementation: anchored on the initial
  thickness, removal past the initial front, apparent mass balance cleared
  beyond it. It does not advect. The ISMIP7 control configuration is
  untouched.
- `vonmises` and `prescribed` use the advected front. The consuming model's
  legacy fixed-front flag yields to any configured law.
- `none` means no level set.

## Phases and gates

### Phase 0. Friction gate on the ice indicator (1 to 2 days)

Expose `ice_mask` from the level set (written from `chi` the way `drag_mask`
is written); the consuming residual multiplies basal friction by it. Works
under today's extent anchor and is the one ISSM behaviour that transfers
cleanly. Own commit, own gate run.

**Gate.** Synthetic ice tongue in the existing test harness: friction is
exactly zero on water cells, unchanged on ice cells; a 32 km ten-step ISMIP7
run keeps its budget residual at 0.00 and the control (pinned) numbers
bit-identical, since it never has ice in water.

### Phase 1. Revive the advected anchor and measure its drift (3 to 5 days)

Run `anchor="advect"` on the standing unit tests, then on a synthetic tongue
for 200 steps with a prescribed rate, tracking the distance between the
advected zero contour and the true front and the reinitialisation residual.
This is where September stopped; the gate is quantitative so the decision is
not taken on impression again.

**Gate.** Front position error under one cell over 200 steps at
`reinit_every` in {5, 10}; reinitialisation restores the signed distance to
within 10% of a cell everywhere; no interior-cell anchor drift, the failure
recorded on 2 Sep.

### Phase 2. Momentum masking (3 to 4 days)

Extend `ice_mask` to the driving stress and membrane terms, leaving the
coercivity terms unmasked. Confirm the Newton solve stays as robust as today,
because this touches the exact-zero-shelf regime that caused the July
blow-ups in ISMIP7.

**Gate.** Newton iteration counts and converged residuals on the 32 km and
2500 m Antarctic maps within 10% of today's; water-cell speeds below 1 m/yr;
front-adjacent ice velocity unchanged to 1% against the extent anchor on the
same geometry.

### Phase 3. Thickness follows the front (4 to 6 days)

Floor and book thickness outside `phi > 0`, activate cells the front enters,
retire the sub-cell shed and slivers in advected mode, split the calving and
clamp columns as above. Keep the extent anchor selectable for A/B until
Phase 5 closes.

**Gate.** Budget identity `SMB - melt - calv - outflux - clamp = dM/dt` closes
to machine precision every step; the same prescribed-rate tongue calves the
same total mass under both anchors within 2%; a cell the front enters gains
mass from upstream in the next step.

### Phase 4. Icebergs, restart, checkpoint (2 to 3 days)

Port ISSM's kill-icebergs: connected components of ice cells not attached to
the grounded sheet get `phi` set positive and are reinitialised, with their
mass booked to calving. Save and restore `phi` so a restart continues the
same front rather than rebuilding it from thickness.

**Gate.** Restart at step 5 of a 10-step run reproduces the continuous run's
front, thickness and budget to round-off; a deliberately detached patch is
removed and booked in one step.

### Phase 5. Validation against observations and the control (5 to 8 days)

Score the advected von Mises front against the Greene ice masks (24 epochs,
1997 to 2021) and the ~1300 Gt/yr calving flux the same masks imply, on the
2500 m Antarctic mesh, starting from the net-constrained map so the discharge
feeding the front is right. Run the control with `fixed` before and after and
diff every budget column.

**Gate.** Control budget columns bit-identical before and after; a 25-year
von Mises projection at 2500 m with front position error against Greene under
two cells at the 2021 epoch on the major fronts, and calving flux within the
1300 +/- 140 Gt/yr observed band. If the September tuning problem recurs,
where the flux arriving at the front is far below observed, stop and fix
discharge first.

### Phase 6. Process (ongoing)

Each phase is its own branch, gated through no-mistakes with the phase's gate
written into the intent. Toolbox changes go in here first, as their own pull
requests, since CalvingMIP runs the same object; consuming models pin the
toolbox version they validated against.

## Risks that are already on record

- **Reinitialisation drift.** The 2 Sep decision to anchor on the extent came
  from advected fronts drifting between reinitialisations and interior
  anchors being reset to distorted values. Phase 1's gate exists to measure
  this rather than argue it; if it cannot be met at `reinit_every` of 5, the
  plan stops there.
- **Near-singular momentum.** Masking membrane terms in water is the regime
  where the ISMIP7 RC forward blew up in July. The coercivity terms are
  deliberately left unmasked and Phase 2's gate is on Newton behaviour, not
  on the answer looking right.
- **Discharge deficit at the front.** Any calving-law tuning on a state whose
  grounding-line discharge is 25% short fits the wrong thing. Phase 5 starts
  from the net-constrained map.
- **Two conventions in one code.** Until Phase 5 closes both anchors stay
  selectable; the test suite must run both, and the readme must say which is
  the default for which experiment.
- **Effort.** About four working weeks of focused effort including
  validation. Phase 0 is cheap and stands on its own if the rest is deferred.

## Decisions needed before Phase 1

1. The thickness floor outside the front: ISSM's 1 m, or exactly zero with
   the momentum masking carrying the coercivity. Zero is cleaner for the
   budget; 1 m is what ISSM validated.
2. Whether the ISMIP7 matrix driver should stop exporting its legacy
   fixed-front flag unconditionally, so a projection can request a free front
   through that driver at all.

Nothing in this plan changes the ISMIP7 control run. The pinned front, the
apparent-mass-balance reference cleared beyond the 2014 extent, and the
calving tally past it are the protocol's "calving set constant to end of 2014
conditions" and are preserved by construction at every phase.
