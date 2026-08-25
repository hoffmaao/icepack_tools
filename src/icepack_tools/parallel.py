r"""Rank-independent statistics and control-vector gathers.

Every consumer of this package eventually prints a diagnostic like

.. code:: python

    print(f"speed max {np.hypot(*u.dat.data_ro.T).max():.0f} m/yr")

which is **rank-local**: ``PETSc.Sys.Print`` emits from rank 0 only, so on
``N`` ranks the number reported is rank 0's share of the mesh, not the
domain's.  Maxima come out too small and medians come out arbitrary --
and, worse, the answer changes with the rank count, so a run reproduced
on a different machine silently disagrees with itself.  Measured on a
4,679-vertex Store mesh, the velocity RMS misfit read 191.9 m/yr in
serial and 44.2 m/yr on four ranks, from bit-identical states.

The fix is not to sprinkle ``allreduce`` over each statistic: a median or
a percentile is not a reduction, and computing one per rank and averaging
is simply wrong.  :func:`gather` brings the whole nodal array to every
rank, after which ordinary NumPy gives the serial answer exactly, on any
rank count.  A control or diagnostic field is O(#vertices), so the gather
is cheap next to the solves it describes; do not use it inside an
assembly loop.

:func:`gather` and :func:`scatter` are also what let a serial optimiser
-- ``scipy.optimize.minimize`` in every inversion in this ecosystem --
drive a distributed control.  Each rank runs the optimiser redundantly
over the same gathered vector, so every rank takes the same step without
any inter-rank agreement protocol.
"""

import numpy as np
from firedrake.petsc import PETSc

__all__ = ["gather", "scatter", "gather_vector", "gather_speed", "stats",
           "format_stats"]


def gather(f):
    r"""Gather a ``Function``/``Cofunction``'s nodal values onto every rank.

    Returns the full global array, identical on all ranks and identical
    to what the serial run would hold, so NumPy reductions over it are
    rank-independent.  For a vector-valued field the array is flat and
    node-major; use :func:`gather_vector` to get it shaped ``(N, dim)``.

    Assembled 1-forms (``Cofunction``) gather correctly too: ``assemble``
    has already summed each shared node's contributions into its owner,
    so no further reduction is needed here.
    """
    with f.dat.vec_ro as v:
        sc, seq = PETSc.Scatter.toAll(v)
        sc.scatter(v, seq, mode=PETSc.Scatter.Mode.FORWARD)
        out = seq.array.copy()
        sc.destroy()
        seq.destroy()
        return out


def scatter(arr, f):
    r"""Write a full global array back into a distributed ``Function``.

    Inverse of :func:`gather`.  ``arr`` must be the global array -- the
    same length on every rank -- and every rank must pass the same
    values, which is what makes a redundantly-run serial optimiser safe.
    """
    with f.dat.vec_wo as v:
        seq = PETSc.Vec().createSeq(len(arr), comm=PETSc.COMM_SELF)
        seq.array[:] = arr
        sc, _ = PETSc.Scatter.toAll(v)
        sc.scatter(seq, v, mode=PETSc.Scatter.Mode.REVERSE)
        sc.destroy()
        seq.destroy()


def gather_vector(f):
    r"""Gather a vector-valued field as an ``(N, dim)`` array."""
    dim = f.function_space().value_size
    return gather(f).reshape(-1, dim)


def gather_speed(u):
    r"""Gather ``|u|`` at every node of a vector field."""
    return np.linalg.norm(gather_vector(u), axis=1)


def stats(values, mask=None, percentiles=(50, 90, 99)):
    r"""Rank-independent summary of an already-gathered array.

    Pass the output of :func:`gather` / :func:`gather_speed`, optionally
    with a boolean ``mask`` over the same nodes -- typically an ice mask,
    since a buffered domain's ice-free cells otherwise dominate exactly
    the extrema a report is meant to convey.

    Returns a dict with ``n``, ``min``, ``max``, ``mean``, ``rms`` and one
    entry per requested percentile (``p50``, ``p90``, ...).  Returns
    ``n = 0`` and NaNs for an empty selection rather than raising, so a
    report line survives a mask that happens to select nothing.
    """
    v = np.asarray(values)
    if mask is not None:
        v = v[np.asarray(mask, dtype=bool)]
    if v.size == 0:
        out = {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan,
               "rms": np.nan}
        out.update({f"p{p:g}": np.nan for p in percentiles})
        return out
    out = {"n": int(v.size), "min": float(v.min()), "max": float(v.max()),
           "mean": float(v.mean()), "rms": float(np.sqrt((v ** 2).mean()))}
    if len(percentiles):
        pv = np.percentile(v, list(percentiles))
        out.update({f"p{p:g}": float(x) for p, x in zip(percentiles, pv)})
    return out


def format_stats(s, fmt="8.1f", percentiles=(50, 90, 99)):
    r"""One-line rendering of a :func:`stats` dict, for report output."""
    if not s["n"]:
        return "(empty)"
    parts = [f"p{p:g} {s[f'p{p:g}']:{fmt}}" for p in percentiles
             if f"p{p:g}" in s]
    parts.append(f"max {s['max']:{fmt}}")
    return "  ".join(parts)
