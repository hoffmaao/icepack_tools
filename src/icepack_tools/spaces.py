r"""Mixed function spaces for the icepack2 dual form.

The dual form carries stress as an unknown alongside velocity, so its
state lives in a mixed space.  For one layer that is

.. code::

    Z = V x Sigma x T          velocity, membrane stress, basal stress

with ``V`` a CG vector space, ``Sigma`` a symmetric-tensor DG space and
``T`` a vector DG space -- exactly what the single-layer consumers build
by hand (``ismip7/antarctica/scripts/diagnostic_solve.py`` writes
``Z = V * Sigma * T``).

For ``L`` layers the same triple repeats, ordered bottom to top:

.. code::

    Z = [u^1, M^1, S^0,  u^2, M^2, S^1,  ...,  u^L, M^L, S^(L-1)]

where ``S^0`` is the basal stress and ``S^l`` for ``l > 0`` is the
interlayer stress on the interface below layer ``l + 1``.  ``L = 1``
reduces to the single-layer space above with no interlayer unknowns, so
one residual builder serves both.

Providing this here means a single-layer consumer never has to depend on
the multilayer package to use the rest of ``icepack_tools``.
"""

import firedrake


def dual_function_space(mesh, num_layers=1, degree=1, cell=None):
    r"""Mixed space for an ``num_layers``-layer dual model.

    Parameters
    ----------
    mesh : firedrake.Mesh
    num_layers : int
        1 gives the standard single-layer dual space ``V x Sigma x T``.
    degree : int
        Polynomial degree for velocity (CG).  Stresses use ``degree - 1``
        (DG), so the default ``degree = 1`` pairs CG1 velocity with DG0
        stresses.
    cell : ufl.Cell or str, optional
        Cell to build the elements on; ``mesh.ufl_cell()`` when omitted.
        Passing the cell object through unchanged is what keeps extruded
        meshes (``TensorProductCell``) working.

    Returns
    -------
    firedrake.MixedFunctionSpace
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if cell is None:
        cell = mesh.ufl_cell()

    cg = firedrake.FiniteElement("CG", cell, degree)
    dg = firedrake.FiniteElement("DG", cell, degree - 1)
    V = firedrake.VectorFunctionSpace(mesh, cg)
    Sigma = firedrake.TensorFunctionSpace(mesh, dg, symmetry=True)
    T = firedrake.VectorFunctionSpace(mesh, dg)

    return firedrake.MixedFunctionSpace([V, Sigma, T] * num_layers)


#: Tolerance on ``sum(fractions) - 1`` in :func:`layer_thicknesses`.
FRACTION_TOL = 1e-10


def layer_thicknesses(H, num_layers, fractions=None):
    r"""Per-layer thicknesses, ordered bottom to top.

    ``fractions`` must sum to 1; uniform layers if omitted.  With
    ``num_layers = 1`` this is just ``[H]``.

    The sum is checked rather than trusted: fractions that miss 1 lose
    that share of the column from both the membrane coupling and the
    driving stress, which shows up only as a quietly wrong velocity.
    """
    if fractions is None:
        return [H / num_layers] * num_layers
    if len(fractions) != num_layers:
        raise ValueError(
            f"got {len(fractions)} fractions for {num_layers} layers")
    total = sum(fractions)
    if abs(total - 1.0) > FRACTION_TOL:
        raise ValueError(
            f"fractions must sum to 1, got {total!r} (off by {total - 1.0:.3e})")
    return [H * f for f in fractions]


def split_layers(z, num_layers):
    r"""Group a split dual state into per-layer ``(u, M, S)`` triples.

    ``S`` is the stress on the interface *below* that layer, so
    ``layers[0]["interlayer_stress"]`` is the basal stress.  That key is
    the canonical one on every layer -- matching
    ``multilayer.model.utilities.split_fields``, so a consumer migrating
    off that helper indexes ``layers[l]["interlayer_stress"]`` uniformly
    across layers.  Layer 0 additionally carries ``"basal_stress"``
    pointing at the *same* unknown, purely as a readable alias; the two
    names are interchangeable there.  This is the same backwards
    compatibility bargain as ``momentum.multilayer_rc_residual``.
    """
    fields = firedrake.split(z) if not isinstance(z, (list, tuple)) else z
    if len(fields) != 3 * num_layers:
        raise ValueError(
            f"state has {len(fields)} blocks, expected {3 * num_layers} for "
            f"{num_layers} layers")
    layers = []
    for l in range(num_layers):
        layer = {
            "velocity": fields[3 * l],
            "membrane_stress": fields[3 * l + 1],
            "interlayer_stress": fields[3 * l + 2],
        }
        if l == 0:
            layer["basal_stress"] = layer["interlayer_stress"]
        layers.append(layer)
    return layers
