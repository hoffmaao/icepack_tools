r"""Mixed function spaces for the icepack2 dual form.

The dual form carries stress as an unknown alongside velocity, so its
state lives in a mixed space.  For one layer that is

.. code::

    Z = V x Sigma x T          velocity, membrane stress, basal stress

with ``V`` a CG vector space, ``Sigma`` a symmetric-tensor DG space and
``T`` a vector DG space (CG with ``stress_family="CG"``) -- exactly what the single-layer consumers build
by hand (``ismip7/antarctica/scripts/diagnostic_solve.py`` writes
``Z = V * Sigma * T``).

For ``L`` layers the same triple repeats, ordered bottom to top:

.. code::

    Z = [u^1, M^1, S^0,  u^2, M^2, S^1,  ...,  u^L, M^L, S^(L-1)]

where ``S^0`` is the basal stress and ``S^l`` for ``l > 0`` is the
interlayer stress on the interface below layer ``l + 1``.  ``L = 1``
reduces to the single-layer space above with no interlayer unknowns, so
one residual builder serves both.

The basal and interlayer stresses live one value per cell by default
(DG), or at the vertices with the velocities (CG, **nodal stresses**),
whose closures are collocated there with :func:`vertex_measure`.
"""

import firedrake
from icepack2.model.utilities import get_test_function

#: Where the basal and interlayer stresses may live.
STRESS_FAMILIES = ("DG", "CG")


def dual_function_space(mesh, num_layers=1, degree=1, cell=None,
                        stress_family="DG"):
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
    stress_family : {"DG", "CG"}
        Where the basal and interlayer stresses live.  ``"DG"`` (the
        default, icepack2's) is one value per cell, closed on the cell
        means of the velocities.  ``"CG"`` puts them at the vertices with
        the velocities, degree 1 (so ``degree`` must be 1), for closures
        collocated there with :func:`vertex_measure`: **nodal stresses**.
        With them an empty layer's momentum balance equates its two
        interface stresses node by node rather than in projection, so a
        layer of zero thickness drops out of the solution exactly, and the
        sliding velocity under an empty bottom layer is determined node by
        node.  The membrane stress stays cellwise either way.  See
        :func:`icepack_tools.momentum.dual_residual`.

    Returns
    -------
    firedrake.MixedFunctionSpace
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if stress_family not in STRESS_FAMILIES:
        raise ValueError(f"stress_family must be one of {STRESS_FAMILIES}, "
                         f"got {stress_family!r}")
    if stress_family == "CG" and degree != 1:
        raise ValueError("nodal stresses need degree 1: they are closed with "
                         "vertex quadrature, under which higher-degree basis "
                         "functions vanish at every quadrature point")
    if cell is None:
        cell = mesh.ufl_cell()
    cg = firedrake.FiniteElement("CG", cell, degree)
    dg = firedrake.FiniteElement("DG", cell, degree - 1)
    V = firedrake.VectorFunctionSpace(mesh, cg)
    Sigma = firedrake.TensorFunctionSpace(mesh, dg, symmetry=True)
    T = firedrake.VectorFunctionSpace(mesh, cg if stress_family == "CG" else dg)
    return firedrake.MixedFunctionSpace([V, Sigma, T] * num_layers)


def stress_family(space):
    r"""The element family of a (vector) stress space: ``"Lagrange"`` for
    nodal stresses, ``"Discontinuous Lagrange"`` for cellwise ones."""
    elem = space.ufl_element()
    fam = elem.family()
    if fam == "Mixed":                     # a vector element that hides its scalar
        sub = elem.sub_elements
        fam = (sub() if callable(sub) else sub)[0].family()
    return fam


def is_nodal(space):
    r"""Does this stress space put its values at the vertices?"""
    return stress_family(space) in ("Lagrange", "Q")


def vertex_measure(mesh):
    r"""``dx`` with vertex quadrature on triangles.

    A term integrated with it against a continuous test function is
    collocated at the vertices (the lumped mass), so a closure taken in it
    holds node by node: :math:`\tau_i = -\tau_b(u_i)`,
    :math:`c(S_i)\,S_i = (u^{l}_i - u^{l-1}_i)/h_i`.  Nodal stresses
    (:func:`dual_function_space` with ``stress_family="CG"``) take every
    basal and interlayer stress term in this measure -- the closures and
    the stress terms of the momentum balance alike -- so the whole thing
    is still the derivative of one action.
    """
    from finat.point_set import PointSet
    from finat.quadrature import QuadratureRule
    name = mesh.ufl_cell().cellname
    if (name() if callable(name) else name) != "triangle":
        raise NotImplementedError(
            "nodal stresses: the vertex quadrature rule is set up for triangles")
    rule = QuadratureRule(PointSet([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                          [1.0 / 6.0] * 3)
    return firedrake.dx(scheme=rule)


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
    multilayer's ``split_fields``, so a consumer migrating
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


def _test(field, kwargs):
    """The test function paired with ``field``: the caller's
    (``kwargs["test_function"]``), or the one of the mixed space ``field``
    was split from."""
    sigma = kwargs.get("test_function")
    return get_test_function(field) if sigma is None else sigma
