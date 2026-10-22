r"""Thickness transport at a calving front that a level set carries.

An advected level set (:class:`icepack_tools.levelset.LevelSet`,
``anchor="advect"``) moves the front with the velocity ``u - c n_f`` (ice
velocity ``u``, frontal ablation rate ``c``, outward front normal
``n_f``).  The thickness transport has to move the ice with the same
front, and the obvious coupling does not.  That coupling transports with
the plain upwind scheme everywhere, then empties every cell beyond the
front.  The ice the transport carries across a front facet in one step is
``h (u.n) dt``, whatever the front does, and emptying the cell it lands in
books all of it as calving.  So a front advancing at the ice speed
(``c = 0``) calves every bit of ice that reaches it, and the front cells,
which keep exporting into the water at the full ice speed, drain.  In
CalvingMIP experiment 4 that coupling left the front a third of the
ensemble's thickness and booked 27-73 km3/yr of calving at zero rate.  On
the ISMIP7 Antarctic forward on a 5 km buffered mesh it booked 6,650 Gt/yr
in its first step, where the extent-anchored front booked 1,370.

:class:`FrontTransport` is the coupling that agrees with the front.  It
was built and tested in the CalvingMIP model, and it moves here so that
every project with an advected front runs the same one.

* **Only cells inside the front export ice.**  Across a facet from an
  inside cell into a cell beyond the front, the ice deposited is what
  follows the front's own advance, ``h ((u - c n_f) . n)+``, capped at
  ``h (u . n)+``.  The rest of ``h (u . n)+`` is the calving flux, and the
  transport books it itself (:meth:`FrontTransport.calving`).  A free
  front (``c = 0``) calves nothing.  A held front calves everything that
  reaches it: that is the ``fixed`` law everywhere, and the barrier where
  it binds.  A retreating front calves ``h c`` per unit length.
* **The front rules after the solve** (:meth:`FrontTransport.rules`).  A
  cell beyond the front that shares an edge with the ice is a *band*
  cell.  A band cell keeps what the advance carried in.  If the front
  retreated off it this step, it keeps the area fraction of it still
  behind the front.  Every other cell beyond the front is emptied.
* **A band cell holds no more than the ice it came from**
  (:meth:`FrontTransport.excess`).  The flow along a front leaks across
  the mesh faces oblique to it, while the level set moves the front along
  its normal.  Where the front is slow or stands still, that ice has
  nowhere to go: on Thule's ridge tips one cell reached 2,482 m.  The
  excess over the inside neighbours' thickness goes back to them.
* **Open water the level set encloses in the ice is put back inside the
  front** (:meth:`FrontTransport.fill_enclosed_water`), at the start and
  after every advance, under every law.  A calving front faces the ocean,
  so water with no path through water to the domain edge is not a front.
  A nunatak, a lake or a level-set artefact on a grounded ridge took in
  ice from every side and could not export it, and under a held front
  everything flowing into a nunatak would calve.
* **Ice cut off from grounded ice is an iceberg**, and leaves as calving
  (:meth:`FrontTransport.icebergs`).
* Optionally a **barrier**: the front may advance no further than a given
  extent ``phi0``.  That is the retreat-only front of most ISMIP6 models.
  Given the cells of ``sea`` it holds only there: a land-terminating front
  moves on with the ice, and what it carries into the sea past the barrier
  calves.

Every removal is a volume the caller books, so the mass budget closes to
the linear solver's tolerance.

The thickness lives on the level set's own cells.  It is either a DG0
field, or a DG1 field on the same mesh whose cell means the rules read;
the caller applies the per-cell fractions the rules return to the whole
cell.  :meth:`FrontTransport.divergence` adds the DG1 volume term on the
inside cells.

A step, with ``ft = FrontTransport(level_set, u)``:

.. code::

    c_mean = ft.advance(dt, rate)        # the front, then the inside set
    # the consumer's transport solve, its divergence term ft.divergence(h, w),
    # sources on the inside cells; then the volumes it books:
    calving = assemble(ft.calving(h)) * dt
    keep, band, touch = ft.rules()       # h *= keep; the rest is calving
    give, back, moved = ft.excess(h, band, touch)
    berg = ft.icebergs(h, grounded)      # emptied; booked as calving

Units: metres, years.
"""

import numpy as np

from firedrake import (
    Function, TestFunction, FacetNormal, assemble, dot, dx, dS, ds, grad,
    inner, max_value, min_value,
)

from .geometry import cg1_lift

__all__ = ["FrontTransport"]


def _global_count(mask, comm):
    return int(comm.allreduce(int(np.count_nonzero(np.asarray(mask)))))


class FrontTransport:
    r"""The front-aware transport fluxes and the front rules of an advected
    level set.

    Parameters
    ----------
    level_set : icepack_tools.levelset.LevelSet
        The front, ``anchor="advect"``.  An extent-anchored front reports
        its removal through ``LevelSet.calving_masks`` and needs none of
        this.
    u : Function
        The ice velocity the transport and the level set move with (the
        same Function the consumer's transport form reads).
    barrier : bool
        Hold the front at ``phi0``: it may retreat but not advance past it.
    phi0 : Function (DG0) or None
        The barrier's extent.  Defaults to the level set's construction-time
        ``phi0``.  A consumer restarting from a saved level set passes the
        t = 0 distance, because the construction-time field of a restarted
        run is the restart's front.
    sea : array of bool or None
        Where the barrier holds, one entry per owned cell: the cells whose
        bed is below sea level.  A land-terminating front does not calve;
        it moves with the ice, so the barrier lets it advance over land,
        and land ice that flows on into a cell of sea past the barrier
        calves there (see ``icepack_tools.calving``, *Land-terminating
        fronts*).  ``None`` holds the barrier everywhere.  The ``fixed``
        law holds its whole front either way.
    """

    def __init__(self, level_set, u, barrier=False, phi0=None, sea=None):
        if level_set.anchor != "advect":
            raise ValueError(
                "FrontTransport couples the transport to an advected front "
                "(anchor='advect'); an extent-anchored front reports its "
                "removal through LevelSet.calving_masks")
        ls = self.level_set = level_set
        self.mesh, self.comm = ls.mesh, ls.comm
        self.u = u
        self.barrier = bool(barrier)
        self.phi0 = phi0 if phi0 is not None else ls.phi0
        n_own = len(ls.phi.dat.data_ro)
        if sea is None:
            self.sea = np.ones(n_own, dtype=bool)
        else:
            self.sea = np.asarray(sea, dtype=bool)
            if self.sea.shape != (n_own,):
                raise ValueError(f"sea names {self.sea.shape} cells; the level set "
                                 f"owns {n_own}")
        Q0, W0 = ls.Q0, ls.W0
        self.chi_in = Function(Q0, name="inside_front")
        self.c_vec = Function(W0, name="calving_velocity")
        self.held = Function(Q0, name="front_held")
        #: cells the front may not advance into: beyond the barrier's extent
        self.barred = Function(Q0, name="beyond_barrier")
        self.nfront = Function(W0, name="front_normal")
        #: owned cells inside the front when the last advance began
        self.inside_old = None
        #: lake cells the last advance filled back into the ice (all ranks)
        self.n_filled = 0
        self._edges = None
        self._flag = Function(Q0, name="connected")
        self._h_cell = Function(Q0, name="cell_thickness")
        self._g_cell = Function(Q0, name="grounded_cell")
        self._q_ret = Function(Q0, name="excess_per_edge")

        n = FacetNormal(self.mesh)
        un = dot(u, n)
        self._n = n
        self._un_plus = (un + abs(un)) / 2
        w = TestFunction(Q0)
        self._w0 = w
        chi, hc, q_ret = self.chi_in, self._h_cell, self._q_ret
        # length of each cell's edges shared with an inside cell
        self._touch_form = (chi("+") * w("-") + chi("-") * w("+")) * dS
        # and those inside neighbours' thickness, length-weighted
        self._nbr_form = (chi("+") * hc("+") * w("-")
                          + chi("-") * hc("-") * w("+")) * dS
        # what a band cell hands back, to the inside cells it touches
        self._return_form = (chi("+") * q_ret("-") * w("+")
                             + chi("-") * q_ret("+") * w("-")) * dS
        # The water the level set encloses at the start belongs to the ice
        # too (a nunatak, a lake), and to the barrier's extent: a barrier
        # that kept it beyond the front would push it back out every step.
        self.n_filled = self.fill_enclosed_water()
        if self.barrier:
            self.phi0 = self.phi0.copy(deepcopy=True)
            lake = self._lakes(self.phi0)
            if _global_count(lake, self.comm):
                self.phi0.dat.data[lake] = -0.5 * ls.cell_diam.dat.data_ro[lake]
            self.barred.dat.data[:] = np.where(
                (self.phi0.dat.data_ro > 0.0) & self.sea, 1.0, 0.0)
        self.refresh()

    # ------------------------------------------------------------------
    # The transport's fluxes
    # ------------------------------------------------------------------
    def _deposit(self, side):
        r"""Speed carrying ice across a facet from ``side``, per unit
        thickness.  All of it between two inside cells.  Into a cell beyond
        the front, the normal component of the front's own velocity
        ``u - c n_f``, capped at the ice's; nothing where the front is held,
        and nothing into a cell beyond the barrier, which the front can never
        reach.  ``c`` is the exporting cell's, which is the rate the level
        set moves a front facet with (it reads the rate from the ice side).

        The barrier needs both sides.  The exporting side alone (held where
        the barrier binds) leaks: where the ice is slow, the periodic
        reinitialisation lifts a front cell's level set a little above the
        barrier, the cell is no longer held, and it deposits into cells past
        the barrier that the front never claims.  On a strip at 40 m/yr that
        kept a quarter of the ice reaching the barrier beyond it instead of
        calving it."""
        n, un_plus = self._n, self._un_plus
        chi, cv, held, barred = self.chi_in, self.c_vec, self.held, self.barred
        other = "-" if side == "+" else "+"
        to_front = (1.0 - held(side)) * (1.0 - barred(other)) * min_value(
            un_plus(side), max_value(dot(self.u(side) - cv(side), n(side)), 0.0))
        return chi(side) * (chi(other) * un_plus(side)
                            + (1.0 - chi(other)) * to_front)

    def flux(self, q, w):
        r"""The facet terms of the upwind ``div(q u)`` tested against ``w``,
        with only the inside cells exporting: every inside cell's outflow
        leaves it, the deposit arrives in the cell across, and the domain
        edge lets the inside cells' ice out and none in."""
        chi, un_plus = self.chi_in, self._un_plus
        out_p, out_m = chi("+") * un_plus("+"), chi("-") * un_plus("-")
        return ((out_p * q("+") * w("+") - self._deposit("+") * q("+") * w("-")
                 + out_m * q("-") * w("-") - self._deposit("-") * q("-") * w("+"))
                * dS + chi * un_plus * q * w * ds)

    def divergence(self, q, w):
        r"""The whole upwind ``div(q u)`` tested against ``w``: :meth:`flux`,
        and for a DG1 (or higher) test function the volume term
        ``-q u . grad(w)`` on the inside cells.  A cell beyond the front only
        accumulates what the front deposits, so it has no volume term."""
        form = self.flux(q, w)
        if w.ufl_element().degree() > 0:
            form += -self.chi_in * q * inner(self.u, grad(w)) * dx
        return form

    def calving(self, q, w=None):
        r"""The calving flux, per unit time: the ice the inside cells export
        across the front that does not follow it.  Without ``w`` a 0-form
        (the total); with a DG0 test function ``w``, each exporting cell's
        share."""
        chi, un_plus = self.chi_in, self._un_plus
        lose_p = (chi("+") * un_plus("+") - self._deposit("+")) * q("+")
        lose_m = (chi("-") * un_plus("-") - self._deposit("-")) * q("-")
        if w is None:
            return (lose_p + lose_m) * dS
        return (lose_p * w("+") + lose_m * w("-")) * dS

    def outflow(self, q):
        r"""Ice the inside cells export across the domain edge, per unit
        time: a 0-form."""
        return self.chi_in * self._un_plus * q * ds

    # ------------------------------------------------------------------
    # The front, before the solve
    # ------------------------------------------------------------------
    def refresh(self):
        r"""The inside set from the level set as it is now.  :meth:`advance`
        calls it; a consumer that rewinds the level set calls it too."""
        self.chi_in.dat.data[:] = np.where(
            self.level_set.phi.dat.data_ro <= 0.0, 1.0, 0.0)

    def advance(self, dt, rate=None):
        r"""Move the front over ``dt`` with the velocity and, for the
        ``prescribed`` law, the ablation ``rate`` (see
        ``LevelSet.advance``); then hold it at the barrier, fill the water
        it encloses, and take the inside set the transport exports from.
        Returns the level set's mean front rate [m/yr]."""
        ls = self.level_set
        self.inside_old = ~ls.beyond_front()
        self.held.assign(1.0)
        self.c_vec.assign(0.0)
        if ls.law == "fixed":
            # the front stays where it is, so everything that reaches it
            # calves; the water it encloses was filled at construction
            c_mean = ls.advance(dt, self.u, rate=rate)
            self.n_filled = self.fill_enclosed_water()
            self.refresh()
            return c_mean
        # The front normal of the front the advance starts from, which is
        # the one the level set's own step uses.
        g = ls.cell_gradient()
        mag = np.sqrt((g * g).sum(axis=1) + 1e-12)
        self.nfront.dat.data[:] = g / mag[:, None]
        c_mean = ls.advance(dt, self.u, rate=rate)
        # The deposit follows the front's velocity u - c n_f, with c the level
        # set's own extended rate, not the change in phi, which the periodic
        # reinitialisation perturbs (in CalvingMIP it swung between 0 and
        # 1,200 m/yr on a 500 m/yr advance).
        self.c_vec.interpolate(ls.c_cell * self.nfront)
        self.held.assign(0.0)
        if self.barrier:
            # held where the advance would carry the front past the barrier,
            # on the sea; over land the front moves on with the ice
            phi, phi0 = ls.phi.dat.data, self.phi0.dat.data_ro
            hold = (phi0 >= phi) & self.sea
            self.held.dat.data[hold] = 1.0
            phi[hold] = phi0[hold]
            ls.update_cell_fields()
        self.n_filled = self.fill_enclosed_water()
        self.refresh()
        return c_mean

    # ------------------------------------------------------------------
    # The front rules, after the solve
    # ------------------------------------------------------------------
    def rules(self):
        r"""``(keep, band, touch)`` per owned cell, after the transport.

        ``keep`` is the fraction of each cell's thickness that stays: 1
        inside the front; for a band cell (beyond the front, sharing an edge
        with an inside cell), 1 if it was already beyond the front when the
        advance began, since it holds what the deposit carried in, or, if
        the front retreated off it this step, the area fraction of it still
        behind the front; 0 for every other cell beyond the front.  The
        consumer books ``h (1 - keep)`` as calving.  ``touch`` is each
        cell's edge length shared with inside cells.  Collective."""
        ls = self.level_set
        beyond = ls.beyond_front()
        touch = assemble(self._touch_form).dat.data_ro.copy()
        band = beyond & (touch > 0.0)
        inside_old = self.inside_old if self.inside_old is not None else ~beyond
        keep = np.where(beyond, 0.0, 1.0)
        keep[band & ~inside_old] = 1.0
        uncovered = band & inside_old
        frac = self.negative_fraction()
        keep[uncovered] = frac[uncovered]
        return keep, band, touch

    def excess(self, h, band, touch):
        r"""``(give, back, moved)``: what the band cells hold above the ice
        they came from, handed back to it.

        ``h`` is each owned cell's thickness (the cell mean of a DG1
        field), after the cut :meth:`rules` asks for.  A band cell's excess
        is its thickness above the length-weighted mean thickness of its
        inside edge neighbours.  ``give`` is that excess as a fraction of
        the cell's thickness, ``back`` the thickness each inside cell
        receives (the excess, spread over the inside cells it touches in
        proportion to the shared edge), and ``moved`` the volume handed back
        [m3, all ranks].  Nothing is created or calved: the consumer scales
        each band cell by ``1 - give`` and adds ``back``.  A band cell
        filling behind a free advance, always thinner than its source, is
        left alone.  Collective."""
        h = np.asarray(h, dtype=float)
        self._h_cell.dat.data[:] = h
        nbr = assemble(self._nbr_form).dat.data_ro
        h_ext = np.where(band, nbr / np.where(band, touch, 1.0), 0.0)
        excess = np.where(band, np.maximum(h - h_ext, 0.0), 0.0)
        give = np.where(excess > 0.0, excess / np.where(excess > 0.0, h, 1.0), 0.0)
        back = self.hand_back(excess, touch)
        moved = self.comm.allreduce(float((excess * self.level_set.cell_area).sum()))
        return give, back, moved

    def hand_back(self, amount, touch):
        r"""Spread ``amount`` (thickness) of each band cell over the inside
        cells it touches in proportion to the shared edge; returns the
        thickness each owned cell receives.  Collective."""
        area = self.level_set.cell_area
        self._q_ret.dat.data[:] = np.asarray(amount) * area / np.where(touch > 0.0, touch, 1.0)
        return assemble(self._return_form).dat.data_ro / area

    def negative_fraction(self):
        r"""Area fraction of each owned cell behind the front: where the
        lumped P1 interpolant of ``phi`` is negative, in closed form from
        its three vertex values (one negative vertex cuts off a corner
        triangle; two leave the complement of the positive corner).  The
        reinitialisation's contour reset keeps the front on the same
        interpolant's zero contour.  Collective."""
        ls = self.level_set
        phi_v = Function(ls.D1).interpolate(
            cg1_lift(ls.phi)).dat.data_ro_with_halos.reshape(-1, 3)
        p = phi_v[:len(ls.phi.dat.data_ro)]
        neg = p < 0.0
        k = neg.sum(axis=1)
        f = np.where(k == 3, 1.0, 0.0)
        eps = 1e-30
        for target, one in ((1, True), (2, False)):
            i = np.flatnonzero(k == target)
            if not i.size:
                continue
            # the vertex alone on its side: the negative one, or the positive
            v = np.argmax(neg[i], axis=1) if one else np.argmin(neg[i], axis=1)
            frac = np.empty(i.size)
            for a in range(3):
                m = v == a
                if m.any():
                    b, c = (a + 1) % 3, (a + 2) % 3
                    ii = i[m]
                    num = p[ii, a]
                    frac[m] = ((num / (num - p[ii, b] + eps))
                               * (num / (num - p[ii, c] + eps)))
            f[i] = np.clip(frac if one else 1.0 - frac, 0.0, 1.0)
        return f

    # ------------------------------------------------------------------
    # Connectivity: enclosed water and icebergs
    # ------------------------------------------------------------------
    def _cell_edges(self):
        r"""The pairs of cells (DG0 dofs, owned and ghost) that share an
        edge, each pair in both orders.

        Regions are grown over a shared edge, not a shared vertex, so two
        patches that meet at a point are two patches.
        ``interior_facets.facet_cell`` is that adjacency,
        and it includes the facets between an owned and a ghost cell, which
        is what carries a flood across a partition."""
        if self._edges is None:
            fc = np.asarray(self.mesh.interior_facets.facet_cell, dtype=np.int64)
            # a rank's interior-facet list can name a cell it does not hold
            # as -1, which numpy would read as the last cell (on 8 ranks that
            # joined an unrelated cell to the open-water flood in CalvingMIP)
            fc = fc[(fc >= 0).all(axis=1)]
            dof = np.asarray(self.level_set.Q0.cell_node_map().values_with_halo)[:, 0]
            a, b = dof[fc[:, 0]], dof[fc[:, 1]]
            self._edges = (np.concatenate([a, b]), np.concatenate([b, a]))
        return self._edges

    def flood(self, region, seed, what="region"):
        r"""Cells of ``region`` connected to ``seed`` across shared edges, on
        every rank.  ``region`` and ``seed`` are boolean arrays over the owned
        and ghost cells; so is the result.

        A connected-component labelling of the region's subgraph on each
        rank, then the outer loop of a sweep across the ranks: mark,
        exchange the halo, take any newly marked ghost cell as a further
        seed, and repeat until no rank marks anything new.  The marked set
        only grows, so every exchange but the last marks a new cell and the
        sweep settles within as many exchanges as the region has cells; one
        that does not raises ``RuntimeError``.  Collective."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        rows, cols = self._cell_edges()
        n = len(region)
        n_own = len(self.level_set.phi.dat.data_ro)
        live = region[rows] & region[cols]
        graph = coo_matrix(
            (np.ones(int(live.sum()), np.int8), (rows[live], cols[live])),
            shape=(n, n)).tocsr()
        n_comp, label = connected_components(graph, directed=False)
        flag = self._flag
        seed = np.asarray(seed, dtype=bool).copy()
        limit = _global_count(region[:n_own], self.comm) + 1
        for _ in range(limit):
            hit = np.zeros(n_comp, bool)
            hit[label[seed]] = True
            reached = region & hit[label]
            flag.dat.data[:] = reached[:n_own]
            reached = region & (flag.dat.data_ro_with_halos > 0.5)
            added = int((reached & ~seed).sum())
            if self.comm.allreduce(added) == 0:
                break
            seed |= reached
        else:
            raise RuntimeError(f"the {what} flood did not settle in {limit} "
                               "halo exchanges")
        return reached

    def _lakes(self, phi):
        r"""Owned cells where ``phi`` (DG0) is water with no path through
        water to the domain edge.  Collective."""
        ls = self.level_set
        water = phi.dat.data_ro_with_halos > 0.0
        edge = ls.chi_bnd.dat.data_ro_with_halos > 0.5
        open_water = self.flood(water, water & edge, "open-water")
        return (water & ~open_water)[:len(phi.dat.data_ro)]

    def fill_enclosed_water(self):
        r"""Put back inside the front any water the level set encloses in
        the ice; returns the number of cells filled (all ranks).

        A calving front faces the ocean.  The water is flooded from the
        domain edge across shared edges; water the flood never reaches has
        no path to the ocean, and its level set is set half a cell inside
        the front (the next reinitialisation restores the distance).  In
        CalvingMIP experiment 4 one such cell on Thule's grounded ridges,
        128 km inside the front, took in ice from every side, could not
        export it, piled up 1,800 m and ran the momentum solve away at
        43,000 m/yr.  A model whose ice mask is "any vertex of the element
        is inside" never sees such a pocket (Bondzio et al., 2016, Sect.
        2.3); a cell-wise level set can.  Collective."""
        ls = self.level_set
        lake = self._lakes(ls.phi)
        count = _global_count(lake, self.comm)
        if count:
            ls.phi.dat.data[lake] = -0.5 * ls.cell_diam.dat.data_ro[lake]
            ls.update_cell_fields()
        return count

    def icebergs(self, h, grounded, h_min=None):
        r"""Owned cells of ice that no path through ice (shared edges)
        connects to a grounded cell: icebergs, for the consumer to empty
        and book as calving.

        ``h`` is each owned cell's thickness, ``grounded`` its grounded
        indicator; ice is ``h > h_min`` (the level set's) inside the front.
        A raft of floating ice with no path back to the bed is left when a
        front retreats through a pinning point or eats a shelf from both
        sides; kept, its momentum problem is singular up to a rigid-body
        motion, and its own accumulation keeps feeding it.  The level set is
        pushed one cell diameter past zero on the icebergs, so they read as
        beyond the front until the next reinitialisation restores the
        distance.  Collective."""
        ls = self.level_set
        h_min = ls.h_min if h_min is None else float(h_min)
        self._h_cell.dat.data[:] = np.asarray(h, dtype=float)
        self._g_cell.dat.data[:] = np.where(np.asarray(grounded, dtype=bool), 1.0, 0.0)
        ice = ((self._h_cell.dat.data_ro_with_halos > h_min)
               & (ls.phi.dat.data_ro_with_halos <= 0.0))
        keep = self.flood(ice, ice & (self._g_cell.dat.data_ro_with_halos > 0.5),
                          "iceberg")
        berg = (ice & ~keep)[:len(ls.phi.dat.data_ro)]
        self.retreat_from(berg)
        return berg

    def retreat_from(self, cells):
        r"""The front leaves the owned ``cells``, a boolean mask: their level
        set is pushed one cell diameter past zero, so they read as beyond the
        front until the next reinitialisation restores the distance.  For a
        consumer that empties cells by a rule of its own (an ice-shelf
        collapse mask), so the front and the thickness keep describing the
        same ice.  Collective; a no-op when no rank names a cell."""
        cells = np.asarray(cells, dtype=bool)
        if _global_count(cells, self.comm):
            ls = self.level_set
            phi, diam = ls.phi.dat.data, ls.cell_diam.dat.data_ro
            phi[cells] = np.maximum(phi[cells], diam[cells])
            ls.update_cell_fields()
            self.refresh()
