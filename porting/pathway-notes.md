# Pathway-aware state-space restriction: notes

Status (2026-10-27): exploratory, not scheduled. Nothing here is to be implemented until the open questions
below are answered. The file records the idea, what the kernel already does, and a possible shape of the work,
so that the exploration can start from it.

## The question

A phase cycle (or gradients) selects only a few coherence pathways of a pulse sequence, say two or three. If the
simulation knows which, can it drop the states that no selected pathway visits and run faster, especially for
large spin systems, without changing much in the kernel?

## What the kernel already does

`propagation.evolution` calls `reduction.reduce` on every call: zero track elimination from `rho`, path tracing
into non-interacting subspaces, and destination state screening against the coil (or `destination`). In a
period where `L` conserves coherence order (rotating-frame `H`, secular relaxation) the non-interacting
subspaces are the order blocks. So when a script filters `rho` with `filters.coherence` (as `cosy`, `noesy` and
`hsqc` in `sequences.py` do), each period is already propagated in the order block it occupies. The pathway
knowledge is used, by hand, through those filters.

## What it does not do

1. The basis, `H`, `R`, `K` and `L` are built on the full space once; nothing shrinks them, so memory and build
   time at large N are untouched.
2. `step` (pulses) runs its Taylor series on the full space, and `evolution` re-slices `P^H L P` on every call.
3. A pulse conserves the coherence order of every isotope it does not act on, exactly, and a pathway fixes that
   order. Nothing uses this.
4. The pathway list is not machine-readable input. `ctps` computes it (`PathwayResult.orders`, per period, with
   amplitudes that include non-ideal pulse leakage), but the sequences hard-code their filters.

## Estimated gain (to be measured)

For N spin-1/2 of one isotope the number of states of total order p is C(2N, N+p) out of 4^N, a fraction of about
1/sqrt(pi N): 0.17 at N = 10, 0.13 at N = 20, so 6-8 times fewer states, and more for per-block dense
exponentials (cubic in the block size). The union of orders {-1, 0, 1} over a whole sequence keeps about
3/sqrt(pi N) (half the space at N = 10), so restricting only the basis gains little for one isotope. The large
gain is multiplicative over isotopes (1H, 13C, 15N, 31P): the order of each unpulsed isotope is fixed in each
pulse and period, and each extra isotope divides the space again. Rough expectation: modest for a single isotope,
an order of magnitude for heteronuclear large systems. These are state counts, not timings.

## Open questions

- Where does the time go at N = 6-12 for COSY, NOESY and HSQC-like cases: basis build, `L` build, pulses, or the
  evolution periods (already reduced)? A profile (`line_profiler` is in the dev environment) decides whether
  gaps 1-3 above matter at all.
- Does a single-isotope system gain enough to justify the machinery, or only heteronuclear ones?
- How large is the non-secular leakage with real relaxation, that is, how wrong is dropping the part of `L` that
  leaves an order block?
- Do dense Hilbert-space simulations (`zeeman-hilb`) matter here? Sector blocking by total Fz would shrink blocks
  from 2^N to C(N, k).
- Should sequences take a pathway plan, or should users keep writing `coherence` filters themselves?

## A possible design (for later)

All opt-in; defaults and the parity suites unchanged.

- A pathway plan is plain data in the format `filters.coherence` already takes, one spec per period, for example
  `[[("1H", [0])], [("1H", [-1])], [("1H", [-1, 1])]]`. `ctps` results convert to it from `PathwayResult.orders`
  by a recipe in the guides, not by an import: `ctps` imports nothing from `spynach` and that stays.
- `filters.coherence_mask(spin_system, spec)` factored out of `coherence`, which calls it.
- A `subspace=` keyword (mask or index array) on `evolution` and `step`: slice `L`, `rho` and `coil` with
  `reduction._selector`, run the existing code, scatter back; optionally cache the sliced `L` on the mask.
- A `pathways` module: period masks, pulse masks (restrict only the isotopes the pulse does not act on, which is
  exact), per-isotope order windows for the basis (extending `bas['projections']`, which restricts the total
  order only), and a `leakage(L, mask)` check (norm of the part of `L` leaving the block over `||L||`) so that the
  secular assumption is measured, not trusted.
- Optional `parameters['pathways']` in `cosy`, `noesy` and `hsqc`, replacing their hand-written filters.

The restriction is exact when `L` commutes with the order projectors. The contract would be: the restricted
result equals the filtered full-space result to tolerance.

## If it is ever built

Per `CLAUDE.md`: user documentation in the same commit (a section in `docs/pulse-sequences.md`, the ctps recipe
in `docs/ctps.md`, an entry in `docs/coming-from-spinach.md` since this is an extension, not in Spinach),
notebooks rebuilt, a CHANGELOG line, unit tests (mask equals the `coherence` filter; a plan built from a ctps
result; restricted COSY and HSQC equal the full ones; `leakage` near zero for secular `L` and large for a
deliberate violator), and a timing case under `comparison/`.
