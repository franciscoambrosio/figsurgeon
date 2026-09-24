# Audit: the colormap remap and its check (2026-09-15)

**Corpus:** 27 figures fetched independently from Wikimedia Commons, 29 test cases. Cached
under `evals/_corpus/audit_remap/` (gitignored, credited in its `CREDITS.txt`). Colormaps
were identified by fitting each candidate against the figure's own pixels, not by eye:
viridis, jet (MATLAB's, close to matplotlib's), turbo, inferno, cividis and afmhot all turned
up with tight fits; plasma, magma and Greys did not appear in any real chart found, so this
audit says nothing about those three.

Categories covered: heatmaps, contour plots, spectrograms, scatter-with-colorbar, maps,
multi-panel figures, and six cases whose scale is not any matplotlib colormap (a custom
geoprofiling scale, an NWS categorical radar key, hand-made scales, and two mislabelled
diverging maps).

## What it found

**1. The leftover measure traced contour lines.** Their anti-aliased edges are colourful,
unchanged and near the source colormap — every condition the measure tests. Contaminated on
4 of 19 primary cases, all filled-contour plots, and on one it flipped the verdict: a clean
turbo re-theme reported *"a band of values survived (around 0.00–0.70)"*, the nonsense range
itself a symptom of tracing lines across the whole scale.

*Fixed.* A line trace is one or two pixels wide and does not survive erosion; a band does.
Measured: contaminated cases 1.6 / 2.0 / 2.8 % → 0.01 / 0.27 / 0.00 %, true survivors
3.2 → 2.6, 2.3 → 1.7, 6.0 → 3.0.

**2. The outside-box note fired on figures with no colorbar.** 3 of 5 boxed stress tests —
a chart with no colormap anywhere, a custom magenta/cyan scale, a categorical radar key —
each cleared the 5/20 spread gate by coincidence. That gate was fitted on one positive and
one negative, and the negative was a single-colour artifact, so a multi-colour non-matplotlib
scale was never tested.

*Fixed.* A colorbar or a left-out panel is one connected region: largest component 1.00 and
0.96 on the two true cases, 0.01–0.38 on the three false ones. The note now requires half the
survivors in one region and reports that share.

## Per-threshold verdicts

| threshold | supported by | verdict |
|---|---|---|
| `_OLD_SCALE_OK = 0.005` | true passes 0.00–0.27 % across 27 figures | keep, comfortable margin |
| `_OLD_SCALE_BAD` | true survivors 1.7–3.0 % after the erosion fix | **changed 0.02 → 0.015**, now on 27 figures |
| `_VALUE_DRIFT = 0.02` | passes 0.000–0.006; wrong-colormap cases 0.000–0.477 | keep for the main boundary |
| leftover definition | 4/19 contaminated before the fix | **changed** (erosion) |
| outside-box gates | 3/5 false positives before the fix | **changed** (one-region requirement) |

## Still open

- **Value drift on small-marker scatter plots.** One correctly-named viridis scatter read
  drift 0.115 — far above the 0.02 line — from a small matched population dominated by
  marker-edge fringe. Visual inspection says it is not a real failure. n=1: cannot be decided,
  and not worth a rule until a second case exists.
- **"Looks like viridis" is not "is matplotlib's viridis".** Two real human-made charts that
  a person would call viridis sit 35–52 away from the LUT, outside `match_tolerance=15`. The
  check correctly refuses them, but a caller who trusts their eyes will not understand why.
  A "which colormap is this actually?" helper (the audit's `audit_remap_identify.py` method)
  would answer it, and does not exist in the package.
- **plasma, magma, Greys** were never exercised on a real figure.
