[← back to README](../README.md)

## Known rough edges, honestly

- Background estimation is a per-column heuristic (frequency-then-lightest, with a
  neighbour-consistency pass for isolated contamination). It is not exact on every column —
  on the reference figure a handful of columns near a dash-dot marker differ from a
  hand-tuned build by single-digit RGB units. `verify.assert_clean()`'s `tolerance` param
  exists to absorb exactly this class of harmless disagreement; read the pixels before
  raising it further.
- `Bands` assumes full-height, column-constant, vertical spans drawn *behind* the lines.
  Horizontal spans, partial-height spans, or spans drawn *over* the lines are not handled.
- **A wrong box passes every check** — see `legend_wrong_box` in the eval. The looking
  tools exist because this is not fixable by a better check.
- `remap_colormap` matches colours against matplotlib's LUT for the named colormap, and a
  published figure's colormap is not always exactly that. When a band of values falls
  outside `match_tolerance` it survives in the old colours and the figure ends up carrying
  two colour scales; `check_colormap_remapped` fails the edit and names the band, but the
  remap itself has no way to recover those pixels. Measured at 3.1 % of the colormap content
  on a real `jet` figure.
- `clean_transparent_box` assumes the legend's fill is achromatic. A cream or pale-yellow
  legend has chroma of its own, never satisfies the untinted test, and is either refused or
  filled with the wrong colour. Two relaxed variants were measured against 91 ground-truth
  cases: both fix the tinted-legend cases but regress the common one — total error 1240 →
  2076 taking the fill's modal chroma, → 2245 taking the modal light colour — because on a
  white legend over a saturated chart most light pixels are bleed, and any rule that trusts
  them follows the bleed. Left as a known limit rather than traded for a regression
  everywhere else.
- `clean_transparent_box` needs something untinted inside the box to measure the legend's own
  fill colour from -- in practice the spots where a white gridline or contour passes behind
  it, which is 2-3 % of a real legend's box. Where uniformly saturated content sits behind
  the whole box, the fill is not recoverable from one observation and the call raises rather
  than guessing.
- The same function assumes a light box with dark text. A dark-themed legend is refused, not
  inverted.
- The same function tells legend text from grey bleed-through by stroke width (a letter
  mostly disappears after one erosion). Large legend fonts have strokes thick enough to fail
  that test: a synthetic scatter legend at font size 34 lost its text and two of its three
  swatches (at 24-26 it still clipped letters, though there the box also overshot the frame).
- `refine_box` snaps to whatever segmentation finds inside the padded box, which is not
  necessarily the object meant when two things share the box (the astronaut's suit box also
  catches a corner of a second helmet). It reports coverage so the caller can judge, and
  refuses rather than guessing when segmentation finds ~everything or ~nothing.
- `ImageWorkspace` history holds full images, capped at 25 steps (same reasoning as
  `ImageSession`: replaying is not reliable when some operations are non-deterministic).
- Multi-series `highlight()` composites survivors from a shared classification pass rather
  than re-running the engine per series, to avoid double-recompositing anti-aliased overlaps
  — but that means two kept series that directly overlap will show whichever was classified
  as owning that pixel, not a blend of both.

