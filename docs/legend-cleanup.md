[← back to README](../README.md)

## Cleaning a semi-transparent legend/annotation box

`advanced.clean_transparent_box(img, box)` handles both single-column and multi-column
legends. `box` is an approximate region, like `remove_object` — this function does not
locate the legend for you.

**Measured against ground truth.** `evals/legend_clean.py` renders the same legend over a
real published figure twice — at `framealpha=0.75` and at `1.0` — so the opaque render is
exactly what a correct clean should produce, pixel for pixel. Over dark content:

| | uncleaned | naive fill | current |
|---|---|---|---|
| scatter chart (light behind) | 3.7 | 2.2 | **1.8** |
| viridis heatmap | 51.1 | 14.3 | **5.2** |
| viridis contour plot | 43.1 | **158.0** | **8.3** |

(mean RGB error inside the box against the opaque render.) The third row shows what happens
when the fill colour is sampled as the modal untinted colour: in a bleeding legend the
untinted pixels are the text, since the fill itself is tinted by definition, so the box gets
flat-filled black — 3.7x worse than leaving it alone. The fill is sampled from light,
non-glyph pixels instead, and a box where no such pixel exists is refused rather than
guessed at.

Bleed-through touching a swatch can merge with it into one connected component whose mean
saturation falls below the glyph floor, erasing the swatch as if it were a ghost — 34 % and
31 % of swatch pixels surviving on the two figures above, reported as "legend box cleaned".
Components are found at the glyph level, so survival is 100 %.

**Measured across 91 ground-truth cases** — nine backgrounds, five fill colours, five
alphas, three text colours, four swatch sets, three legend sizes and two positions. Two
failure modes show up at this scale: bleed-through with no chroma at all (grey pavement,
black clothing, a dark logo) reads as ink and gets greyed rather than cleaned, leaving eight
cases with the error unchanged (59.6 → 59.0) under a "verified" verdict; and sampling the
fill as the commonest untinted colour picks up the background's own blend over a light
background rather than the legend (243 measured against a true 251). Ink is told from bleed
by stroke width — a letter is thin however large the font, a bleeding shape is solid — and
the fill is the lightest common untinted colour. Across the sweep: cleaning improves 54 → 63
cases, does nothing on 20 → 7, and total error falls 1240 → 1036.

`verify_photo.check_transparent_box_cleaned` measures the area of the legend's drawn glyphs
before and after — losing one is losing what the colours mean — and whether the faint tint
that is bleed-through actually dropped. It fails a lost swatch, fails a box that came out
darker than it went in, and gives no verdict when nothing measurable happened. Its residual
measure counts lightness as well as chroma; a chroma-only measure calls eight useless cleans
verified. Across the sweep, edits verified while being useless or harmful fall from 15 to 5.

**The separating signal is saturation, not raw distance from the fill colour.** A real glyph
swatch is drawn opaque, on top of the box; bleed-through is data colour blended through the
box's alpha. Distance from the fill colour's minimum channel is hue-biased: a pale colour
(pink, ~(230,133,200)) has a naturally high minimum channel even at full opacity, so it
scores lower than a saturated colour (navy blue) blended halfway to white — a real pink
"East" swatch drops below threshold on a 2x2 grid legend while a genuine blue bleed-through
blob scores higher. Saturation (max-min channel spread) avoids this: blending any hue toward
white reduces its spread proportionally, so opaque swatches across every hue tested (blue,
pink, cyan, red, orange) cluster comfortably above genuine bleed-through under one threshold.

**Real-vs-bleed disambiguation** combines two signals with OR: text-ink presence in a thin
strip immediately right of the candidate, and x-column clustering. Neither alone works: x-
clustering alone (true only for a single-column legend) protects background scatter dots
sharing an x-coordinate by coincidence while deleting real swatches in a 3-column legend;
a text-proximity radius alone over-protects almost everything in a compact box, since nearly
every pixel in a small dense legend is within a generous radius of some text.
`text_search_width` is a tunable parameter rather than a hardcoded 19px, because the real
swatch-to-text gap measures 93px on one chart and 25-40px on synthetic test charts at lower
resolution — font size and `handletextpad` scale with DPI in ways this function doesn't detect.

On a densely overlapping legend where a semi-transparent box (framealpha ~0.75) sits over
opaque-ish markers (alpha ~0.8), individual bleed-through pixels can retain ~60% of full
colour strength — close enough to a real swatch's that no local pixel-level heuristic
reliably tells them apart. Raising the saturation floor further still leaves dozens of
spurious candidate blobs, and morphological opening (to break merged blobs apart) destroys
the real swatch along with them. `tests/test_clean_transparent_box.py` locks in the two
cases that work correctly (single-column, and a realistic 2x2 multi-column grid); the
adversarial high-density case is disclosed as unreliable rather than silently mishandled.

