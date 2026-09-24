# Audit: the legend cleaner and its check (2026-09-15)

**Corpus:** 91 ground-truth cases, against the 3 the code was built on. The method is the
valuable part and is reusable: render the same matplotlib legend over a real background
twice, at `framealpha < 1` and at `1.0`. The opaque render is exactly what a correct clean
should produce, pixel for pixel, so "did it work" is an error in RGB rather than an opinion.

Varied: 9 backgrounds (dark saturated heatmap, dark contour with white lines, light scatter,
mid-grey textured photo, busy street photo, food photo, solid-black product shot, night
photo, flat light UI), 5 fill colours, 5 alphas (0.5–0.95), 3 text colours, 4 swatch sets
including a pale pink and a dark navy, 3 legend sizes, 2 positions.

## What it found

**1. Bleed-through with no chroma was classified as ink.** "Neutral and dark" describes a
letter; it also describes grey pavement, black clothing or a dark logo showing through a
white legend. Those were greyed rather than cleaned. Measured on a street photograph at
alpha 0.75: that mask covered **73 % of the box in two blobs of 22,081 and 7,376 px**, against
real glyph components of 16–114 px. Eight cases came back with the error unchanged
(59.6 → 59.0) over the word "verified" — and the check's own ghost metric was chroma-only, so
it had the identical blind spot.

*Fixed.* Ink is told from bleed by stroke width: a letter is thin however large the font, a
bleeding shape is solid. Area was tried first and is the wrong measure — it depends on font
size and on how much text a legend carries.

**2. The fill was sampled as the commonest untinted colour**, which over a light background
is the background's own blend rather than the legend: 243 measured against a true 251.

*Fixed.* The fill is the lightest common untinted colour, which is what a legend carrying
dark text is.

## Result across the sweep

| | before | after |
|---|---|---|
| cleaning improves the image | 54 / 91 | **63 / 91** |
| cleaning does nothing | 20 | **7** |
| total error across all boxes | 1240 | **1036** |
| verified while useless or harmful | 15 | **5** |

## Still open

- **Chromatic legend fills.** Cream and pale yellow have chroma of their own, never satisfy
  the untinted test, and are either refused or filled with the wrong colour (error 22.6 → 51.5
  on one, reported verified). Two fixes were built and measured and both were dropped: taking
  the fill's modal chroma put total error at 2076, taking the modal light colour at 2245 —
  because on a white legend over a saturated chart most light pixels are bleed, so any rule
  that trusts them follows the bleed. Whatever solves this has to separate "the fill is
  chromatic" from "the fill is white and the bleed is chromatic", and neither of those two
  attempts did.
- **The 15 % light-fraction refusal threshold is unproven rather than validated.** No case in
  91 landed near it in either direction — the transition is a step (5.7 % → 22.1 % between
  alpha 0.60 and 0.65), so the specific value has never been tested.
- **The 0.85 swatch-area line was never exercised**: completed cases ran 98.2–100 %. The only
  sub-0.85 evidence is the pre-fix 33 % / 0 % cases that motivated it.
