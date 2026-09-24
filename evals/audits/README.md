# Threshold audits: what each number in this package was measured on

Every check in `verify_photo.py` and every heuristic in `rebrand.py` / `advanced.py` carries
a threshold, and each was chosen from whatever cases existed when it was written — between 2
and 16 of them. On 2026-09-15 three audits ran against real corpora, each instructed to
measure and not to change shipped code. All three found wrong answers in code that passed
its own tests.

**Read this before touching a threshold.** The question to ask of any number here is: *what
set was this measured on, and would it hold on a layout I have not seen?* The answer belongs
next to the number, and where it is missing, that is the next thing to measure.

| audit | corpus | wrong answers found | fixed | still open |
|---|---|---|---|---|
| [colormap remap](REMAP.md) | 27 published figures | 5 | 2 defects | value drift on small-marker scatter (n=1) |
| [legend cleaner](LEGEND.md) | 91 ground-truth renders | 10 | 2 defects | chromatic legend fills (cream, pale yellow) |
| [phrase grounding](PHRASE.md) | 46 labelled (photo, phrase) pairs | 3 | 2 of 3, by scoping | two models agreeing on the same wrong blob |

## The instruments

Each audit's scripts are here, prefixed by which one they belong to. They were written to
run against a scratchpad and need their output paths pointed somewhere real before reuse;
what they are kept for is the method, which is reusable and was the expensive part.

- `audit_legend_sweep.py` — the most valuable of the three. Renders the same matplotlib
  legend over a real background twice, at `framealpha<1` and at `1.0`, so the opaque render
  is exactly the correct answer and error is RGB rather than opinion. Sweeps backgrounds,
  fill colours, alphas, text colours, swatch sets, sizes and positions.
- `audit_remap_fetch.py` / `_identify.py` / `_measure.py` / `_overlay.py` — fetches figures
  from Commons, fits each candidate matplotlib colormap against the figure's own pixels to
  identify it by number rather than by eye, runs the shipped remap and check, and paints the
  check's own leftover mask back onto the output so its spatial pattern can be judged.
- `audit_phrase_cases.py` / `_measure_iou.py` / `_pad_sweep.py` / `_labels.py` — the 46
  (photograph, phrase) pairs, the IoU measurement, the `_EXTENT_PAD` sweep, and the hand
  labels. `_labels.py` is the part that cannot be regenerated: someone looked at every mask
  and wrote down GOOD / ROUGH / WRONG before reading the number.

## The lesson that generalises

Three of the four fixes written earlier the same day were themselves fitted to their
examples, and only the corpora caught them:

- the colormap leftover measure, written against two figures with gridlines, traced contour
  lines on plots that had them;
- the edge pass assumed overlays were white or black, true of the two figures it was built
  on, and left grey and magenta ones untouched;
- the outside-box note rested on one positive and one negative and fired on 3 of 5 figures
  that had no colorbar at all.

And one test fixture was wrong in exactly the dimension under test — it painted 100×14 solid
rectangles and called them text — which failed working code until it was fixed.
