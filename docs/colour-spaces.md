[← back to README](../README.md)

## Which colour space selects the pixels: `space='hsv'` vs `space='oklab'`

`isolate_colour`, `replace_colour` and `preview_colour_mask` take `space`. The default
`'hsv'` is a hue window; `'oklab'` is distance in a perceptually uniform space:

```python
photo.isolate_colour(img, suit_colour)                       # hue window, tol 0.07
photo.isolate_colour(img, suit_colour, space='oklab')        # perceptual, tol 0.06
```

Hue ignores lightness and saturation entirely, so on a photograph a near-black pixel and a
vivid one are "the same colour" if their hues agree. Selecting the astronaut's orange suit
takes 62.7 % of the frame in HSV, in 2403 fragments — her face, her hair, the flag, the
warm background — against 25.5 % in 154 fragments in OKLab, each at its own default
tolerance. At the tightest hue window in the sweep, HSV still takes 24 %: no tolerance
fixes it, because the window is the wrong shape.

`evals/colour_spaces.py` measures this against ground truth that is constructed rather than
hand-drawn — the same matplotlib figure rendered twice with one series in two colours, so
the pixels that differ *are* that series, antialiasing and legend swatch included. Each
space is given its best tolerance from a sweep, because comparing two spaces at their
defaults measures the defaults:

| case (constructed ground truth) | HSV best IoU | OKLab best IoU |
|---|---|---|
| one flat series among three (chart) | 0.68 | **0.77** |
| shaded object + a same-hue dark distractor | 0.68 | **1.00** |
| white patch among grey and black | 0.00 | **1.00** |

A hue window is not the right tool for a chart's flat series colours: HSV's saturation
floor drops the antialiased edge of a line where it blends into the white page (those
pixels have median saturation 0.056 against an auto floor of 0.099), and OKLab keeps them.
On neutrals it is not close, because hue carries no information without chroma — "isolate
the white sneaker" is unservable by a hue window at any tolerance.

### The default: HSV

That table gives each space its best swept tolerance, which is the right way to compare two
spaces and the wrong way to choose a default: a default is used at its default tolerance,
by a caller who passed none. `evals/colour_space_defaults.py` runs the comparison that way
— same call, same target, one word changed — and the answer splits by domain:

| case (constructed ground truth) | HSV @ 0.07 | OKLab @ 0.06 |
|---|---|---|
| chart, one flat series among three | **0.68** | 0.56 |
| chart, a series drawn as a line **plus a translucent 95 % band** | **0.84** | 0.07 |
| chart, a **grey** series among coloured ones | 0.00 | 0.09 |
| shaded object + a same-hue dark distractor | 0.68 | **0.99** |
| white patch among grey and black | 0.00 | **0.50** |

The blocking case is translucency, a common chart idiom. A series drawn as a line plus its
confidence band is one colour to a reader and two to a distance metric: OKLab at 0.06 takes
the line and leaves the band untouched — visible in
`evals/out_colour_space_defaults/default_chart_band_oklab.png` (written by
`evals/colour_space_defaults.py`), where the band is plainly grey and the line plainly green. The same mechanism costs it the plain chart case, where what it
drops is the stretch of line running under the semi-transparent legend frame. Widening
`oklab_tol` does not fix this, it moves the hole: the band needs 0.08–0.10, the neutral
patch needs 0.02–0.04, and at 0.10 the shaded object falls back to 0.68. No single number
serves both domains, which is what having two spaces is for.

Swept over band opacity: at alpha 0.15 both spaces miss the band (HSV 0.10, OKLab 0.07 —
the band is below HSV's automatic saturation floor); at 0.4 and 0.6 HSV holds 0.84. At 0.25
it sits on a knife edge — the auto floor lands within 0.02 of the band's own saturation, and
adding a title to the same figure moves HSV between 0.83 and 0.09. The headline case is
quoted at 0.4 for that reason.

On photographs the flip is right — `edit_astronaut.png` and `edit_red_shoes*.png` in the
same directory are three-up sheets. HSV colour-pops her face, her hair, the flag and the
rocket, plus speckle across the background; OKLab keeps the suit. HSV stays the default,
and every tool description says to pass `space='oklab'` on a photograph.

The plain-language path depends on this too: `photo_describe` resolves a colour word to one
fixed RGB — "red" is `(220, 30, 30)` — because a word is all it has, and a hue window is
coarse in the same way a word is coarse. A perceptual distance of 0.06 is not: on 7 of the
12 corpus images, "colour pop the red" selects essentially nothing in OKLab (car 32.9 % →
0.06 %, pizza 31.3 % → 0.02 %). Running `evals/text_path.py` with the default flipped turns
21 of 141 correct edits into failures, every one of them a "colour pop the red" or a "change
red to blue", and the sheet for the red car shows the whole street greyed with the car
greyed along with it. `evals/edit_quality.py` moves too, though less: the rocket's warm
lights go from 99 % to 31 % of their saturation retained — OKLab keeps the flare cores and
greys the warm-lit gantry towers.

Neither space serves a neutral series at its default, which is worth knowing before reading
the 1.00 in the table above: that is OKLab at tolerance 0.02. On the real
`_corpus/chart_health.png`, asking for the grey bar series recoloured 45 % of the orange one
in HSV (hue is noise without chroma) and selected 92.7 % of the page in OKLab (the white
background is a short perceptual step from a light grey). Sample the colour, preview the
mask, and tighten `oklab_tolerance` to about 0.02.

Two defaults were set by measurement:

- `oklab_tol` is 0.06. At 0.12 (a perceptual distance of ~0.1 reads as "clearly different"),
  the mask ramp reaches zero at twice the tolerance, so 0.12 selected out to 0.24 and took
  64 % of the astronaut — no better than the window it replaces.
- `lightness_weight` is 0.25. It controls how much "the shadowed side of a red car is still
  the red car" counts. Swept, 0.25 has the best mean IoU across the three cases (0.92
  against 0.87 at 0.5). Lower is not better: the neutral case is in the eval precisely
  because every other case prefers 0.0, where white and black become the same colour and the
  selection takes the whole frame (IoU 0.16).

The eval also writes mask overlays for four real photographs, where there is no ground truth
and each space is tuned to select the same fraction of the frame, so "took less" cannot
masquerade as "was better". The numbers only rank — look at the pictures.

