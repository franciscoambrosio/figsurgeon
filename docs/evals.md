[← back to README](../README.md)

## Eighteen evals, asking different questions

None of these are part of `pytest`; each writes pictures, and looking at them is the point.

| Eval | Question | Coverage |
|---|---|---|
| `evals/edit_quality.py` | is one operation's result any good? | 30 scenarios x 18 images |
| `evals/use_cases.py` | does a whole workflow hold together? | 13 workflows, 2-4 steps each |
| `evals/robustness.py` | what happens on input nothing was designed for? | 13 input shapes x 52 calls |
| `evals/real_world.py` | does any of this survive real photographs? | 17 requests on 12 Wikimedia images, timed |
| `evals/colour_spaces.py` | which colour space should select the pixels? | 2 constructed ground truths + 4 photographs |
| `evals/colour_space_defaults.py` | should the DEFAULT space change? | 5 constructed ground truths + 3 real charts + 4 photographs |
| `evals/text_path.py` | does the plain-language path get the VERDICT right? | 144 correct + 84 deliberately broken edits, both directions |
| `evals/grounding.py` | does a phrase resolve to the right region? | 8 phrases, 2 of which must FAIL; needs the `grounding` extra |
| `evals/segmentation_models/` | should GrabCut be replaced, and by what? | 4 segmenters x 8 hand-drawn boxes, + 4 constructed-truth figures; own `RESULTS.md` |
| `evals/object_completeness.py` | can "did ALL of the object change?" be measured? | 4 candidate measures x 6 photographs, both directions -- all four dropped |
| `evals/instance_phrases.py` | does the verdict survive "the middle one"? | 11 instances on 4 photographs of several-of-one-thing, both directions |
| `evals/object_verdicts.py` | did the RIGHT thing keep its colour, or get cut out? | 2 operations x 10 photographs x 3 variants, incl. the inversions that passed blind |
| `evals/occlusion_phrases.py` | does a thing behind a fence still get graded? | 10 cases on 6 photographs, both directions |
| `evals/recolour_vs_generative.py` | is a hosted image model a better recolour than the deterministic one? | 8 photographs x 4 arms, one shared mask, both directions for the structure measures |
| `evals/agent_loop/` | can a model driving this recover from its own mistakes? | 20 requests, unseen images, no human in the loop; pictures judged afterwards, and the driver's own account of the result judged against them |
| `evals/space_aware_checks.py` | does the check select pixels the way the OPERATION did? | 5 photographs x 2 spaces x 2 operations + 40 constructed failures |
| `evals/chart_data_drift.py` | does a hosted editor move the DATA when it recolours a series? | 7 synthetic charts with known values, read back in data units against their own axes |
| `evals/frontier_editors.py` | does a frontier model, given the whole picture, change only what was asked? | 2 charts + 4 photographs, drift / exact colour / chart data; judged by eye |
| `evals/segmentation_models/degenerate_masks.py` | how often does the segmenter hand back the box? | 22 boxes on 10 photographs, every mask looked at |

A single operation on a well-formed RGB photograph is the easy case; the later evals push
past it — into workflows, hostile input, real photographs, and colour spaces — because each
of those surfaces failures the easy case cannot.

### Whole workflows (`evals/use_cases.py`)

Real requests are three to six steps deep, and the failures that matter only appear in
sequence. Each workflow writes a *filmstrip* -- the original plus every intermediate state
-- because a single before/after cannot show which step went wrong.

Covered: an e-commerce cutout (cut out, white background, square crop, resize), a profile
photo, a low-light rescue, a colour-cast removal, a document straighten-and-sharpen, a
social crop with a caption, a product colour variant that is rejected and retried through
`undo`, an object cleanup, a poster treatment, a UI screenshot crop, and a reset.

### Hostile input (`evals/robustness.py`)

Every combination of 13 input shapes and 52 calls, asserting three things: nothing raises,
an error names its cause, and transparency survives. Inputs include palette / CMYK / 16-bit
/ 1-bit / greyscale modes, a 16x12 thumbnail, a 1200x40 panorama, a single pixel, a
flat-colour frame, and a real RGBA cutout. Calls include inverted boxes, out-of-bounds
boxes, floats where ints are expected, a 0..1 parameter given as 5.0, missing required
arguments, and a tool that does not exist.

## Quality eval across diverse images (`evals/edit_quality.py`)

`python evals/edit_quality.py [outdir]` runs 30 realistic scenarios across 18 images —
portraits, animals, still life, outdoor scenes, microscopy, greyscale photographs, charts,
a synthetic image built to defeat hue matching, and images with nothing for the machinery
to grab on to at all (a lawn with no subject, a near-black deep-field frame, flat vector
graphics, pure high-frequency texture) — and writes a captioned before/after sheet for
each, because "is the result any good" is not answerable from a number.

Scenarios declare what *should* happen, including failing:

- `grey_saturate` must FAIL and must warn (a greyscale photo has no colour to boost)
- `samehue_wrong_tool` must come back INCONCLUSIVE, not green
- `portrait_suit_blue` and `portrait_hue_trap` are a pair: the same request, the spatial
  tool preserving the flag's red stripes (0 channel movement) and the hue-based tool
  destroying them (112). The eval fails if *either* half stops behaving that way.

An eval whose every scenario is expected to pass measures only what already works — and
that includes the eval's own scenarios: `rocket_erase_mast` is checked with a no-op guard,
because its box segments nothing and the edit would otherwise do nothing while reporting
success. The issues below came out of these three evals.

## Issues these evals surface

### Silent no-ops that report success

`saturation` on a greyscale image is one known case; the same class shows up in three more
places, each capable of a confident green verdict for an operation that did not occur.

**Segmentation finding nothing.** On the rocket photo, a plausible-looking box that GrabCut
resolves nothing inside returns coverage 0.0, changes zero pixels, and reports
`object erased (0.0% of frame) | verified: 0 px changed outside the target region` — two
green statements about an operation that never happened. The localisation check cannot
catch this either: an image that changed nowhere trivially changed nothing outside its box.
All four object tools warn and leave the image untouched in this case.

**No subject to separate.** "Blur the background" of a lawn: U2Net finds an arbitrary speck
(0.1 % of the frame), the blur follows it, and the entire photograph is destroyed, while
verification agrees, because "subject detail retained 61 %, background detail retained
1 %" is exactly what a successful background blur looks like. The two populations are far
apart: 0.47–0.61 of the frame for a real subject (cat, astronaut, coffee cup), 0.000–0.004
for one with none (grass, brick, checkerboard). `blur_background`, `remove_background` and
`replace_background` report the coverage and refuse below 2 %.

**Denoising something that was not noisy.** A flat-colour image gives `np.corrcoef` a zero
standard deviation, returning nan; `nan > 0.9` is False, so a no-op reports CHECK FAILED.
Same shape on a 16x12 thumbnail, where the vignette check's corner sample is an empty
slice. Both cases are inconclusive instead — a verdict manufactured from arithmetic rather
than evidence is worse than no verdict.

### Transparency destroyed by the next step

"Remove the background, then brighten it" is an ordinary two-step request. It returns a
cutout silently flattened onto black, because nearly every function here begins with
`img.convert('RGB')`. The verdict reads "verified" — brightness went up, which is all that
check can see. `dispatch` splits the alpha off before an operation that does not understand
it and re-attaches it after; `rotate` fills a cutout's exposed corners transparent instead
of opaque white.

### Ordinary file formats

Real files arrive as palette PNGs, scanned `L` pages, CMYK print assets, 16-bit TIFFs.
`blur_background` raises "image has wrong mode" on P/1/I; `rotate` raises "color must be
int" on L/LA/1/I — both build a PIL call against the input's mode instead of converting
first, like everything else does. Normalising once at the dispatch boundary fixes the class
rather than the two instances.

### A missing system binary

`replace_font` wraps the tesseract binary, which pip cannot install. It raises
`TesseractNotFoundError` straight through `dispatch` on all 13 sweep inputs. A missing
dependency is not bad input and not a programming error: nothing the calling model does can
fix it, but it can route around it, provided it is told rather than disconnected.

### Operation order, which no model can derive from the tool list

The low-light workflow brightened first and ends noisier than it started (grain
5.40 → 13.76): brightening multiplies grain along with signal and no later denoise recovers
from it. Measured both ways on the same frame — denoise(18) then brighten(2.2) ends at
grain 2.45, the reverse at 5.37. This is stated in `denoise`'s tool description and the
system prompt, where the model driving the tools can act on it.

### A check that confounds its own inputs

`adjust` applies brightness, contrast and saturation in one call, and each is measured
against the combined result. On a scanned page, `brightness=1.05` with `contrast=1.9` moves
mean brightness 174.0 → 169.6 — raising contrast pushes the darks down further than the
small lift raises everything — so a correct, useful edit reports as failed. Combined
adjustments are inconclusive with the measurements reported; a lone adjustment keeps its
strict verdict, which is where the real signal is.

### Geometry changing mid-session

After a crop or resize, every box measured earlier points somewhere else. An out-of-range
box errors clearly; the dangerous one is the box still in range, which edits confidently
and silently in the wrong place. Any operation that changes the image's size says so.

**Feathering halves colour-pop strength.** `isolate_colour` and `replace_colour` soften
their mask with `gaussian_filter(mask, sigma=6)`. That blurs the edge and weakens the
interior, and for anything thinner than the kernel the second effect dominates:
colour-popping the warm lights on the rocket photo (thin bright shapes) holds the matched
pixels at mean mask strength 0.479, so a "keep this in full colour" operation delivers less
than half the colour it was asked for. Broad subjects — retinal vessels, a stained cell
field — are barely affected. Taking the maximum of the hard mask and its blur
(`photo._feather`) keeps selected pixels at full strength and lets the blur produce only the
outward falloff: saturation retained on the rocket goes from 41 % to 99 %.

**The spatial tools did not honour their box.** The same feathering spreads
`segment_object`'s mask outside the box the caller named — measured on `chelsea` with a box
around one eye: 1567 pixels outside the box carry mask weight, up to 0.40 strength, so
`recolour_object` visibly recolours pixels the caller had excluded. "Only what is inside
this box changes" is the reason the spatial tools exist alongside the hue-based ones.

**Re-theming a chart is slow.** `remap_colormap` matches pixels against the source colormap
with an all-pairs distance array: for a 633x249 plot area that is a 157k x 512 x 3 float
array, about 1.9 GB, and `detect_colormap_alpha` rebuilds it once per alpha step, 31 times.
A KD-tree gives the same answers — identical distances, and where the chosen index differs
it is a tie between two of the 256 duplicate LUT rows, worth 0.24/255 in the output — in
0.74 s instead of 106 s. Two minutes per call is not usable in an interactive loop.

**A verification that fails correct work.** `isolate_colour` called at tolerance 0.06 is
checked at a fixed 0.09, so every pixel in the 0.06–0.09 band counts as "should have been
kept" when the operation was right to exclude it. A visibly good edit comes back FAILED. A
check that cries wolf is worse than no check: it teaches the caller to stop reading the
verdict. Checks use the tolerance the operation was actually called with.

**A verification that passes vacuously.** `check_isolate_colour` substitutes "0% retained"
when there are no off-hue pixels to grey out. On an image whose subject and background
share a hue — where colour isolation provably cannot separate anything — it reports "target
hue retained 100%, other hues retained 0%" and a confident green. This case is inconclusive
instead, and names the tool that can do the job.

**A negative coordinate inverts a check.** `check_unchanged_outside` excludes its region
with `d[y0:y1, x0:x1]`; a box padded past the top edge gives `d[-8:188]`, which on a 427-row
image is rows 419..188 — an empty slice, so nothing is excluded and every changed pixel
counts as a violation, reporting 8536 px changed outside a region the operation had not left
at all. Regions reaching past the image edge are normal input, since callers pad boxes.

**`isolate_object` checked as a localised edit.** Desaturating everything outside the box is
its purpose; the unchanged-outside rule would report a correct edit as a 131,100-pixel
violation. Taking a box does not make an operation localised, so it has its own check.

**The agent loop required the SDK even with a client supplied.** Passing your own
`client` — a stub, a proxy, a different SDK — still raised `ImportError` for `anthropic`.
The loop now has nine tests, run offline against a stub.

