# Replacing GrabCut: what four segmenters actually do on eight boxes

Measured 2026-08-25, CPU only, no GPU anywhere in this. **The decision was since taken and
SlimSAM is wired in**, as one of two backends rather than a replacement: `objects.choose_backend`
sends photographs to SlimSAM and flat-colour figures to GrabCut, because on figures this
result reverses (`synthetic.py`, added after the table below). This directory stays as the
evidence, and `clipping.py` measures what the package's box guarantee costs a SAM mask.

`objects.segment_object` is GrabCut. The review calls segmentation-from-a-box commodity, and
CONTINUE.md records eight hand-drawn boxes where the masks were looked at. The question was
whether a SAM-class model fixes it, and whether it is fast enough to use.

## The answer

**`nielsr/slimsam-77-uniform` is the pick.** Apache-2.0, ~40 MB, about ten lines through
`transformers`. It has the best masks of everything tried, the smallest download, and its
speed is indistinguishable from the fastest.

| | GrabCut (ships now) | **SlimSAM-77** | SAM 2.1-tiny | SAM ViT-B |
|---|---|---|---|---|
| per call, warm, 0.1-1.5 MP | 0.7-10.8 s | **2.7-3.1 s** | 2.2-3.7 s | 7.1-8.3 s |
| flat with image size | **no** | yes | yes | yes |
| download | -- | **~40 MB** | ~150 MB | ~375 MB |
| licence | -- | Apache-2.0 | Apache-2.0 | Apache-2.0 |

## Read this before trusting any sheet: the first verdicts here were wrong

The first pass judged these masks from full-frame thumbnails and got two of them backwards.
It reported that SAM 2.1-tiny "kept the basil" on the pizza and "took the whole woman" on
the portrait. Measured:

- **basil kept**: GrabCut 99.9 %, SlimSAM 98.3 %, **SAM 2.1-tiny 77.8 %**, ViT-B 18.4 %
- **the hand gripping the umbrella**: SlimSAM 69.0 %, ViT-B 64.7 %, GrabCut 47.9 %,
  **SAM 2.1-tiny 5.9 %** -- it drops the hand entirely

Both are obvious once the panel is cropped to the box and invisible at full-frame scale.
The sheets in `sheets/` are cropped now, and every claim below has a number behind it. This
is the same failure the package's own history keeps recording: a verdict read off something
too small to carry the evidence.

## Per-case measurements

Each number expresses the specific request, in the style of `evals/real_world.py`'s
assertions. Bold is best.

| measurement (what it means) | GrabCut | SlimSAM | SAM2.1-tiny | ViT-B |
|---|---|---|---|---|
| astronaut: mask over the black helmet, lower better (it is not the suit) | 75.4 % | **10.4 %** | 15.7 % | 11.4 % |
| portrait: mask over the hand on the handle, higher better | 47.9 % | **69.0 %** | 5.9 % | 64.7 % |
| pizza: basil kept, higher better | **99.9 %** | 98.3 % | 77.8 % | 18.4 % |
| flower: fraction of the box rectangle filled, lower better | 90.0 % | 72.3 % | **64.1 %** | 70.5 % |
| coffee: connected components, lower is less speckle | **26** | 59 | 54 | 57 |

Read together: **SlimSAM is best or near-best on every case**. GrabCut wins two (the basil,
and the cleanest cup) and loses badly on the helmet and the flower. SAM 2.1-tiny is the
fastest and the least reliable of the three models -- it loses content. SAM ViT-B is 3x
slower than either and worse than both.

## Case by case, from the cropped sheets

| box | GrabCut | SlimSAM |
|---|---|---|
| astronaut suit | suit **+ the black helmet** (75 % of it) | the suit |
| rocket | **the sky** | **the sky -- all four fail** |
| chelsea cat | patch of cheek | patch of cheek (the box does not contain the cat) |
| coffee cup | cup + saucer + coffee, cleanest edges | cup + saucer, some speckle at the handle |
| red car, small in a street | the car | the car, **including the wheels below the box** |
| studio portrait | woman + **a wedge of blue umbrella** | woman, no blue left |
| pizza | pizza + **a wedge of white plate** | the pizza |
| flower centre | 90 % of the box rectangle | the seed head's shape |

**No model fixes a bad box.** All four take the sky on the rocket, because the box is mostly
sky. Swapping the segmenter fixes mask shape, not box choice.

**Bigger is not better.** SAM ViT-B is the largest and slowest and drops 82 % of the basil,
at score 0.81 -- confident and wrong.

## What it looks like as an edit, which is what a user sees

`demo_edits.py` runs the package's own compositing on each mask, so the difference shows up
where it matters. `edits/portrait_studio_isolate.png` is the clearest: "keep the woman in
colour" leaves a **blue wedge of umbrella** behind her with GrabCut and nothing with
SlimSAM.

## The cost argument, which is independent of all of the above

Same box, same photograph, three resolutions (`scaling.py`):

| | 0.4 MP | 1.5 MP | 4.3 MP |
|---|---|---|---|
| GrabCut | 1.6 s | 10.1 s | **27.9 s** |
| SlimSAM | 2.83 s | 2.78 s | **2.81 s** |

A SAM encoder always sees 1024x1024, so cost is flat in image size and in box area, and the
mask is the same at every scale. That removes the "box measured on a downscaled preview"
failure class, and it retires `workspace.SLOW_CALL_SECONDS`'s advice ("call resize first")
rather than rewording it.

## Two things that make this a decision, not a port

1. **A SAM mask legitimately extends outside the prompt box** -- the car's wheels, the
   astronaut's shoulder. `composite.enforce_region` guarantees nothing changes outside the
   box. Clipping is what GrabCut effectively does and it is why the car loses its wheels.
   Keeping the guarantee and keeping the object's true extent are not both possible.
2. **It costs a torch + torchvision dependency.** rembg already brings onnxruntime, so an
   ONNX route would add no runtime -- but do not hand-roll it. A community ONNX conversion
   of MobileSAM was tried first and abandoned: undocumented preprocessing, and every guess
   produced speckle rather than masks. What makes this a ten-line integration is the
   documented `transformers` processor doing the box coordinate transform. Export from a
   known checkpoint yourself if the dependency matters.

## Encode once, prompt many -- measured, and now what `segment_objects` does

`multibox.py`. The cost is the image encoder, and the encoder does not depend on the prompt,
so extra boxes on the same photograph are nearly free -- `segment_objects` used to re-encode
per box. Three people and a shop awning in a street photograph at 1400 px:

| boxes | one at a time | together |
|---|---|---|
| 1 | 3.5 s | 3.4 s |
| 2 | 6.3 s | 4.7 s |
| 4 | **14.9 s** | **4.3 s** |

Absolute timings on this machine wander by about a third between runs (the same four boxes
came back 6.6 s together in an earlier run), so the claim that does not depend on the
machine is the model's own share, within one run: **2.90 s for one box, 3.12 s for four**.
Flat, as the encoder argument predicts.

The saving is only worth having if the mask does not move, and it does not: over twelve
boxes on five photographs, ten batched masks are byte-identical to the one-at-a-time mask
and two differ by exactly one pixel -- a boundary logit summing either side of zero. Each
path is deterministic on repeat; it is the two against each other that differ, by a pixel.
`tests/test_segmentation_backend.py` holds that, on mask weight (feathering spreads the one
pixel over its neighbourhood, peak 0.26, nothing crossing 0.5), with the per-box path
monkeypatched to raise so the test cannot pass by quietly doing it the old way.

This is a saving for `recolour_object(boxes=[...])`, `isolate_object`, `erase_object` and
`extract_object` -- everything that goes through `segment_objects`. A single box pays
nothing extra and GrabCut is untouched: it has no shared work to share.

## The eight-box exercise, rerun on photographs nobody here had seen (2026-09-15)

`eight_boxes.py`, and the overlays it writes are committed in `sheets_eight_boxes/`. Eight
fresh Commons photographs, one box drawn per obvious subject off a coordinate grid, no
retries, `backend='auto'` -- which chose SlimSAM on all eight. **Seven of the eight masks
are the object**, against one of eight for GrabCut in 2026-08.

| case | what the mask took |
|---|---|
| German shepherd on grass | the dog, tail to front paws |
| kingfisher on a branch | the bird, open beak and both wings; the branch excluded |
| backpack on a grey sweep | the bag, both handles, the background between them resolved |
| motorcycle against a railing | the whole bike: spoked wheels, plate, mirror. Railing out |
| sailing boat at sunset | hull, mast, boom, most of the rigging, at 2 % of the frame |
| terrace chair | that chair; the second chair and the table left alone |
| grey horse in a field | the horse to the hooves, mane and tail included |
| **acoustic guitar on stage** | **the guitar plus the fretting hand and forearm** |

The guitar is the one failure, and no box fixes it: the arm lies across the neck, so every
box around the guitar contains it. Recoloured, the player's arm goes blue.

**What the rerun found is that the verdicts were the weaker half.** Each edit was graded by
`verify_photo.check_object_recoloured` with the caller's words: five correct passes, two
abstentions, and one false failure -- a correct recolour of the sailing boat reported as
"66 % of it kept its original colour". No wrong mask was called verified; the guitar
abstained, though for the wrong reason, since over-reach is not what a completeness measure
looks at.

**The false failure is fixed, and not with a threshold.** CLIPSeg sees a fixed 352x352
input, so a boat that is 2 % of the frame is a handful of pixels in what the model actually
reads, and a thin object at that size comes back as its filled silhouette -- sky and all,
6.1 % of the frame, of which only a quarter could ever change. Resolving the phrase on the
box plus context instead puts the object at a size the model can read: the region stops
swallowing the sky (44 % of the box -> 16 %), its solidity falls 0.88 -> 0.43 as it starts
tracing the rigging, and the reading goes **0.34 -> 0.61**, out of failed and into the
no-verdict band. Two other candidates were measured and dropped first: a solidity-gap rule
(one positive example, and four fresh skeletal objects failed to reproduce it -- a jetty
passes at 0.86 and a pylon's phrase region is less solid than its edit) and probability
weighting (boat 0.38 against a half-done 0.37: no separation at all). Verdicts on this set
now: six passes, two abstentions, no failures.

## Caveat on the 1-of-8 figure

These eight are not the eight in CONTINUE.md. Four of those images (sushi, whale, poster,
grasshopper) were never committed, so that benchmark is not reproducible from this repo. Of
the four that are, **GrabCut does better here than the note claims** -- the note records the
coffee box as "a blob, not the cup", and it comes back a clean cup and saucer, with the
fewest components of any method. Treat "one of eight" as unverified. The four added here
(`car_red`, `portrait_studio`, `food_pizza`, `flower_macro`) were drawn by hand off
`show_grid`.

## The give-up warning does not separate on SlimSAM (CONTINUE.md item 5)

`degenerate_masks.py`, 22 boxes on 10 photographs, every mask looked at and hand-labelled
`degenerate` (box has no distinct object; a give-up mask should say so) or `correct` (a
real object, whether or not it happens to fill the box) from the overlay -- 6 cases remain
ambiguous even on close inspection and are excluded from the two group ranges below, the
same way `evals/occlusion_phrases.py` excludes masks it judges wrong rather than scoring
them either way.

Four quantities taken from the returned mask were already measured and already fail to
separate (area fraction, IoU with the box, solidity, component count -- see the docstring
in `degenerate_masks.py`). What had not been tried is a signal that does not come from the
mask at all: whether the box's interior has any edge/structure to find, on the original
image, before segmentation runs. Two variants, plus two more not shown in the table below
because they fail the same way (Sobel-gradient and Laplacian-variance normalised by the
crop's own contrast; a 4x4-tile coefficient-of-variation "patchiness" measure) -- normalising
by contrast blows up on flat, low-noise crops, which is exactly the group it most needs to
separate.

| case | degenerate? | area_frac | edge_frac | grad_mean |
|---|---|---|---|---|
| nightcity_sky (sky gradient) | YES | 0.89 | **0.000** | 2.35 |
| shoe_blackbg (flat studio bg) | YES | 0.91 | **0.000** | 0.00 |
| chairs_water (flat dark water) | YES | 0.97 | **0.000** | 4.88 |
| chairs_wall (flat white wall) | YES | 0.97 | **0.000** | 4.71 |
| city_sky (sky gradient) | YES | 0.31 | 0.003 | 4.63 |
| shoe_carpet (repeating carpet) | YES | 0.85 | 0.087 | 33.89 |
| brick_wall (repeating brick, wholly bg) | YES | 0.66 | 0.055 | 37.86 |
| **facade_slats (repeating slats + plants)** | **YES** | 0.94 | **0.277** | **149.72** |
| **horse_grass (flat grass texture)** | **YES** | 0.72 | **0.340** | **131.74** |
| umbrella_flat (real, flat blue fabric, box-filling) | no | 0.89 | **0.008** | 5.97 |
| flower_petal (real petal edge) | no | 0.63 | 0.008 | 14.09 |
| city_railing (real railing, box-filling) | no | 0.99 | 0.245 | 59.87 |
| chairs_railing (real thin X railing) | no | 0.11 | 0.088 | 26.69 |
| flower_seedhead (real seed-head) | no | 0.72 | 0.155 | 47.80 |
| shoe_body (real shoe, low contrast) | no | 0.53 | 0.050 | 55.95 |
| facade_glass (real glass storefront) | no | 0.87 | 0.275 | 92.39 |

**degenerate group: edge_frac 0.000-0.340, grad_mean 0.00-149.72. correct group: edge_frac
0.008-0.275, grad_mean 5.97-92.39. The degenerate range fully contains the correct range on
both numbers -- no threshold separates them.**

Two counter-examples in each direction, found by going looking rather than by trusting an
early-looking sweep (CONTINUE.md's own standing rule):

- **False negatives (degenerate, but reads as "has structure"): `facade_slats`,
  `horse_grass`.** A repeating texture with no distinct object -- window slats and potted
  plants, a lawn -- generates dense local edges by itself. `horse_grass` has the highest
  edge_frac and grad_mean of any of the 22 boxes, degenerate or not.
- **False positive (correct, but reads as "no structure", same as a flat empty
  background): `umbrella_flat`.** A real, correctly-segmented, box-filling object that
  happens to be flat blue fabric reads edge_frac 0.008 -- indistinguishable from
  `chairs_wall`'s flat white wall (0.000) or `shoe_blackbg`'s flat black background
  (0.000), which are both give-ups. This is the same failure the shipped 0.95 coverage
  check has on the correct railing, for a different signal.

**Verdict: no. The signal does not separate, in either direction, and nothing else in the
"edge/gradient/texture-uniformity" family tried here does either** -- the fundamental
problem is that "has a distinct object" and "has local texture/edges" are different
properties: a real object can be flat (umbrella, railing) and a non-object can be textured
(grass, slatted facade with plants behind it). Per CONTINUE.md's standing rule, this is
reported rather than shipped: `figsurgeon/tools.py`'s SlimSAM path keeps the GrabCut-only
`coverage > 0.95` warning it already had (documented there as GrabCut's number, still true
for SlimSAM only by accident on the railing false-alarm case), and no new SlimSAM-specific
warning was added. Item 5 stays open.

## Not tested, and probably worth more than any of the above

- **Grounding: OWLv2, Grounding DINO, Florence-2.** Text -> box. The capability CONTINUE.md
  names as making most of the looking tools unnecessary, and the only thing that would fix
  the rocket. With SlimSAM behind it: "the red car" -> mask, end to end.
- **BiRefNet or RMBG-2.0 for the subject path.** `remove_background` and `blur_background`
  run U2Net through rembg, a 2020 model. A different job from box -> mask, and probably a
  bigger win for the tools people actually run.

## Reproducing

```bash
# masks: needs torch, in a throwaway environment (see run_models.py's docstring)
uv venv --python 3.11 /tmp/samenv
uv pip install --python /tmp/samenv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --python /tmp/samenv/bin/python transformers pillow numpy scikit-image
/tmp/samenv/bin/python evals/segmentation_models/run_models.py nielsr/slimsam-77-uniform slimsam
/tmp/samenv/bin/python evals/segmentation_models/run_models.py facebook/sam2.1-hiera-tiny sam2tiny
/tmp/samenv/bin/python evals/segmentation_models/run_models.py facebook/sam-vit-base samvitb

# sheets, edits and GrabCut: the project environment, no torch
.venv/bin/python evals/segmentation_models/compare.py slimsam sam2tiny samvitb
.venv/bin/python evals/segmentation_models/demo_edits.py slimsam
.venv/bin/python evals/segmentation_models/scaling.py
/tmp/samenv/bin/python evals/segmentation_models/scaling.py nielsr/slimsam-77-uniform
```

`masks_*/` hold the saved masks and timings, so everything except the masks themselves
rebuilds without torch. The corpus images download on first use, as the other evals do.
