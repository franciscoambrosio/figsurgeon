# Audit: phrase grounding and the shape check (2026-09-15)

**Corpus:** 46 (photograph, phrase) pairs — 12 corpus photographs, the 8 eight-box
photographs, and 4 newly fetched for the hard kinds (a bicycle, a cat, sheep behind a wire
fence, bare winter branches). Every phrase mask was rendered as an overlay and **labelled by
hand — GOOD / ROUGH / WRONG — before the number was read**. That labelling is in
`audit_phrase_labels.py` and is the part that cannot be regenerated.

Deliberately included: plural phrases, part-of-object phrases, thin/skeletal objects, one of
several identical things, occlusion, very small and very large objects, a phrase matching
nothing, and ambiguous phrases. 10 of the 46 were refused by `grounding.mask_for` itself
before any check ran, and all ten refusals were correct.

## What it found

| label | n | IoU range | mean |
|---|---|---|---|
| GOOD | 25 | **0.42 – 0.95** | 0.80 |
| ROUGH | 7 | 0.02 – 0.62 | 0.44 |
| WRONG | 4 | 0.13 – 0.78 | 0.45 |

**GOOD and WRONG overlap across a third of the scale.** On the 16 photographs the bands were
calibrated on — all single, compact, whole objects — they separated with margin +0.14. Add
plurals, parts and thin objects and the margin is **−0.36**. Three wrong verdicts:

1. *"the pedestrians"*, a correct three-person mask, **failed** at 0.42. A segmenter prompted
   with one box around all three cannot reproduce a disjoint mask.
2. *"the horse's tail"*, a correct thin part, **failed** at 0.52. The second opinion segments
   whole objects.
3. *"the fence"* → patches of grass, **passed** at 0.78. CLIPSeg missed the thin wire fence
   and picked a blob; SlimSAM agreed with that blob.

*Fixed for 1 and 2, by scope rather than by a new number.* Several separate regions, a
part-of-object phrase, and a skeletal region (under 40 % of its own convex hull; compact
objects measure 0.54–0.97, a boat that is mostly rigging 0.36) are declined, each saying
which it is. Sliding the bands instead would cost three correct verdicts to catch one wrong
one.

## Per-threshold verdicts

| threshold | supported by | verdict |
|---|---|---|
| `_CORROBORATED_SHAPE` / `_DISAGREES` | 46 pairs, margin **negative** | cannot be supported as a general rule; **scoped** instead |
| `_EXTENT_PAD = 0.3` | 15 (box, phrase) pairs, 0.02–0.99 of frame | keep, with a caveat |
| `_COMPLETE` / `_INCOMPLETE` / `_CORROBORATED` | 9 rigorous cases, reproduced | keep — margin +0.23 held up |

## Still open

- **Two models agreeing on the same wrong blob** has no fix found. Peak phrase confidence was
  measured as a discriminator and dropped: GOOD 0.68–0.97 against WRONG 0.69–0.93, fully
  overlapping. The passing verdict says out loud that agreement is not the same as the right
  object; that sentence is currently the only protection.
- **`_EXTENT_PAD` costs decisiveness in the 29–46 %-of-frame range.** Two correct complete
  recolours read COMPLETE at pad 0.0 and only no-verdict at 0.3. A smaller pad may recover
  them at no measured cost, but the sweep was not large enough to recommend changing it.
- **Two classes the completeness eval's 9 photographs do not cover**: occlusion, and one of
  several identical objects. Both degrade every check measured here; both deserve cases in
  `evals/object_completeness.py`.
