[← back to README](../README.md)

## Naming a thing instead of drawing a box: `grounding.py`

```python
from figsurgeon import grounding
out, note = grounding.recolour_subject(img, 'the sky', to_rgb=(230, 130, 180))
out, note = grounding.isolate_subject(img, 'the woman')      # keep it, grey the rest
mask, note = grounding.mask_for(img, 'the red car')          # or just the mask
```

The plain-language front-end routes to these automatically when the extra is installed, so
`apply_text(img, 'make the sky pink')` works; without it, that request is refused with the
install line rather than guessed at. About 0.6 s a call on CPU.

**CLIPSeg, not a detector plus SAM.** An alternative build is open-vocabulary detection
(Grounding DINO, OWLv2) for a box, then SAM for the mask inside it. Measured on eight real
phrases it is right about as often, costs 13 s a call against 0.6 s, needs two model
downloads instead of one, and cannot serve half the requests, because sky is not an object
with an outline. Asked for "the sky", both detectors return a box containing the whole
frame, rocket and gantry towers included, and SAM masks all of it. CLIPSeg goes from the
words to the region and excludes the rocket. Objects and stuff are the same kind of request
to a person; only one approach treats them that way. The cost is a softer edge —
`sharpen=True` re-cuts the mask's bounding box with SAM when that matters.

**Two guards against degenerate matches.** A phrase matching over 90 % of the frame is
refused: "keep the flowers in colour" on a macro where blooms fill the picture matches 95 %,
and the 5 % left over is the flower's own centre — a grey blob punched through an otherwise
untouched photograph. A phrase matching almost nothing is refused too, which is what "keep
the cat in colour" gets on a photograph of a pizza. `evals/grounding.py` holds both, and
writes pictures.

