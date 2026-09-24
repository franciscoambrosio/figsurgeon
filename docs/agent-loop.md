[← back to README](../README.md)

## Editing with a model driving it: look, edit, verify, undo

This is the layer the package is built for. A vision model can see the picture, which is
what makes it better than a regex parser at "the orange of her shirt" — but it is weak at
*absolute pixel coordinates*, and almost every tool here takes a box. Boxes drawn by eye
around each of the cat's eyes in `chelsea` (451x300) were both recognisably in the right
place and both visibly wrong, clipping the eye they were aiming at.

A wrong box is not the problem by itself: without a way to catch it, the next call commits
an irreversible edit to the wrong pixels and reports success. Three additions close that.

**1. Looking tools (`locate.py`) — they return an image to inspect and change nothing.**

| Tool | Answers |
|---|---|
| `show_grid` | "what are the actual coordinates here?" — a labelled pixel grid to read off |
| `preview_region` | "does my box contain the object?" — the box drawn on the image |
| `refine_box` | "can this box be snapped onto the object?" — and says so when it cannot |
| `preview_object_mask` / `preview_colour_mask` | "which pixels would this edit touch?" |
| `zoom` | "what does the edge of that edit look like?" — where the artifacts are |

![Four panels: grid overlay, box refinement, mask preview, and the resulting edit](images/looking_demo_grid.png)

Run `python examples/demo_looking.py`. The guessed box `(150,90,215,140)` snaps to `(138,87,206,146)`;
steps 1–3 leave the image untouched.

These tools rely on `dispatch`'s contract: it returns `(image, note)`, and the agent loop
assigns that image as the new working image, so a tool that returns an annotated copy must
not overwrite the picture. `ToolResult` (a tuple subclass, so every existing
`out, note = dispatch(...)` still works) adds `mutates` and `preview` to distinguish the two.

**2. `ImageWorkspace` — state, undo, and automatic verification.**

```python
from figsurgeon import ImageWorkspace
w = ImageWorkspace(img)
w.apply('show_grid', {})                                    # look; nothing recorded
w.apply('recolour_object', {'box': [...], 'to_rgb': [30,60,180]})
w.apply('undo', {})                                         # that box was wrong
print(w.summary())                                          # every edit + its verdict
```

`verify_photo.py`'s checks run against an offline eval suite; here, every edit is also
checked against that operation's own intent, and the verdict goes back in the tool note:

```
blurred background (focus=subject) | verified: subject detail retained 96%, background 19%
adjusted {'saturation': 1.8} | WARNING: this image is essentially greyscale …
  | CHECK FAILED: mean saturation 0.000 -> 0.000. The operation ran but did not do what it
  is meant to do. Call undo and try a different approach …
```

A failed check is not auto-reverted. Some checks are conservative and some edits are
stylistic; reverting on a heuristic verdict would silently throw away good work. It reports,
and `undo` makes acting on it cheap.

**3. Multiple regions in one call, graded one at a time.** "Colour pop the eyes" is two boxes.
`isolate_object(boxes=[...])` and `recolour_object(boxes=[...])` take a list. Calling the
tool twice is not equivalent for the isolating operations — the second pass desaturates
everything outside its own box, undoing the first. It is also faster: a SAM call's cost is
the image encoder, and the encoder does not depend on the prompt, so every box goes through
one forward pass. Three people and an awning in a street photograph at 1400 px: four boxes
cost 14.9 s one at a time against 4.3 s together, and the model's own share is flat in the
number of boxes (2.90 s for one, 3.12 s for four). The mask is the same mask either way —
over twelve boxes on five photographs, ten came back byte-identical and two differed by a
single pixel.

Each box gets its own verdict, because a pooled one is worse than none: three people
recoloured correctly came back as "only 55 % of what changed is inside 'the people walking'",
since the union of three boxes is mostly the pavement between them. Per box the same edit
reads 81 %, 90 %, and — for the third — *"the people walking IS in this picture, but
essentially none of it is in the region you changed"*, which is correct, because that man is
standing still. The phrase is still resolved once over all the boxes, so a plural request
keeps working: asked about a crop of one person, "the people walking" matches nothing.

### What this does not fix

A wrong box passes every check. `evals/edit_quality.py` keeps a scenario
(`legend_wrong_box`) that does exactly this on purpose: `clean_transparent_box` with a box
200 px above the legend punches a rectangular hole in real plot data, and the verdict is a
clean green, because the edit did stay inside the region it was given. Nothing measurable
distinguishes "edited the region you asked for" from "you asked for the wrong region"; only
looking does. That is the case for the looking tools, and why they cannot be replaced by a
better check.
