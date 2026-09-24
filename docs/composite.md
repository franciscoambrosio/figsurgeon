[← back to README](../README.md)

## Keeping an edit inside its region: `compose_within`

"Only what is inside this box changes" is what the box-based tools are *for*, and it is the
one promise a generative editor cannot make: encoding an image to latents and decoding it
back leaves measurable error across the whole frame, not only where the edit was asked for.
`compose_within` makes that irrelevant — it takes the edited pixels from inside `region` and
copies the original bytes everywhere else, so the result is byte-identical outside `region`
no matter what the editor did.

```python
from figsurgeon import compose_within

edited = some_editor(img)                       # anything: a model, a filter, a service
out, report = compose_within(img, (60, 180, 420, 512), edited)
report['leaked_px']     # 133366 -- what the editor changed outside its region
report['leaked_max']    # 155/255 -- and by how much
```

`edited` may be the full frame or just the crop, so a crop-in/crop-out editor needs no
round trip. The leak is reported, not silently swallowed: an editor that changed 133 366
pixels outside the box it was given is misbehaving, and the caller is told even though the
composite has already made it harmless. The composite is a hard paste — feather inside the
region for a soft edge, because softening across the boundary would mean changing pixels
outside it, which is the one thing this function promises not to do.

Every localised tool goes through it. `dispatch` composites the result of
`recolour_object`, `erase_object`, `remove_object`, `clean_transparent_box` and
`remap_colormap` back into the region that operation is allowed to touch, and puts any leak
in the note:

```
object erased (1.3% of frame) | 227268 px outside the region this tool is allowed to touch
were reverted (up to 155/255)
```

"Allowed to touch" is the box plus that operation's *own* known reach — `erase_object`
dilates its mask by `grow` and inpaints a further `radius` around it, which is intended
behaviour, not a leak. That table lives in `composite.py` and is shared with the
after-the-fact check in `workspace.verify`, so enforcement and verification cannot disagree
about what the box meant. `isolate_object` is deliberately not localised: desaturating
everything *outside* the box is its entire purpose.

`tests/test_composite.py` checks all of this against deliberately leaky editors (a
whole-frame blur, and a ±1-everywhere stand-in for a latent round trip), including the two
checks that stop it from passing vacuously: the edit inside the region must survive intact,
and enforcement must leave a tool that was already correct byte-identical.

