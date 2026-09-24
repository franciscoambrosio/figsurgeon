[← back to README](../README.md)

## The three layers

1. **`analyze.py`** — measure before editing: colour census, per-column background recovery,
   collinearity detection (series that share a base colour, like a translucent line and an
   opaque one of the same hue — no tolerance mask can tell them apart), band-assumption
   checks.
2. **`compose.py` + `spec.py`** — the engine. A `FigureSpec` declares one figure's geometry
   and series (see `specs/timeseries_demo.py` for a worked example). `recolour()`
   re-composites matched pixels at their recovered anti-aliasing coverage, rather than
   flattening them to a constant, so edges stay soft.
3. **`verify.py`** — invariant checks: nothing changes outside the axes, white stays white,
   every non-kept series actually disappears, the kept series survives untouched. Run this
   on every output, and diff every rebuild against the previous one
   (`report(..., previous=old_array)`).

## Describing edits in plain language

```python
from figsurgeon import load, apply_text
a = load('chart.png')
out, verb, kwargs, info = apply_text(a, spec, "highlight the MLP line")
out, verb, kwargs, info = apply_text(a, spec, "keep MLP and Offline, grey the rest")
out, verb, kwargs, info = apply_text(a, spec, "make the MLP line red")
```

`describe.py` maps a handful of common phrasings to three primitives in `edit.py`:
`highlight(keep=[...])`, `recolour_series(target, colour)`, `thicken(target, factor)`.

`describe.parse()` is a small keyword matcher, not a general NLP engine — it exists so this
also works headless, outside a chat. Inside an interactive session, the calling model (e.g.
Claude) is the real interpreter: it reads the `FigureSpec`, resolves "the red line" to
`linear` from context, and can call `edit.py` directly for anything the parser doesn't
cover. Extend `describe.py`'s patterns for your own common phrasings; it will not handle
everything a person might type.

`thicken()` has no stroke width in a raster PNG to change — only pixels already on the page.
Dilating the recovered mask approximates a thicker line well up to about 2x; beyond that, or
where a line crosses another protected annotation (a legend frame, a divider marker), the
result gets visibly blockier and should be checked. `nl_demo_grid.png` shows this case.

## Applying to a new figure

1. Run `analyze.census()`, `analyze.probe_interior()` on the new PNG. Confirm geometry by
   eye (`nl_demo` style crops); some figures have no axes spines at all, so don't trust
   auto-detection silently.
2. Write a new `FigureSpec` in `specs/`. Check `analyze.collinear_groups(spec)` — if any
   series share a colour, they need `alpha` set correctly or the spatial split will be wrong.
3. If the figure has shaded background spans, run `analyze.check_band_assumptions()` before
   turning on `Bands(...)` — it tests full-height, vertical, and behind-the-lines, the three
   things the neutralisation logic assumes.
4. Run `verify.report()` + `assert_clean()` on the output before trusting it.

