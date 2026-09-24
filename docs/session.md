[← back to README](../README.md)

## Talking to an image: `ImageSession`

Single instructions are one-shot and stateless, which is not how anyone actually edits:

```python
from figsurgeon import ImageSession
s = ImageSession(img)
s.do('blur the background')       # neural subject mask, background blurred
s.do('make it black and white')   # builds on the previous result
s.do('add a vignette')
s.do('more')                      # strengthens the VIGNETTE specifically
s.do('undo')                      # steps back one turn, exactly
s.do('reset')                     # back to the original
s.transcript()                    # what was applied, in order
```

Also handles compound instructions: `s.do('make it black and white and add a vignette')`
applies both.

Every step is verified against the operation's own intent, through the same table
`ImageWorkspace` uses (`workspace.verify_call`), so the typed path and the tool-calling path
cannot disagree about what an operation means:

```python
s.do('make it black and white')
s.last_check                      # (True, 'mean residual saturation 0.0000')
s.transcript()[-1]                # (...,  'grayscale({}) | verified: ...')

from figsurgeon import apply_photo_text as apply_text
out = apply_text(img, 'brighten it')
apply_text.last_check             # (True/False/None, what was measured)
```

`None` means no check applies, or the check could not decide — kept distinct from `False`
on purpose. On the twelve-photograph corpus, 141 correct instructions verify with no false
failures, and 82 of 84 deliberately broken results are caught; the rest report that there is
nothing measurable (`evals/text_path.py`, which writes a sheet for each).

Two semantics here are easy to get wrong:

- **"more" escalates in the user's original direction.** After "reduce the contrast", "more"
  means less contrast still. Factor-style parameters centre on 1.0, so naive multiplication
  would reverse the user's intent.
- **"less" replaces the previous edit rather than stacking a gentler one on top.** "increase
  the contrast" (std 74 → 93) followed by "less" applying contrast=1.27 to the
  already-boosted image would push std to 102 — the opposite of what was asked. "less" steps
  back to the pre-edit state and re-applies gently, which is the only reading that matches
  intent.

History stores images rather than a replayable command list: some operations are
non-deterministic in cost (neural segmentation) and some are lossy, so re-deriving step 3
from step 0 can differ from what the user actually approved. `tests/test_session.py` covers
these as regressions.

![A multi-turn ImageSession: black and white, then vignette, then more](images/conversation_demo.png)

