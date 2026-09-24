[← back to README](../README.md)

## Automated edit verification (`verify_photo.py` + `evals/eval_suite.py`)

> `ImageWorkspace` runs the relevant check after every edit and puts the verdict in the tool
> note, so the model driving the edit sees it at the moment it matters. See "Editing with a
> model driving it" above.

`evals/eval_suite.py` runs prompts across images and **measures whether each edit did what was
asked**, rather than only whether it ran: a check that only asks "did the image change" would
pass a "blur the background" that blurs the whole image, subject included, since that also
raises no error and visibly changes the image. Looking at the actual effect is what catches it.

Each checker encodes an operation's intent as a measurable property, e.g.
`check_background_blur` requires subject detail to be retained while background detail
drops. Fed a uniform blur it reports "subject detail retained 3%" and fails; fed a correct
background-only blur, 96% and passes.

Two things worth keeping in mind if you extend it:

- **A checker's own assumptions can produce false failures.** Sampling a fixed corner as
  "background" fails on a photo where the subject reaches into that corner (measured: 43 %
  subject in one case). Deriving the region from the same subject segmentation the operation
  uses avoids this.
- **`numpy.bool_` is neither `is True` nor `is False`.** Pass/fail/skip logic that branches on
  `ok is None` misclassifies a genuine failure as a skip if a checker returns `numpy.bool_`
  instead of `bool` -- a green suite hiding a red result. All checkers coerce to real `bool`.

The suite is adversarially tested: each checker is fed a deliberately wrong output (grayscale
where colour-isolation was asked, a darkened image where brightening was asked) and must
reject it, so the checks can't be vacuously green.

