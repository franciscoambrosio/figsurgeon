"""Stateful editing with undo and automatic per-edit verification.

`tools.dispatch` executes one call and forgets it: no undo for the tool-calling path (only
`ImageSession` in session.py has that, and only for the regex parser), and no mid-edit check
against each operation's own intent (`verify_photo.py`'s checks otherwise only back an
offline eval suite).

`ImageWorkspace` closes both gaps. `apply()` runs the tool, verifies the result where a check
exists, records the previous state, and returns a note describing what was measured. A failed
check does not auto-revert -- some checks are conservative and some edits are stylistic -- it
reports, and `undo()` makes deciding cheap.
"""
import copy
import time


from . import advanced as A
from . import photo as P
from . import grounding as G
from . import verify_photo as V
from .composite import LOCALISED, allowed_region
from .tools import ToolResult, _rgb, dispatch

# Segmentation-backed tools; cost scales with pixel count. `scale_object` is included only
# for the slow-call warning below -- it has its own early-return branch in `verify()`.
SEGMENTATION_TOOLS = {'recolour_object', 'isolate_object', 'erase_object', 'extract_object',
                      'preview_object_mask', 'refine_box', 'scale_object'}
SLOW_CALL_SECONDS = 8.0

# Tools that inspect rather than edit -- never recorded in history, never verified.
LOOKING = {'show_grid', 'preview_region', 'refine_box', 'zoom', 'preview_colour_mask',
           'preview_object_mask', 'sample_colour'}

def _direction(factor):
    return 'up' if factor > 1.0 else 'down'


# Operations whose containment holds by construction; each gets its own early-return check
# in `verify()` rather than falling into the generic `LOCALISED` branch.
_BYPASSES_CONTAINMENT = {'generative_fill', 'scale_object'}


def verify(name, args, before, after, region=None, subject=None, mask=None):
    """Check an edit against the operation's own intent. Returns (ok, detail).

    `ok` is True/False, or None when no check applies or none can decide -- reporting "no
    check for crop" as a failure would train a caller to ignore the field.
    `subject` is what the caller said they were editing ("the suit"); unlike the box, it can
    say whether all of the object changed and whether the box found it at all.
    """
    args = args or {}
    # A model passes `subject` as a tool argument, so front-ends need no separate channel.
    subject = subject or args.get('subject')
    try:
        # Must use the tolerance the operation used, not a fixed default, or a correct
        # exclusion near the boundary gets marked a failure.
        if name == 'isolate_colour':
            # Space matters as much as tolerance: grading an OKLab isolation against a
            # hue-window check fails correct work and can steer toward the worse edit.
            return V.check_isolate_colour(before, after, _rgb(args['target_rgb']),
                                          hue_tol=args.get('hue_tolerance', 0.07),
                                          space=args.get('space', 'hsv'),
                                          oklab_tol=args.get('oklab_tolerance', 0.06))
        if name == 'replace_colour':
            return V.check_replace_colour(before, after, _rgb(args['from_rgb']),
                                          _rgb(args['to_rgb']),
                                          hue_tol=args.get('hue_tolerance', 0.07),
                                          space=args.get('space', 'hsv'),
                                          oklab_tol=args.get('oklab_tolerance', 0.06))
        if name == 'isolate_object':
            return V.check_isolate_object(before, after,
                                          args.get('boxes') or [args['box']],
                                          subject=subject)
        if name == 'extract_object':
            # Not localised -- transparency outside the object is the point -- so the check
            # reads the alpha channel as the mask instead.
            return V.check_object_extracted(before, after,
                                            args.get('box') or list(region or ()),
                                            subject=subject)
        if name == 'blur_background':
            return V.check_background_blur(before, after)
        if name == 'remove_background':
            return V.check_background_removed(before, after)
        if name == 'replace_background':
            return V.check_background_replaced(before, after, _rgb(args['colour_rgb']))
        if name == 'crop':
            return V.check_crop(before, after, tuple(args['box']))
        if name == 'denoise':
            return V.check_denoise(before, after)
        if name == 'stylise':
            effect = args.get('effect')
            if effect == 'grayscale':
                return V.check_grayscale(before, after)
            if effect == 'vignette':
                return V.check_vignette(before, after)
            return None, f'no check for the {effect!r} effect'
        if name == 'adjust':
            details, verdicts = [], []
            for key, checker in (('brightness', V.check_brightness),
                                 ('contrast', V.check_contrast),
                                 ('saturation', V.check_saturation)):
                factor = args.get(key, 1.0)
                if factor != 1.0:
                    # The requested factor, not just its direction, sets how much change to
                    # expect (see `verify_photo._check_adjust`).
                    ok, detail = checker(before, after, direction=_direction(factor),
                                         factor=float(factor))
                    verdicts.append(ok)
                    details.append(f'{key}: {detail}')
            if not verdicts:
                # `sharpness` is a real adjustment with no check behind it, so saying "no
                # adjustment requested" about a call that did adjust it would be false.
                unchecked = [k for k in ('sharpness',) if args.get(k, 1.0) != 1.0]
                if unchecked:
                    return None, (f'{", ".join(unchecked)} was adjusted; there is no check '
                                  f'for it, so this edit is unverified rather than verified')
                return None, 'no adjustment was requested; every factor was 1.0'
            if len(verdicts) > 1 and not all(v is True for v in verdicts):
                # Several adjustments in one call confound each other's checks, since each is
                # measured against the combined result. Not stated as fact; the measurements
                # are still reported.
                return None, ('; '.join(details) +
                              ' -- several adjustments were combined in one call, so each '
                              'measurement includes the others\' effects and no single '
                              'one can be judged. Apply them separately to verify one.')
            if any(v is False for v in verdicts):
                return False, '; '.join(details)
            if any(v is None for v in verdicts):
                # A check that could not decide is not a check that failed: collapsing the
                # two would report correct, unmeasurable work (e.g. below 8-bit rounding) as
                # a failure.
                return None, '; '.join(details)
            return True, '; '.join(details)
        if name == 'recolour_subject':
            return V.check_subject_recoloured(before, after, args.get('subject'),
                                              to_rgb=args.get('to_rgb'))
        if name == 'clean_transparent_box':
            # Containment alone is not enough: the two failures that actually happen -- a
            # swatch erased with the bleed-through, a box flat-filled with the colour of its
            # own text -- are both perfectly contained.
            ok, detail = V.check_transparent_box_cleaned(before, after, tuple(args['box']))
            inside, note = V.check_unchanged_outside(before, after,
                                                     allowed_region(name, args) or args['box'])
            if inside is False:
                return False, note
            return ok, detail
        if name == 'remap_colormap':
            # Containment alone would miss this: a band of values that silently keeps the
            # old colours is invisible to "nothing changed outside the box".
            box = tuple(args['box']) if args.get('box') else None
            ok, detail = V.check_colormap_remapped(
                before, after, box,
                source_cmap=args.get('source_cmap', 'viridis'),
                new_colours=tuple(args.get('new_colours', ('#00407A', '#52BDEC'))))
            if box is not None:
                # Containment only means something when the caller named a region to stay
                # inside. Re-theming the whole figure -- the default, and the right default
                # -- has no outside.
                inside, note = V.check_unchanged_outside(before, after,
                                                         allowed_region(name, args) or box)
                if inside is False:
                    return False, note
            return ok, detail
        # `name in _BYPASSES_CONTAINMENT` for both of the next two: before the generic
        # branch on purpose, which would otherwise pass them on containment alone.
        if name == 'generative_fill':
            # See `verify_photo.check_generative_fill`.
            return V.check_generative_fill(before, after, args['box'], mask=mask)
        if name == 'scale_object':
            # `tools.dispatch` clips the output to the region it reported, so containment
            # holds by construction; what varies is whether size changed by roughly `scale`.
            box = args.get('box')
            check_region = region or (tuple(int(v) for v in box) if box else None)
            if check_region is None:
                return None, 'no region given to check against'
            return V.check_object_scaled(before, after, check_region,
                                         float(args.get('scale', 1.0)))
        if name in LOCALISED:
            key = LOCALISED[name]
            box = args.get(key) if key else None
            if not box and region is None:
                return None, 'no region given to check against'
            if not box:
                # Prompted with points rather than a box: there is no caller-drawn
                # rectangle, and the region the operation reported is the whole contract.
                box = list(region)
            stated = region
            region = region or allowed_region(name, args)
            if region is None:
                return None, 'no usable region given to check against'
            ok, detail = V.check_unchanged_outside(before, after, region)
            if ok is False and stated is None and name in SEGMENTATION_TOOLS:
                # The likeliest cause is not a leak: a segmentation-backed edit follows the
                # object past the box, so checking against the bare box misreads a
                # deliberate continuation as a leak.
                detail += ('. NOTE: if this edit followed the object past its box, pass '
                           'the region the tool reported (ImageWorkspace does this for '
                           'you); checked against the box alone, the object\'s own '
                           'continuation counts as a leak')
            if ok is True and name in SEGMENTATION_TOOLS:
                # Containment is not identity: a segmentation-backed edit can be perfectly
                # contained while missing the object entirely (measured across several
                # hand-drawn boxes). No mask-quality or completeness check reliably tells a
                # good mask from a bad one, so the verdict stays open here rather than
                # reporting false confidence or crying wolf on a containment guarantee that
                # does hold. `subject`, drawn from different evidence than the box, closes
                # part of the gap when given; `erase_object` asks the reverse question via
                # `check_object_erased`.
                if name == 'recolour_object':
                    did, did_detail = V.check_object_recoloured(before, after, region,
                                                                 subject=subject,
                                                                 boxes=args.get('boxes'))
                elif name == 'erase_object':
                    did, did_detail = V.check_object_erased(before, after, region,
                                                             subject=subject)
                else:
                    did, did_detail = None, None
                if did is False:
                    return False, did_detail
                if did is True:
                    return True, '; '.join(filter(None, [detail, did_detail]))
                where = 'box' if args.get(key) else 'region'
                return None, ('; '.join(filter(None, [detail, did_detail])) +
                              (f'. Nothing leaked outside the {where} -- that is all this '
                               'confirms. Whether the segmentation found the OBJECT, and '
                               'ALL of it, is not checked and cannot be measured from the '
                               'output alone: call preview_object_mask and look.'
                               if not subject else ''))
            x0, y0, x1, y1 = (int(v) for v in box)
            if region != (x0, y0, x1, y1):
                detail += (f' (allowing {x0 - region[0]} px for this operation\'s own '
                           f'edge handling)')
            return ok, detail
    except (KeyError, ValueError, TypeError) as e:
        return None, f'check could not run ({type(e).__name__}: {e})'
    return None, 'no automatic check for this operation'


# `photo_describe.parse` returns a function and its kwarg names; `verify` above speaks tool
# names and `tools.dispatch`'s argument names. This table translates between them and lives
# here, next to `verify`, so the two cannot drift apart about what an operation means. Only
# operations with a check are listed; an unlisted one (sepia, sketch, white balance) comes
# back with no verdict rather than a false "verified".
_TEXT_OPERATIONS = {
    P.isolate_colour: ('isolate_colour',
                       {'colour_name_or_rgb': 'target_rgb', 'hue_tol': 'hue_tolerance'}, {}),
    P.replace_colour: ('replace_colour',
                       {'from_colour': 'from_rgb', 'to_colour': 'to_rgb',
                        'hue_tol': 'hue_tolerance'}, {}),
    P.blur_background: ('blur_background', {}, {}),
    # The grounded path: no box anywhere.
    G.recolour_subject: ('recolour_subject', {'phrase': 'subject', 'to_rgb': 'to_rgb'}, {}),
    A.remove_background: ('remove_background', {}, {}),
    A.replace_background: ('replace_background', {'new_background': 'colour_rgb'}, {}),
    A.denoise: ('denoise', {}, {}),
    P.grayscale: ('stylise', {}, {'effect': 'grayscale'}),
    P.vignette: ('stylise', {}, {'effect': 'vignette'}),
    P.adjust: ('adjust', {k: k for k in ('brightness', 'contrast', 'saturation',
                                         'sharpness')}, {}),
}


def as_tool_call(fn, kwargs):
    """Translate a (function, kwargs) pair from the text parser into (name, args).

    Returns (None, reason) when nothing here can check that operation.
    """
    kwargs = kwargs or {}
    if fn is P.crop_and_rotate:
        # One function, three operations; only the pure crop has a check.
        if kwargs.get('box') and not kwargs.get('angle') and not kwargs.get('flip'):
            return 'crop', {'box': kwargs['box']}
        return None, 'no automatic check for a rotation or a flip'
    entry = _TEXT_OPERATIONS.get(fn)
    if entry is None:
        return None, f'no automatic check for {getattr(fn, "__name__", fn)}'
    name, renames, extra = entry
    args = dict(extra)
    for key, value in kwargs.items():
        if key in renames:
            args[renames[key]] = value
    return name, args


def verify_call(fn, kwargs, before, after):
    """Verify an edit made through the plain-language path. Same (ok, detail) as `verify`.

    `ImageWorkspace` verifies every edit; the text front-ends (`photo_describe.apply_text`,
    `ImageSession.do`) return an image with no correctness signal of their own, though they
    can half-succeed in exactly the ways these checks catch.
    """
    name, args = as_tool_call(fn, kwargs)
    if name is None:
        return None, args
    return verify(name, args, before, after)


class Step:
    """One recorded edit, and the state it replaced."""

    def __init__(self, name, args, note, before, after, ok, detail):
        self.name, self.args, self.note = name, copy.deepcopy(args), note
        self.before, self.after = before, after
        self.ok, self.detail = ok, detail

    def __repr__(self):
        verdict = {True: 'verified', False: 'FAILED CHECK', None: 'unchecked'}[self.ok]
        return f'<{self.name} [{verdict}] {self.note[:60]}>'


class ImageWorkspace:
    """An image, its edit history, and a verified way to change it.

        w = ImageWorkspace(img)
        w.apply('show_grid', {})                       # look: image untouched
        w.apply('recolour_object', {'box': [...], 'to_rgb': [30,60,180]})
        w.undo()                                       # that box was wrong
        w.transcript()

    `apply` takes the same (name, args) as `tools.dispatch` and returns the same
    `ToolResult`, with the verification verdict appended to its note.
    """

    def __init__(self, image, autoverify=True, max_history=25):
        self.original = image.copy()
        self.image = image
        self.autoverify = autoverify
        self.max_history = max_history
        self.history = []      # list[Step], oldest first
        self.log = []          # list[(name, args, note)] including looking tools

    def __repr__(self):
        return (f'<ImageWorkspace {self.image.size[0]}x{self.image.size[1]}, '
                f'{len(self.history)} edits, {len(self.log)} calls>')

    def ask(self, text):
        """Apply one plain-language instruction. Returns whatever `apply` returns.

        `photo_describe.parse` reaches only some operations; `route_text` covers most of the
        rest by meaning. Two gaps are filled here rather than guessed: an object named by
        phrase has no rectangle until resolved via `grounding.region_for` (unmatched words
        are reported, nothing edited), and an operation missing an argument nobody can infer
        (e.g. which blues are "our brand blues") asks the caller instead of defaulting.
        """
        from . import grounding
        routed = route_text(text)
        if routed is None:
            raise ValueError(
                f'Could not turn {text!r} into an operation this package performs. '
                f'Nothing was changed.')
        name, args = routed
        args = dict(args or {})
        subject = args.get('subject')
        if name in NEEDS_A_REGION and subject and not args.get('box') and not args.get('boxes'):
            if not grounding.available():
                raise ValueError(
                    f'"{text}" names {subject!r}, and finding it needs the grounding extra: '
                    f'pip install "figsurgeon[grounding]". Or pass a box yourself.')
            box = grounding.region_for(self.image, subject)
            if box is None:
                return self._refuse(f'{subject!r} does not match anything in this picture, '
                                    f'so there is nothing to edit. Nothing was changed.')
            args['box'] = list(box)
        if name == 'remap_colormap' and not args.get('new_colours'):
            return self._refuse(
                'which colours? Name them as hex stops -- for example new_colours='
                '["#00407A", "#52BDEC"] -- because a brand palette is not something this '
                'can infer. Nothing was changed.')
        return self.apply(name, args)

    def _refuse(self, why):
        """Say no without touching the image, in the shape `apply` returns."""
        from .tools import ToolResult
        return ToolResult(self.image, 'refused: ' + why, mutates=False)

    def apply(self, name, args=None):
        """Run one tool call against the current image."""
        if name == 'undo':
            return self.undo(int((args or {}).get('steps', 1)))
        if name == 'reset':
            return self.reset()

        before = self.image
        started = time.time()
        result = dispatch(name, args, before)
        elapsed = time.time() - started
        self.log.append((name, copy.deepcopy(args), result.note))

        if elapsed > SLOW_CALL_SECONDS and name in SEGMENTATION_TOOLS:
            # The remedy depends on which segmenter ran: GrabCut's cost scales with pixel
            # count, so resizing helps; a SAM backend's cost and mask are both nearly flat
            # across sizes, so resizing there buys nothing.
            from . import objects as O
            megapixels = (before.size[0] * before.size[1]) / 1e6
            remedy = ('Cost is nearly flat in image size on this backend and the mask is '
                      'the same at every scale, so resizing will not make it faster.'
                      if O.choose_backend(before) == 'slimsam' else
                      'Segmentation cost scales with pixel count on this backend. If more '
                      'object edits are coming, call resize first (e.g. scale 0.5) and work '
                      'at that size.')
            result = ToolResult(
                result.image,
                result.note + (
                    f' | NOTE: this took {elapsed:.0f}s on a {megapixels:.1f} MP image. '
                    + remedy),
                preview=result.preview, mutates=result.mutates, data=result.data)

        if not result.mutates or name in LOOKING:
            # Inspection, or a reported error: no new state, so nothing goes on the undo
            # stack (an undo stack full of no-ops would make "undo" mean less each time).
            return result

        ok, detail = (None, 'verification disabled')
        if self.autoverify:
            ok, detail = verify(name, args, before, result.image,
                                region=(result.data or {}).get('region'),
                                mask=(result.data or {}).get('mask'))

        self.image = result.image
        step = Step(name, args, result.note, before, result.image, ok, detail)
        self.history.append(step)
        if len(self.history) > self.max_history:
            # Bound the memory: history holds full images (see session.py).
            self.history.pop(0)

        note = result.note
        if result.image.size != before.size:
            # The dangerous case is not the out-of-range box (that errors clearly) but one
            # still in range after a crop or resize: it edits confidently in the wrong place.
            note += (f' | NOTE: the image is now {result.image.size[0]}x'
                     f'{result.image.size[1]} (was {before.size[0]}x{before.size[1]}). Any '
                     f'box coordinates measured before this call are no longer valid -- '
                     f'call show_grid again before using one.')
        if ok is False:
            note += (f' | CHECK FAILED: {detail}. The operation ran but did not do what it '
                     f'is meant to do. Call undo and try a different approach rather than '
                     f'stacking another edit on top of this one.')
        elif ok is True:
            note += f' | verified: {detail}'
        elif detail and not detail.startswith('no automatic check'):
            note += f' | check inconclusive: {detail}'
        return ToolResult(result.image, note, preview=result.preview,
                          mutates=True, data={**result.data, 'verified': ok})

    def undo(self, steps=1):
        if not self.history:
            return ToolResult(self.image, 'nothing to undo', mutates=False)
        n = 0
        for _ in range(max(1, int(steps))):
            if not self.history:
                break
            step = self.history.pop()
            self.image = step.before
            n += 1
        names = 'edit' if n == 1 else 'edits'
        note = f'undid {n} {names}; back to the state before {step.name!r}'
        self.log.append(('undo', {'steps': n}, note))
        return ToolResult(self.image, note, mutates=False)

    def reset(self):
        self.image = self.original
        self.history.clear()
        self.log.append(('reset', {}, 'reset to the original image'))
        return ToolResult(self.image, 'reset to the original image', mutates=False)

    def transcript(self):
        """Every call made, in order, with what it reported."""
        return list(self.log)

    def failures(self):
        """Recorded edits whose verification check failed."""
        return [s for s in self.history if s.ok is False]

    def summary(self):
        lines = []
        for i, s in enumerate(self.history, 1):
            verdict = {True: 'ok', False: 'FAILED', None: '--'}[s.ok]
            lines.append(f'{i}. {s.name} [{verdict}] {s.note}')
        return '\n'.join(lines) or '(no edits applied)'


# Operations that work on a region but are asked for by name; `grounding.region_for` bridges
# the phrase to a box. `recolour_object` goes through a box (rather than `recolour_subject`)
# so the mask comes from a model reading pixels, corroborating the phrase.
NEEDS_A_REGION = {'recolour_object', 'erase_object', 'isolate_object', 'extract_object',
                  'preview_object_mask', 'remove_object'}


def route_text(text):
    """A sentence to (tool, args), by the exact rules first and the router for the rest.

    The keyword rules in `photo_describe` are tried first because they encode specific,
    tested knowledge and are right by construction where they match; the router only knows
    which operation sounds nearest. Returns None when neither can serve the sentence -- a
    refusal, not a wrong edit.
    """
    from . import intent
    from .photo_describe import parse
    try:
        name, args = as_tool_call(*parse(text))
        if name is not None:
            return name, args
    except ValueError:
        # The parser's "not mine". Anything else is a bug and propagates, rather than being
        # quietly handed to the router as if the sentence had merely not matched.
        pass
    return intent.resolve(text) if intent.available() else None
