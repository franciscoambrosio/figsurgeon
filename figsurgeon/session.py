"""Conversational editing: talk to an image across turns instead of one-shot commands.

Handles "more", "less", "undo", and compound instructions ("do X and Y"), which need state
the stateless parser (`photo_describe.parse`) does not carry. History stores images, not
commands, since some operations are lossy or non-deterministic: undo must restore exactly
what was approved. Refinement re-applies the last operation to the current image with scaled
parameters, so repeated "more" compounds and stays undoable.
"""
import re
import copy

from .photo_describe import parse as _parse


# words that refine the previous instruction rather than starting a new one
_MORE = r'\b(more|stronger|harder|again|further|increase it|too little|not enough)\b'
_LESS = r'\b(less|weaker|softer|subtler|tone it down|too much|back off|dial it back)\b'
_UNDO = r'\b(undo|revert|go back|nope|never ?mind|scratch that|take that back)\b'
# "original" alone is not a reset: "keep the original red" is an edit. Only going BACK to it is.
_RESET = (r'\b(reset|start over|start again|from scratch)\b'
          r'|\b(back to|restore|return to|revert to)\s+(the\s+)?original\b')

# Factor params centre on 1.0 (1.4 = more, 0.7 = less); scale away from 1.0, not by naive multiply.
_FACTOR_PARAMS = {'brightness', 'contrast', 'saturation', 'sharpness'}
_MAGNITUDE_PARAMS = {'strength', 'radius', 'hue_tol', 'feather', 'flatten'}

# No scalable kwarg by default: isolate_colour's knob is flatten, replace_colour's is hue_tol.
_DEFAULT_KNOB = {
    'isolate_colour': ('flatten', 0.5),
    'replace_colour': ('hue_tol', 0.07),
}


def _intensify(kwargs, direction):
    """Return kwargs nudged stronger ('more') or gentler ('less')."""
    out = dict(kwargs)
    step = 1.5 if direction == 'more' else 1 / 1.5
    for k, v in list(out.items()):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if k in _FACTOR_PARAMS:
            out[k] = 1.0 + (v - 1.0) * step
        elif k in _MAGNITUDE_PARAMS:
            scaled = v * step
            if k in ('strength',) and scaled > 0.95:
                scaled = 0.95
            if k == 'flatten':
                scaled = min(scaled, 1.0)
            out[k] = type(v)(scaled) if isinstance(v, int) else scaled
    return out


def split_instructions(text):
    """Split a compound instruction ("do X and Y") into ordered steps, parsed independently.

    'and' also appears inside operation names ("black and white"), so those phrases are
    protected before splitting and restored after.
    """
    protected = text
    guards = {'black and white': '\x00BW\x00', 'salt and pepper': '\x00SP\x00'}
    for phrase, token in guards.items():
        protected = re.sub(phrase, token, protected, flags=re.I)
    parts = re.split(r'\bthen\b|\band then\b|,\s*(?:and\s+)?|\band\b|;', protected)
    out = []
    for p in parts:
        for phrase, token in guards.items():
            p = p.replace(token, phrase)
        p = p.strip()
        if p:
            out.append(p)
    return out


class ImageSession:
    """A conversation with one image.

        s = ImageSession(img)
        s.do('make it black and white')
        s.do('add a vignette')
        s.do('more')          # stronger vignette
        s.undo()
        s.image               # current state
    """

    def __init__(self, image, verify=True):
        self.original = image.convert('RGB')
        self.image = self.original
        self.history = []          # list of (instruction, image_before)
        self._last_call = None     # (fn, kwargs) for refinement
        self.log = []
        self.last_warning = None
        # Checked via `workspace.verify_call`; a miss here is compounded by the next refinement.
        self.verify = verify
        self.last_check = (None, 'nothing applied yet')

    def __repr__(self):
        return f'<ImageSession {self.image.size[0]}x{self.image.size[1]}, {len(self.history)} edits>'

    def transcript(self):
        return list(self.log)

    def do(self, text):
        """Apply one instruction (or several, if compound). Returns the current image."""
        steps = split_instructions(text)
        # a refinement/undo word alone is never a compound instruction
        if len(steps) > 1 and any(re.search(p, text, re.I) for p in (_UNDO, _RESET)):
            steps = [text]
        for step in steps:
            self._do_one(step)
        return self.image

    def _do_one(self, text):
        t = text.strip().lower()

        if re.search(_RESET, t):
            self.image = self.original
            self.history.clear()
            self._last_call = None
            self.log.append((text, 'reset to original'))
            return self.image

        if re.search(_UNDO, t):
            return self.undo()

        # refinement of the previous operation
        direction = 'more' if re.search(_MORE, t) else 'less' if re.search(_LESS, t) else None
        # only treat as pure refinement when there is no new operation named in the text
        if direction and self._last_call:
            try:
                _parse(t)
                is_pure_refinement = False
            except ValueError:
                is_pure_refinement = True
            if is_pure_refinement:
                fn, kwargs = self._last_call
                new_kwargs = _intensify(kwargs, direction)
                if new_kwargs == kwargs:
                    knob = _DEFAULT_KNOB.get(fn.__name__)
                    if knob:
                        key, default = knob
                        kwargs = dict(kwargs, **{key: default})
                        new_kwargs = _intensify(kwargs, direction)
                if new_kwargs == kwargs:
                    self.log.append((text, f'{fn.__name__} has no scalable parameter to adjust'))
                    return self.image
                # Replaces the last edit rather than stacking: stacking "less" on an
                # already-boosted image pushes further from the target, not back toward it.
                if self.history:
                    _, previous = self.history.pop()
                    self.image = previous
                self._apply(fn, new_kwargs, text, note=f'{direction} -> {new_kwargs}')
                return self.image

        try:
            fn, kwargs = _parse(t)
        except ValueError as e:
            if direction:
                self.log.append((text, 'nothing to refine yet'))
                raise ValueError(
                    f'{text!r} refines a previous edit, but no edit has been made yet.') from e
            raise
        # an explicit intensity word alongside a new operation scales it immediately
        if direction:
            kwargs = _intensify(kwargs, direction)
        self._apply(fn, kwargs, text)
        return self.image

    def _apply(self, fn, kwargs, text, note=None):
        before = self.image
        result = fn(before, **kwargs)
        self.last_warning = None
        if isinstance(result, tuple):
            img, extra = result[0], result[1]
            if isinstance(extra, str):
                self.last_warning = extra
        else:
            img = result
        # Verify before flattening: flattening destroys the transparency some checks measure.
        self.last_check = (None, 'verification disabled')
        if self.verify:
            from .workspace import verify_call
            self.last_check = verify_call(fn, kwargs, before, img)
        if img.mode == 'RGBA':                     # keep the session in RGB for chaining
            from PIL import Image
            flat = Image.new('RGB', img.size, (255, 255, 255))
            flat.paste(img, (0, 0), img)
            img = flat
        self.history.append((text, before))
        self.image = img
        self._last_call = (fn, copy.deepcopy(kwargs))
        entry = note or f'{fn.__name__}({kwargs})'
        ok, detail = self.last_check
        if ok is False:
            entry += f' | CHECK FAILED: {detail}'
        elif ok is True:
            entry += f' | verified: {detail}'
        elif detail and not detail.startswith('no automatic check'):
            entry += f' | check inconclusive: {detail}'
        self.log.append((text, entry))
        return img

    def undo(self):
        if not self.history:
            self.log.append(('undo', 'nothing to undo'))
            return self.image
        text, previous = self.history.pop()
        self.image = previous
        self._last_call = None      # refining after an undo would be ambiguous
        self.log.append(('undo', f'reverted {text!r}'))
        return self.image
