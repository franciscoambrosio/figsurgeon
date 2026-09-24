"""Plain-language front-end for photo.py and advanced.py.

Rule order matters: every "keep colour X" pattern is tested before any global
desaturate/grayscale rule, since e.g. "make it black and white but leave the red" would
otherwise match the grayscale rule and destroy the colour asked to be kept.
tests/test_prompts.py locks that ordering in.

A keyword matcher, not general NLP -- it exists so common phrasings work headless, since
in a chat the calling model reads the image and calls photo.py/advanced.py directly.
"""
import re
from . import photo as P
from . import advanced as A

COLOUR_WORDS = {
    'red': (220, 30, 30), 'blue': (40, 70, 200), 'green': (40, 140, 40),
    'orange': (255, 140, 0), 'yellow': (230, 200, 30), 'purple': (140, 40, 170),
    'brown': (120, 80, 40), 'pink': (230, 130, 180), 'cyan': (40, 190, 200),
    'magenta': (220, 40, 200), 'teal': (30, 150, 150),
}

# Phrasings meaning "keep colour X, desaturate the rest", checked before the global
# grayscale rule. The last entry is loose and falls through on a non-colour capture.
_ISOLATE_PATTERNS = [
    r'(?:colou?r\s+)?pop\s+(?:the\s+)?(\w+)',
    # Optional verb avoids capturing just the verb ("leave" in "b&w but leave the red").
    r'(?:black and white|b&w|grey?scale|desaturated?)\s*(?:everything\s*)?'
    r'(?:but|except|apart from)\s+(?:leave\s+|keep\s+|retain\s+|for\s+)?(?:the\s+)?(\w+)',
    r'(?:keep|leave|retain)\s+(?:only\s+|just\s+)?(?:the\s+)?(\w+)\s+'
    r'(?:in\s+colou?r|colou?rful|coloured)',
    r'(?:grey|gray|desaturate)\s*(?:out)?\s*everything\s+(?:but|except)\s+(?:the\s+)?(\w+)',
    r'(?:only|just)\s+(?:the\s+)?(\w+)\s+(?:to\s+)?(?:should\s+)?(?:stay|remain|be)\s+',
    r'(?:keep|isolate)\s+(?:only\s+|just\s+)?(?:the\s+)?(\w+)\b',
]


# Words naming the whole picture, not a thing in it (else "make it blue" looks for "it").
_NOT_A_SUBJECT = {'it', 'this', 'that', 'them', 'these', 'image', 'photo', 'picture',
                  'whole image', 'the image', 'everything', 'all', 'colours', 'colors',
                  'colour', 'color', 'background', 'bg'}


_UNSERVED_RECOLOUR = (
    'Understood "recolour the {subject} to {colour}", but resolving which pixels are the '
    '{subject} needs a grounding model, which is an optional extra: install it with '
    'pip install "figsurgeon[grounding]". Without it, refusing beats guessing -- recolouring '
    'by hue instead would repaint every {colour}-ish pixel in the frame, not the {subject}. '
    'Or pass the region yourself: objects.recolour_object(img, box=(x0, y0, x1, y1), '
    'to_rgb=...).')


_UNSERVED_SUBJECT = (
    'Understood "keep the {subject} in colour, grey out the rest", but {subject!r} names a '
    'SUBJECT rather than a colour, and resolving which pixels are the {subject} needs '
    'a grounding model, which is an optional extra: install it with pip install '
    '"figsurgeon[grounding]". Without it, refusing beats guessing: falling '
    'through to plain grayscale would desaturate the whole image, which is the opposite of '
    'what was asked. Either name the colour to keep '
    '(photo.isolate_colour(img, (220, 30, 30))), or sample it from the subject '
    '(photo.sample_colour(img, box=(x0, y0, x1, y1))).')


def parse(text):
    """Return (fn, kwargs) for a plain-language instruction, or raise ValueError."""
    t = text.strip().lower()

    # ---- 1. selective colour isolation (precedes global grayscale/desaturate) ----
    for i, pat in enumerate(_ISOLATE_PATTERNS):
        m = re.search(pat, t)
        if not m:
            continue
        if m.group(1) in COLOUR_WORDS:
            return P.isolate_colour, {'colour_name_or_rgb': COLOUR_WORDS[m.group(1)]}
        if i < len(_ISOLATE_PATTERNS) - 1:
            # X is a subject, not a colour; falling through would desaturate the whole image.
            from . import grounding
            if grounding.available():
                # Keep the article: CLIPSeg reads "the flowers" differently from "flowers".
                return grounding.isolate_subject, {'phrase': f'the {m.group(1)}'}
            raise ValueError(_UNSERVED_SUBJECT.format(subject=m.group(1)))

    # ---- 1b. named-colour replacement: "replace/change/turn the red into orange" ----
    m = (re.search(r'(?:replace|change|turn|swap)\s+(?:the\s+)?(\w+)\s+'
                  r'(?:colou?r\s+)?(?:with|into|to|for|by)\s+(?:the\s+)?(\w+)', t)
         # passive voice: colour comes before the verb, unlike the active pattern above
         or re.search(r'(?:the\s+)?(\w+)\s+(?:colou?r\s+)?(?:be\s+)?replaced\s+'
                      r'(?:with|by)\s+(?:the\s+)?(\w+)', t))
    # A colour word can also start a region reference ("orange of her shirt"); what
    # follows the match settles it.
    is_region_ref = bool(m) and re.match(r'\s*(?:of|from|in|on)\b', t[m.end():])
    if m and m.group(1) in COLOUR_WORDS and m.group(2) in COLOUR_WORDS and not is_region_ref:
        return P.replace_colour, {'from_colour': COLOUR_WORDS[m.group(1)],
                                  'to_colour': COLOUR_WORDS[m.group(2)]}
    if m and m.group(1) in COLOUR_WORDS and (m.group(2) not in COLOUR_WORDS or is_region_ref):
        # The target is a region reference, not a named colour -- needs object detection.
        raise ValueError(
            f'Could not resolve target colour {m.group(2)!r}. If this refers to a region '
            f'of the photo ("the orange of her shirt") rather than a colour name, sample it '
            f'directly: photo.replace_colour(img, from_colour=..., '
            f'to_colour=photo.sample_colour(img, box=(x0,y0,x1,y1))).')

    # ---- 1c. recolour a named thing (after named-colour rules, so "red to blue" is a hue) ----
    m = re.search(r'(?:make|turn|paint|colou?r)\s+(?:the\s+)?'
                  r'([a-z]+(?:\s+[a-z]+)??)\s+(\w+)\b', t)
    if m and m.group(2) in COLOUR_WORDS and m.group(1) not in COLOUR_WORDS \
            and m.group(1) not in _NOT_A_SUBJECT:
        from . import grounding
        if grounding.available():
            return grounding.recolour_subject, {'phrase': f'the {m.group(1)}',
                                                'to_rgb': COLOUR_WORDS[m.group(2)]}
        raise ValueError(_UNSERVED_RECOLOUR.format(subject=m.group(1),
                                                   colour=m.group(2)))

    # ---- 2. background operations ----
    if re.search(r'(remove|delete|cut out|knock out|drop)\s+(?:the\s+)?(background|bg)\b', t) or \
       re.search(r'cut\s+out\s+the\s+(subject|person|object|foreground|cat|dog)', t) or \
       re.search(r'(background|bg)\s+(?:should be\s+)?(transparent|removed|gone)', t) or \
       re.search(r'make the background transparent', t):
        return A.remove_background, {}

    m = (re.search(r'(?:put|place|composite|stick)\b.*?\bon(?:to)?\s+a?\s*(\w+)\s+background', t)
         or re.search(r'replace the background with\s+(?:a\s+)?(\w+)', t)
         or re.search(r'(\w+)\s+background\s+instead', t)
         or re.search(r'change the background to\s+(?:a\s+)?(\w+)', t))
    if m:
        word = m.group(1)
        if word == 'transparent':
            return A.remove_background, {}
        colour = {'white': (255, 255, 255), 'black': (0, 0, 0),
                  'grey': (200, 200, 200), 'gray': (200, 200, 200),
                  'cream': (245, 235, 220), 'blue': (120, 170, 230),
                  'green': (120, 200, 130)}.get(word)
        if colour:
            return A.replace_background, {'new_background': colour}

    if 'blur' in t and ('background' in t or 'bg' in t):
        return P.blur_background, {'focus': 'subject'}

    # ---- 3. cleanup / correction ----
    if re.search(r'(remove|reduce|clean up|get rid of|kill)\b.*\b(noise|grain|speckle)', t) or \
       'denoise' in t:
        strength = 12 if re.search(r'\ba lot\b|heavy|very|lots', t) else 8
        return A.denoise, {'strength': strength}

    if re.search(r'white ?balance|colou?r cast|too (orange|blue|warm|cool|yellow|green)'
                 r'|looks? too (orange|blue|warm|cool)|neutralise|neutralize', t):
        return A.auto_white_balance, {}

    if re.search(r'(remove|erase|delete|get rid of|paint out)\b.*\b'
                 r'(object|thing|person|pole|wire|blemish|spot|mast)', t):
        raise ValueError(
            'Object removal needs to know WHICH pixels to remove. There is no reliable way '
            'to resolve "the object in the middle" from text alone without instance '
            'segmentation, so this is kept explicit rather than guessed: call '
            'advanced.remove_object(img, region=(x0,y0,x1,y1)), or pass a polyline/mask.')

    if re.search(r'sharpen|sharper|crisper|more detail', t):
        return P.adjust, {'sharpness': 2.0}

    # ---- 4. stylise ----
    if re.search(r'sepia|vintage|old photo|aged|nostalgic', t):
        return P.sepia, {}
    if re.search(r'sketch|pencil|line drawing', t):
        return P.sketch, {}
    if 'vignette' in t:
        strength = 0.7 if re.search(r'strong|heavy|dramatic', t) else \
                   0.35 if re.search(r'subtle|slight|light', t) else 0.55
        return P.vignette, {'strength': strength}

    # ---- 5. global tone (AFTER isolation, so "b&w except red" never lands here) ----
    if re.search(r'black and white|b&w|grey?scale|monochrome', t):
        return P.grayscale, {}

    bright = re.search(r'brighten|brighter|lighten|too dark|darken|darker', t)
    contrast = re.search(r'contrast|flat\b|punchy', t)
    sat = re.search(r'saturat|vivid|vibrant|muted|colou?rful|washed out', t)
    pop = re.search(r'make it pop|needs? to pop', t)
    if bright or contrast or sat or pop:
        kwargs = {}
        if bright:
            darker = re.search(r'darker|darken', t) and not re.search(r'too dark', t)
            kwargs['brightness'] = 0.75 if darker else 1.3
        if contrast:
            lower = re.search(r'reduce|less|lower|soften|flatten', t)
            kwargs['contrast'] = 0.7 if lower else 1.4
        if sat:
            lower = re.search(r'mut|desaturat|less|washed', t)
            kwargs['saturation'] = 0.4 if lower else 1.6
        if pop and not kwargs:
            kwargs = {'contrast': 1.3, 'saturation': 1.4}
        return P.adjust, kwargs

    # ---- 6. geometry ----
    m = re.search(r'rotate.*?(\d+)', t)
    if m:
        return P.crop_and_rotate, {'angle': int(m.group(1))}
    if re.search(r'\bflip\b|\bmirror\b', t):
        axis = 'vertical' if re.search(r'vertical|upside', t) else 'horizontal'
        return P.crop_and_rotate, {'flip': axis}
    if 'crop' in t:
        raise ValueError(
            'Crop needs an explicit box: photo.crop_and_rotate(img, box=(x0,y0,x1,y1)). '
            'Kept explicit rather than guessing what "the interesting part" means.')

    raise ValueError(f'Could not parse instruction: {text!r}')


def apply_text(img, text, verify=True, **overrides):
    """Parse and apply. Returns the edited image.

    Some underlying functions return (image, extra) -- unwrapped here so callers get a
    consistent type, with any warning exposed via `apply_text.last_warning`.

    The edit is also checked against the operation's own intent, via the same checks the
    tool-calling path uses, with the verdict left on `apply_text.last_check` as
    (ok, detail). Pass `verify=False` to skip it.
    """
    fn, kwargs = parse(text)
    kwargs.update(overrides)
    result = fn(img, **kwargs)
    apply_text.last_warning = None
    if isinstance(result, tuple):
        img_out, extra = result[0], result[1]
        if isinstance(extra, str):
            apply_text.last_warning = extra
    else:
        img_out = result
    apply_text.last_check = (None, 'verification not requested')
    if verify:
        from .workspace import verify_call
        apply_text.last_check = verify_call(fn, kwargs, img, img_out)
    return img_out


apply_text.last_warning = None
apply_text.last_check = (None, 'nothing applied yet')
