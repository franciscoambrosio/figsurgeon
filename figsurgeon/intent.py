"""Which operation a sentence asks for, decided by meaning rather than keywords.

`photo_describe.parse` is a keyword matcher covering roughly half of this package's
operations; this module routes the rest by embedding the sentence and comparing it with
example phrasings of each operation (a small sentence-transformer, not a generative model,
so it can only ever return an operation that exists).

Colours, colormap names, and the subject of an object edit are read from the sentence as
written, not guessed. `tools.md` is the catalogue this module compares against, and the
source of its examples.

`resolve` returns None when the nearest operation isn't near enough, or two are too close
to call.
"""
import os
import re

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
_CACHE = {}

# Chosen by leave-one-out cross-validation over EXAMPLES (see evals/text_coverage.py).
FLOOR = 0.42        # nearest operation must score at least this
MARGIN = 0.015      # and beat the runner-up by at least this

CATALOGUE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools.md')


def catalogue(path=CATALOGUE):
    """Read `tools.md` into {operation: {'does': str, 'needs': str, 'examples': [str]}}."""
    if 'catalogue' in _CACHE:
        return _CACHE['catalogue']
    entries, name = {}, None
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('## '):
                name = line[3:].strip()
                entries[name] = {'does': '', 'needs': '', 'examples': []}
            elif name and line.startswith('**Does:**'):
                entries[name]['does'] = line.split('**Does:**', 1)[1].strip()
            elif name and line.startswith('**Needs:**'):
                entries[name]['needs'] = line.split('**Needs:**', 1)[1].strip()
            elif name and line.startswith('- '):
                entries[name]['examples'].append(line[2:].strip())
    if not entries:
        raise ValueError(f'no operations found in {path}')
    _CACHE['catalogue'] = entries
    return entries


def EXAMPLES():
    return {name: entry['examples'] for name, entry in catalogue().items()}


def available():
    """True when the router can run. It needs the same torch + transformers as grounding."""
    try:
        import torch                                                     # noqa: F401
        import transformers                                              # noqa: F401
        return True
    except ImportError:
        return False


def _encoder():
    if 'model' not in _CACHE:
        from transformers import AutoModel, AutoTokenizer
        _CACHE['tok'] = AutoTokenizer.from_pretrained(MODEL_ID)
        _CACHE['model'] = AutoModel.from_pretrained(MODEL_ID).eval()
    return _CACHE['model'], _CACHE['tok']


def embed(sentences):
    """Mean-pooled, L2-normalised sentence embeddings."""
    import torch
    model, tok = _encoder()
    batch = tok(list(sentences), padding=True, truncation=True, max_length=64,
                return_tensors='pt')
    with torch.no_grad():
        out = model(**batch).last_hidden_state
    mask = batch['attention_mask'].unsqueeze(-1).float()
    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, dim=1)


def _library():
    """The example bank, encoded once."""
    if 'lib' not in _CACHE:
        names, texts = [], []
        for name, examples in EXAMPLES().items():
            names.extend([name] * len(examples))
            texts.extend(examples)
        _CACHE['lib'] = (names, embed(texts))
    return _CACHE['lib']


def rank(text):
    """Every operation, scored by its closest example. Highest first."""
    names, vectors = _library()
    scores = (embed([text]) @ vectors.T)[0]
    best = {}
    for name, score in zip(names, scores.tolist()):
        if score > best.get(name, -1):
            best[name] = score
    return sorted(best.items(), key=lambda kv: -kv[1])


def route(text, floor=FLOOR, margin=MARGIN):
    """(operation, confidence) for `text`, or (None, why) when it should be refused."""
    ranked = rank(text)
    (top, top_score), (_, second_score) = ranked[0], ranked[1]
    if top_score < floor:
        return None, (f'nothing here is near enough to that: the closest are '
                      f'{ranked[0][0]} ({top_score:.2f}) and {ranked[1][0]} '
                      f'({second_score:.2f}), against a floor of {floor:.2f}')
    if top_score - second_score < margin:
        return None, (f'{top} and {ranked[1][0]} are too close to call '
                      f'({top_score:.2f} vs {second_score:.2f})')
    return top, top_score


_HEX = re.compile(r'#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b')
# An article/possessive plus the words after it, e.g. "the person on the left".
_SUBJECT = re.compile(
    r'\b(?:the|that|this|those|these|his|her|its|their|my|our)\s+'
    r'((?:[a-z][a-z-]*)(?:\s+(?:of|on|in|at|to)\s+the\s+[a-z][a-z-]*|\s+[a-z][a-z-]*){0,3})')
# Words that name the picture or scene itself, not a thing in it.
_NOT_A_HEAD = {'rest', 'background', 'backdrop', 'image', 'picture', 'photo', 'frame',
               'shot', 'colour', 'color', 'colours', 'colors', 'size', 'web', 'original',
               'edges', 'corners', 'corner'}
# Trailing words to drop as dangling grammar, not part of the name (position words are
# excluded: "the person on the left" is the object as a whole).
_TRAILING = {'and', 'or', 'to', 'into', 'as', 'with', 'of', 'on', 'in', 'at', 'the', 'a',
             'instead', 'please', 'now'}


def colours(text):
    """Every colour named in the sentence, in the order they appear, as RGB tuples."""
    from .photo_describe import COLOUR_WORDS
    found = []
    for match in re.finditer(r'#[0-9a-fA-F]{3,6}\b|[a-z]+', text.lower()):
        token = match.group(0)
        if token.startswith('#'):
            h = token[1:]
            h = ''.join(c * 2 for c in h) if len(h) == 3 else h
            if len(h) == 6:
                found.append(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))
        elif token in COLOUR_WORDS:
            found.append(COLOUR_WORDS[token])
    return found


def colormap(text):
    """A matplotlib colormap named in the sentence, matched against `matplotlib.colormaps`."""
    try:
        import matplotlib
    except ImportError:
        return None
    # Sentence order, not a set (set order depends on PYTHONHASHSEED). An ordinary-word
    # name ("blues") counts only when the sentence marks it as the colormap.
    words = re.findall(r'[a-z_]+', text.lower())
    known = {name.lower(): name for name in matplotlib.colormaps if not name.endswith('_r')}
    for i, word in enumerate(words):
        if word not in known:
            continue
        if word not in _ORDINARY_WORDS or words[i + 1:i + 2] in (['colormap'], ['colourmap'],
                                                                 ['cmap']):
            return known[word]
    return None


# Registry names that are also ordinary words (a colour, a mood, a thing in the picture).
_ORDINARY_WORDS = {'blues', 'greens', 'reds', 'greys', 'grays', 'oranges', 'purples', 'gray',
                   'grey', 'pink', 'bone', 'copper', 'cool', 'hot', 'spring', 'summer',
                   'autumn', 'winter', 'flag', 'prism', 'ocean', 'rainbow', 'terrain',
                   'binary', 'paired', 'accent', 'hsv'}


def subject(text):
    """The thing the sentence is about, with its article -- or None. The article matters:
    the grounding model is prompted with it, and it changes what gets found."""
    from .photo_describe import COLOUR_WORDS
    for match in _SUBJECT.finditer(text.lower()):
        words = match.group(1).split()
        while words and (words[-1] in COLOUR_WORDS or words[-1] in _TRAILING):
            words.pop()
        if not words or words[0] in _NOT_A_HEAD or words[0] in COLOUR_WORDS:
            continue
        head = match.group(0).split()[0]
        return f'{head} {" ".join(words)}'
    return None


def quoted(text):
    """Text in quotes, which is what `add_text` is being asked to draw."""
    m = re.search(r'["“\']([^"”\']+)["”\']', text)
    return m.group(1) if m else None


_SLOTS = {
    'recolour_object': lambda t: _need(subject(t), to_rgb=_last(colours(t))),
    'erase_object': lambda t: _need(subject(t)),
    'isolate_object': lambda t: _need(subject(t)),
    'extract_object': lambda t: _need(subject(t)),
    'preview_object_mask': lambda t: _need(subject(t)),
    # remap_colormap needs hex stops (matplotlib rejects 0-255 tuples) and at least two.
    'remap_colormap': lambda t: {'source_cmap': colormap(t),
                                 'new_colours': _stops(colours(t))},
    'isolate_colour': lambda t: {'target_rgb': _first(colours(t))},
    'replace_colour': lambda t: {'from_rgb': _first(colours(t)),
                                 'to_rgb': _last(colours(t))},
    'preview_colour_mask': lambda t: {'target_rgb': _first(colours(t))},
    'replace_background': lambda t: {'colour_rgb': _first(colours(t))},
    'add_text': lambda t: {'text': quoted(t)},
}


def _need(subject_phrase, **extra):
    """An object operation with no subject cannot be served; say so rather than guess."""
    if not subject_phrase:
        return None
    args = {'subject': subject_phrase}
    args.update({k: v for k, v in extra.items() if v is not None})
    return args


def _stops(rgbs):
    """Colour stops as hex, or None when there are too few to make a gradient."""
    if len(rgbs) < 2:
        return None
    return ['#%02X%02X%02X' % tuple(int(v) for v in rgb) for rgb in rgbs]


def _first(values):
    return values[0] if values else None


def _last(values):
    return values[-1] if len(values) > 1 else (values[0] if values else None)



# Operations whose request shapes an embedding can't separate; redirected by outcome:
#   erase_object thing goes / extract_object thing stays, scene goes transparent /
#   isolate_object thing keeps colour / recolour_object thing changes colour
_FAMILY = {'erase_object', 'extract_object', 'isolate_object', 'recolour_object',
           'remove_object', 'crop'}
_CUES = (
    # the thing survives, alone, on transparency
    ('extract_object', r'\b(on its own|by itself|transparent|transparency|alpha|cut out|'
                       r'cutout|extract|lift (?:it |the )|as a png)\b'),
    # the thing keeps its colour while the rest loses it
    ('isolate_object', r'\b(colou?r pop|in colou?r|stays? colou?red|keep .{0,20}colou?r|'
                       r'(?:grey|gray|desaturate|mute|mono).{0,24}(?:rest|everything|else))\b'),
    # the thing goes away
    ('erase_object', r'\b(erase|delete|remove|get rid of|take .{0,20}out|lose the|'
                     r'paint (?:it |them )?out|clone out)\b'),
)


def _match_cue(lowered):
    """The first `_CUES` pattern the (already-lowered) sentence matches, or None."""
    for target, pattern in _CUES:
        if re.search(pattern, lowered):
            return target
    return None


def _within_family(name, text):
    """Redirect between operations of the same family using what each one DOES."""
    if name not in _FAMILY:
        return name
    lowered = text.lower()
    if colours(lowered) and subject(lowered):
        # A colour and a thing are both named: that's a recolour, whatever verb was used.
        return 'recolour_object'
    return _match_cue(lowered) or name


def _cued(text):
    """A cue-matched operation plus a named object, or None. A cue counts as evidence on
    its own, not just a tie-breaker, but still requires an object to act on."""
    lowered = text.lower()
    if not subject(lowered):
        return None
    # A recolour verb + thing + colour is always a recolour, not e.g. replace_background.
    if colours(lowered) and re.search(r'\b(recolou?r|repaint|paint)\b', lowered):
        return 'recolour_object'
    return _match_cue(lowered)


def resolve(text):
    """(operation, args) for a sentence, or None when it should be refused: an operation
    whose slot filler returns None (e.g. an object edit with no object named) is refused
    rather than called with a hole in it."""
    if not available():
        return None
    name, score = route(text)
    cued = _cued(text)
    if name is None:
        name = cued                 # the words say it plainly even if the vector does not
        if name is None:
            return None
    elif cued == 'recolour_object' and name not in _FAMILY:
        name = cued                 # an explicit recolour verb outranks a family guess
    name = _within_family(name, text)
    filler = _SLOTS.get(name)
    args = filler(text) if filler else {}
    if args is None:
        return None
    return name, {k: v for k, v in args.items() if v is not None}
