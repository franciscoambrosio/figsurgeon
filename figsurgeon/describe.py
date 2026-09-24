"""Parse a plain-language instruction into a call against edit.py.

This is deliberately small.  It is NOT trying to be a general NLP engine -- it resolves
series names (including colour-word synonyms drawn from the spec) and a handful of verbs,
and raises rather than guess when a request doesn't map cleanly.  In an interactive session
the calling model (Claude) is the real parser: it can read the spec, recognise "the red
line" as `linear` because that series is orange-ish-red, and call the edit.py functions
directly with full context.  This module exists so the same mapping also works headless,
outside a chat, for batch or CI use.
"""
import re

COLOUR_WORDS = {
    'red': (220, 30, 30), 'blue': (30, 80, 220), 'green': (0, 128, 0),
    'orange': (255, 140, 0), 'black': (0, 0, 0), 'grey': (204, 204, 204),
    'gray': (204, 204, 204), 'purple': (128, 0, 160), 'yellow': (220, 180, 0),
}


def _nearest_colour_word(rgb):
    r, g, b = rgb
    if max(rgb) - min(rgb) < 25:
        return 'black' if sum(rgb) < 200 else 'grey'
    if r >= g and r >= b:
        return 'orange' if g > 80 else 'red'
    if g >= r and g >= b:
        return 'green'
    return 'blue'


def _clean(phrase):
    """Strip filler WORDS (word-boundary only -- naive substring replace corrupts
    "offline" into "off" because it contains "line")."""
    phrase = re.sub(r'\bthe\b|\bline\b|\bapproach\b', ' ', phrase)
    return re.sub(r'\s+', ' ', phrase).strip().lower()


def resolve_series(word, spec):
    """Match a phrase against series names, their words, or the colour they're drawn in."""
    w = _clean(word).rstrip('s')
    if not w:
        raise ValueError(f'Empty series reference in {word!r}')
    for s in spec.series:
        n = s.name.lower()
        if w == n or w == n.replace('_', ' '):
            return s.name
    # loosest: any shared word between the phrase and the series name (or its display words)
    wwords = set(w.split())
    for s in spec.series:
        nwords = set(s.name.lower().replace('_', ' ').split())
        if wwords & nwords:
            return s.name
    for s in spec.series:
        if _nearest_colour_word(s.colour) == w:
            return s.name
    raise ValueError(f'Could not match {word!r} to a series. '
                      f'Known: {[s.name for s in spec.series]}')


def parse(text, spec):
    """Return (verb, kwargs) for the closest-matching supported instruction.

    Supported shapes (case-insensitive), matched loosely:
      "highlight X" / "keep X" / "only X"                       -> highlight, keep=[X]
      "highlight X and Y" / "keep X, Y"                          -> highlight, keep=[X, Y]
      "grey out everything except X"                             -> highlight, keep=[X]
      "make X red" / "recolour X to <colour>" / "change X colour" -> recolour, colour=...
      "make X thicker" / "double the width of X" / "thinner"      -> thicken, factor=...
    """
    t = text.strip().lower()

    m = re.search(r'(?:highlight|keep|only show|only keep)\s+(?:the\s+)?(.+?)'
                  r'(?:\s+line)?(?:,?\s*(?:and|,)\s*(?:the\s+)?(.+?)(?:\s+line)?)*'
                  r'(?:\s*(?:,\s*grey|and\s+grey|,\s*and\s+grey)?.*)?$', t)
    if any(k in t for k in ('highlight', 'keep only', 'only show')) or \
       (t.startswith('keep ') and 'except' not in t):
        names = re.split(r',|\band\b', re.sub(r'^(highlight|keep|only show|only keep)\s+', '', t))
        keep = [resolve_series(n, spec) for n in names if _clean(n)]
        return 'highlight', {'keep': keep}

    m = re.search(r'except\s+(?:the\s+)?(.+?)(?:\s+line)?$', t)
    if 'except' in t and m:
        names = re.split(r',|\band\b', m.group(1))
        keep = [resolve_series(n, spec) for n in names if n.strip()]
        return 'highlight', {'keep': keep}

    m = re.search(r'(?:make|recolou?r|change)\s+(?:the\s+)?(.+?)\s+(?:line\s+)?'
                  r'(?:to\s+)?(?:colou?r\s+)?(\w+)', t)
    if m and ('colour' in t or 'color' in t or m.group(2) in COLOUR_WORDS):
        name, colour_word = m.group(1).replace('line', '').strip(), m.group(2)
        if colour_word in COLOUR_WORDS:
            return 'recolour', {'target': resolve_series(name, spec),
                                 'colour': COLOUR_WORDS[colour_word]}

    m = re.search(r'(thicker|thinner|thicken|thin|double|width)', t)
    if m:
        target_m = re.search(r'(?:make|thicken|thin)\s+(?:the\s+)?(\w[\w\s]*?)\s+(?:line\s+)?'
                             r'(?:thicker|thinner)', t) or \
                   re.search(r'(?:the\s+)?(\w[\w\s]*?)\s+line', t) or \
                   re.search(r'(?:of|for)\s+(?:the\s+)?(\w[\w\s]*)', t)
        if not target_m:
            raise ValueError(f'Could not find which series to resize in: {text!r}')
        name = target_m.group(1)
        factor = 2.0
        fm = re.search(r'(\d+(\.\d+)?)\s*x', t)
        if fm:
            factor = float(fm.group(1))
        elif 'thin' in t:
            factor = 0.6
        return 'thicken', {'target': resolve_series(name, spec), 'factor': factor}

    raise ValueError(f'Could not parse instruction: {text!r}. '
                      f'Try phrasing like "highlight X", "keep X and Y", '
                      f'"make X red", or "make X thicker".')


def apply_text(image_or_path, spec, text):
    """Parse `text` and run the matching edit.py action in one call."""
    from . import edit
    verb, kwargs = parse(text, spec)
    fn = {'highlight': edit.highlight, 'recolour': edit.recolour_series,
          'thicken': edit.thicken}[verb]
    out, info = fn(image_or_path, spec, **kwargs)
    return out, verb, kwargs, info
