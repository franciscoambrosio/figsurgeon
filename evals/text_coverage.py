"""How much of the 31-operation tool surface plain language actually reaches.

    python evals/text_coverage.py            # scores whatever parser is wired in

Each row is a sentence a person might plausibly type, the operation it should reach, and
the arguments a wrong answer would get wrong (a colour, a phrase, a colormap name). `None`
as the expected operation means the sentence should be REFUSED -- it asks for something
this package does not do, and inventing a tool call for it is worse than saying no.

Sentences are deliberately not phrased like the existing regex patterns, so the battery
doesn't just measure the parser's own regexes against itself.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# (sentence, expected tool or None, {args that matter}). Expected may be a tuple when more
# than one operation can serve the request (e.g. recolour_object vs recolour_subject) --
# scoring those as failures would measure the battery's opinions, not the package's coverage.
CASES = [
    # ---------------------------------------------------------------- colour, whole frame
    ('make this black and white', 'stylise', {'effect': 'grayscale'}),
    ('drain all the colour out of it', 'stylise', {'effect': 'grayscale'}),
    ('give it a sepia wash', ('stylise', 'sepia'), {'effect': 'sepia'}),
    ('darken the corners a bit', 'stylise', {'effect': 'vignette'}),
    ('brighten it up', 'adjust', {'brightness': '>1'}),
    ('this is way too dark, lift it', 'adjust', {'brightness': '>1'}),
    ('punch up the contrast', 'adjust', {'contrast': '>1'}),
    ('the colours look washed out', 'adjust', {'saturation': '>1'}),
    ('fix the white balance', 'auto_white_balance', {}),
    ('clean up the grain', 'denoise', {}),

    # ---------------------------------------------------------------- colour, selective
    ('colour pop the red', 'isolate_colour', {'target_rgb': 'red'}),
    ('keep only the blue and grey out everything else', 'isolate_colour',
     {'target_rgb': 'blue'}),
    ('black and white except the red', 'isolate_colour', {'target_rgb': 'red'}),
    ('turn the red into orange', 'replace_colour', {'from_rgb': 'red', 'to_rgb': 'orange'}),
    ('swap the green for purple', 'replace_colour',
     {'from_rgb': 'green', 'to_rgb': 'purple'}),
    ('what colour is that patch', 'sample_colour', {}),

    # ---------------------------------------------------------------- subject / background
    ('blur the background', 'blur_background', {}),
    ('throw the background out of focus', 'blur_background', {}),
    ('cut me out of the background', 'remove_background', {}),
    ('put her on a white background', 'replace_background', {}),

    # ---------------------------------------------------------------- OBJECT tools
    ('make the horse purple', ('recolour_object', 'recolour_subject'), {'subject': 'the horse'}),
    ('repaint the guitar blue', ('recolour_object', 'recolour_subject'), {'subject': 'the guitar'}),
    ('the car should be green instead', ('recolour_object', 'recolour_subject'), {'subject': 'the car'}),
    ('recolour her dress to teal', ('recolour_object', 'recolour_subject'), {'subject': 'her dress'}),
    ('erase the bird', 'erase_object', {'subject': 'the bird'}),
    ('get rid of the lamp post', 'erase_object', {'subject': 'the lamp post'}),
    ('remove the person on the left', 'erase_object', {'subject': 'the person on the left'}),
    ('take the sign out of the shot', 'erase_object', {'subject': 'the sign'}),
    ('keep the cat in colour and grey the rest', ('isolate_object', 'isolate_subject'), {'subject': 'the cat'}),
    ('colour pop the dog', ('isolate_object', 'isolate_subject'), {'subject': 'the dog'}),
    ('cut out just the bicycle', 'extract_object', {'subject': 'the bicycle'}),
    ('give me the logo on its own with transparency', 'extract_object',
     {'subject': 'the logo'}),

    # ---------------------------------------------------------------- CHART tools
    ('re-theme the viridis colormap to our brand blues', 'remap_colormap',
     {'source_cmap': 'viridis'}),
    ('swap the jet colour scale for purple to yellow', 'remap_colormap',
     {'source_cmap': 'jet'}),
    ('this heatmap uses plasma, make it match #00407A and #52BDEC', 'remap_colormap',
     {'source_cmap': 'plasma'}),
    ('recolour the colorbar and the cells to our palette', 'remap_colormap', {}),
    ('the legend is see-through and the data shows through it, clean it up',
     'clean_transparent_box', {}),
    ('get rid of the ghosting inside the legend box', 'clean_transparent_box', {}),
    ('redraw the axis labels in Arial', 'replace_font', {}),

    # ---------------------------------------------------------------- geometry, text
    ('crop it to the top half', 'crop', {}),
    ('rotate it 90 degrees', ('rotate', 'crop_and_rotate'), {}),
    ('flip it horizontally', ('flip', 'crop_and_rotate'), {}),
    ('make it 800 px wide', 'resize', {}),
    ('put "DRAFT" across the middle', 'add_text', {}),

    # ---------------------------------------------------------------- looking / state
    ('show me the coordinate grid', 'show_grid', {}),
    ('let me see which pixels that would touch', 'preview_colour_mask', {}),
    ('undo that', 'undo', {}),
    ('start over', 'reset', {}),

    # ---------------------------------------------------------------- must be REFUSED
    ('make it look more cinematic', None, {}),
    ('add a hat to the dog', None, {}),
    ('fix the perspective', None, {}),
    ('make the sky more dramatic', None, {}),
    ('turn this into a watercolour painting', None, {}),
    ('deblur the motion blur on the car', None, {}),
    ('extend the image to 16:9', None, {}),
    ('swap the two people around', None, {}),
]


def score(resolve, cases=CASES, verbose=True):
    """`resolve(sentence)` returns (tool_name, args) or None. Returns (hits, misses, wrong).

    HIT reaches the right operation; MISS refuses something it could serve (rephraseable);
    WRONG invents or picks the wrong operation -- the expensive one, since it edits unasked.
    """
    hits, misses, wrong = [], [], []
    for sentence, expected, args in cases:
        try:
            got = resolve(sentence)
        except Exception:
            got = None
        name = got[0] if got else None
        ok = expected if isinstance(expected, tuple) else (expected,)
        if expected is None:
            (hits if name is None else wrong).append((sentence, expected, name))
        elif name in ok:
            hits.append((sentence, expected, name))
        elif name is None:
            misses.append((sentence, expected, name))
        else:
            wrong.append((sentence, expected, name))
    if verbose:
        for label, rows in (('WRONG', wrong), ('MISS', misses)):
            for sentence, expected, name in rows:
                print(f'  {label:5s} {sentence!r:58s} wanted {expected} got {name}')
        total = len(cases)
        print(f'\n  {len(hits)}/{total} reached the right operation '
              f'({len(hits) / total:.0%}); {len(misses)} refused something it can do; '
              f'{len(wrong)} answered the wrong thing')
    return hits, misses, wrong


def regex_resolve(sentence):
    """The shipped parser, as a (name, args) resolver."""
    from figsurgeon.photo_describe import parse
    from figsurgeon.workspace import as_tool_call
    fn, kwargs = parse(sentence)
    name, args = as_tool_call(fn, kwargs)
    if name is None:                       # reached an operation with no tool-layer name
        return getattr(fn, '__name__', str(fn)), kwargs
    return name, args


if __name__ == '__main__':
    print('photo_describe.parse (regex):')
    score(regex_resolve)


def intent_resolve(sentence):
    """The embedding router alone."""
    from figsurgeon import intent
    return intent.resolve(sentence)


def both_resolve(sentence):
    """Exact keyword matcher first, embedding router for what it misses.

    The keyword rules are exact where they match (e.g. "black and white BUT the red" must
    reach isolation, not grayscale); the router only knows what sounds nearest, so running
    it first would overrule a rule that's right by construction.
    """
    try:
        return regex_resolve(sentence)
    except Exception:
        return intent_resolve(sentence)


def calibrate():
    """Choose the router's floor/margin by leave-one-out over its own examples, never CASES
    (the held-out measure, which tuning on would ruin). Each example is routed against a
    library rebuilt without it; (floor, margin) is scored on held-out accuracy plus how many
    out-of-scope sentences stay refused.
    """
    from figsurgeon import intent
    names, texts = [], []
    for name, examples in intent.EXAMPLES.items():
        names.extend([name] * len(examples))
        texts.extend(examples)
    vectors = intent.embed(texts)
    out_of_scope = [s for s, expected, _ in CASES if expected is None]
    oos = intent.embed(out_of_scope)

    rows = []
    for i, (name, text) in enumerate(zip(names, texts)):
        keep = [j for j in range(len(texts)) if j != i]
        scores = (vectors[i:i + 1] @ vectors[keep].T)[0]
        best = {}
        for j, score in zip(keep, scores.tolist()):
            best[names[j]] = max(best.get(names[j], -1), score)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        rows.append((name, ranked[0][0], ranked[0][1], ranked[1][1]))

    oos_rows = []
    for k in range(len(out_of_scope)):
        scores = (oos[k:k + 1] @ vectors.T)[0]
        best = {}
        for j, score in zip(range(len(texts)), scores.tolist()):
            best[names[j]] = max(best.get(names[j], -1), score)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        oos_rows.append((ranked[0][1], ranked[1][1]))

    print(f'  {len(rows)} held-out examples, {len(oos_rows)} out-of-scope sentences')
    best_setting = None
    for floor in [round(0.30 + 0.02 * i, 2) for i in range(16)]:
        for margin in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
            kept = sum(1 for want, got, top, second in rows
                       if got == want and top >= floor and top - second >= margin)
            refused_oos = sum(1 for top, second in oos_rows
                              if top < floor or top - second < margin)
            total = kept + refused_oos
            if best_setting is None or total > best_setting[0]:
                best_setting = (total, floor, margin, kept, refused_oos)
    total, floor, margin, kept, refused = best_setting
    print(f'  best: floor={floor} margin={margin} -> {kept}/{len(rows)} held-out examples '
          f'routed correctly, {refused}/{len(oos_rows)} out-of-scope refused')
    return floor, margin


if __name__ == '__main__':
    import sys as _sys
    if '--calibrate' in _sys.argv:
        print('leave-one-out calibration of the router:')
        calibrate()
    else:
        print('photo_describe.parse (regex):')
        score(regex_resolve)
        print('\nintent.resolve (embedding router):')
        score(intent_resolve)
        print('\nregex first, router for the rest:')
        score(both_resolve)
