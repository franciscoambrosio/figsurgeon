"""Does a frontier image editor, given the whole picture, change only what it was asked to?

    .venv/bin/python evals/frontier_editors.py [model ...]

Each model gets the whole image and one precise recolour instruction -- the way a person
uses it -- and is graded on the same mask:

  drift     what changed OUTSIDE the mask (`background_drift.drift`, after re-registering
            a reframed return, so it is a lower bound)
  intent    whether the asked-for change happened inside it (`background_drift.intent`)
  data      on the charts only: did the plotted values survive (bar heights in data units,
            per-series pixel overlap on the line chart)

and then again after `hybrid.paste` puts the model's pixels back inside the mask only, which
is what figsurgeon adds to any editor. A figsurgeon-only arm is the free baseline.

Models are OpenRouter ids, or `openai-api:<id>` for OpenAI's own Images API (e.g.
`openai-api:gpt-image-2.5-sunburst`, needs OPENAI_API_KEY). Every return is cached under
the out dir, so re-running grades again without paying again.
"""
import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'specs'))

from figsurgeon import objects                                # noqa: E402
from figsurgeon.compose import background, classify                     # noqa: E402
from figsurgeon.perceptual import delta_e
from figsurgeon.tools import dispatch                                   # noqa: E402
import background_drift as BD                                           # noqa: E402
import hybrid                                                           # noqa: E402
import openrouter_edit                                                  # noqa: E402
from recolour_vs_generative import CASES as PHOTO_CASES, load_case     # noqa: E402

OUT = os.path.join(HERE, 'out_frontier_editors')
MODELS = ['openai/gpt-5.4-image-2', 'google/gemini-3-pro-image']
PHOTOS = ['door', 'two_cars', 'coffee_cup', 'portrait_studio']
INSTRUCTION = ('Change the colour of {subject} to {words}. Keep everything else in the '
               'image exactly as it is.')

BAR_BLUE, BAR_ORANGE, BAR_NAVY = (31, 119, 180), (255, 127, 14), (0, 64, 122)
BAR_PX_PER_UNIT = (617 - 89) / 60          # chart_bar.png: y=0 at row 617, y=60 at row 89


# Judged by eye at 1:1 (2026-09-24, google/gemini-3-pro-image, one call per case). The drift
# numbers overstate photos: the background is regenerated at the pixel level (wood grain,
# grass) but reads as unchanged, and every return comes back at ~1 MP whatever was sent.
JUDGED = {
    'chart_bar': 'exact colour (dE 0.009), all 8 bar heights within 1 px; excellent',
    'chart_lines': 'right recolour, legend too; the recoloured line redrawn 2.5 px higher '
                   '(0.06 data units), other series within resampling; frame stretched 0.6%',
    'door': 'clean recolour, nothing else visibly changed; 1920x1278 returned as 1264x848',
    'two_cars': 'clean, second car left red, bumper included (our mask misses it); 1200x896',
    'coffee_cup': 'cup and saucer recoloured, table intact to the eye; 600x400 -> 1264x848',
    'portrait_studio': 'refused: PROHIBITED_CONTENT',
}


def _saturated(a):
    a = a.astype(int)
    return (a.max(axis=2) - a.min(axis=2)) > 40


def bar_centres(img):
    """Centre column of every bar, found by colour in the ORIGINAL so each output is read at
    the same places (blue and orange bars touch, so saturation alone merges them)."""
    a = np.asarray(img.convert('RGB')).astype(int)
    centres = []
    for colour in (BAR_BLUE, BAR_ORANGE):
        row = (np.abs(a[600] - colour).max(axis=1) < 40).astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, row, 0]))
        centres += [(x0 + x1) // 2 for x0, x1 in zip(edges[::2], edges[1::2])]
    return centres


def bar_heights(img, centres):
    """Each bar's height in data units, scanning up from the x axis at its centre column."""
    sat = _saturated(np.asarray(img.convert('RGB')))
    return [round(int(np.argmin(sat[:616, x][::-1])) / BAR_PX_PER_UNIT, 2) for x in centres]


def series_overlap(img, spec, reference):
    """Per-series IoU against the original's classification, for every series but `mlp`."""
    a = np.asarray(img.convert('RGB'))
    masks, _, _ = classify(a, background(a, spec), spec)
    return {n: round(float((masks[n] & reference[n]).sum() / max((masks[n] | reference[n]).sum(), 1)), 3)
            for n in reference if n != 'mlp'}


def chart_cases():
    bar = Image.open(os.path.join(HERE, '_corpus', 'agent_loop', 'chart_bar.png')).convert('RGB')
    a = np.asarray(bar).astype(int)
    bar_mask = (np.abs(a - BAR_BLUE).max(axis=2) < 40).astype(float)
    centres = bar_centres(bar)

    from timeseries_demo import SPEC
    ts = Image.open(os.path.join(ROOT, 'tests', 'data', 'timeseries_demo.png')).convert('RGB')
    t = np.asarray(ts)
    ts_masks, _, _ = classify(t, background(t, SPEC), SPEC)
    ts_mask = _grow(ts_masks['mlp'], 1)        # plus the line's anti-aliased rim

    return [
        {'id': 'chart_bar', 'image': bar, 'mask': bar_mask, 'to_rgb': BAR_NAVY,
         'subject': 'the blue bars (Sales) and their legend swatch',
         'words': 'dark navy #00407A',
         # No figsurgeon-only arm: replace_colour moves hue only, and navy is the same hue
         # as the original blue, so it has no exact-colour recolour for this without a spec.
         'ours': None,
         'data': lambda im: {'bar_heights': bar_heights(im, centres)},
         'truth': {'bar_heights': bar_heights(bar, centres)}},
        {'id': 'chart_lines', 'image': ts, 'mask': ts_mask, 'to_rgb': (200, 0, 0),
         'subject': 'the green MLP line', 'words': 'red',
         'ours': lambda im: dispatch('replace_colour', {'from_rgb': [0, 128, 0],
                                                        'to_rgb': [200, 0, 0]}, im)[0],
         'data': lambda im: {'series_iou': series_overlap(im, SPEC, ts_masks)}},
    ]


def _grow(mask, r):
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, iterations=r).astype(float)


# The README's horse: CC0, Benny Jackson via Unsplash,
# https://commons.wikimedia.org/wiki/File:Horse_in_a_snowy_field_in_close-up_(Unsplash).jpg
# (fetched at 1920 px wide into evals/_corpus/readme/).
README_CASES = [
    {'id': 'horse_snow', 'image': 'readme/horse_snow.jpg', 'box': [40, 270, 870, 1339],
     'subject': 'the horse', 'to_rgb': [25, 25, 25], 'words': 'black'},
]


def photo_cases():
    out = []
    for case in PHOTO_CASES + README_CASES:
        if case['id'] not in PHOTOS + ['horse_snow']:
            continue
        img, box, subject, _ = load_case(case)
        m, _ = objects.segment_objects(img, [box])
        m = np.asarray(m, dtype=float)
        m = m / 255.0 if m.max() > 1.5 else m
        out.append({'id': case['id'], 'image': img, 'mask': m, 'to_rgb': case['to_rgb'],
                    'subject': subject, 'words': case['words'],
                    'ours': lambda im, box=box, c=case['to_rgb']:
                        objects.recolour_object(im, box=box, to_rgb=c)[0]})
    return out


def openai_edit(image, instruction, model):
    """OpenAI's Images API directly (the 2.5 models are not on OpenRouter yet)."""
    import requests
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        path = os.path.join(ROOT, '.env.local')
        for line in open(path) if os.path.exists(path) else []:
            name, _, value = line.strip().partition('=')
            if name == 'OPENAI_API_KEY':
                key = value.strip().strip('"\'')
    if not key:
        return None, {'error': 'no OPENAI_API_KEY in the environment or .env.local'}
    buf = io.BytesIO()
    image.convert('RGB').save(buf, 'PNG')
    r = requests.post('https://api.openai.com/v1/images/edits',
                      headers={'Authorization': 'Bearer ' + key},
                      data={'model': model, 'prompt': instruction},
                      files={'image': ('image.png', buf.getvalue(), 'image/png')},
                      timeout=600)
    if r.status_code != 200:
        return None, {'error': f'HTTP {r.status_code}', 'detail': r.text[:500]}
    body = r.json()
    raw = base64.b64decode(body['data'][0]['b64_json'])
    return Image.open(io.BytesIO(raw)), {'model': model, 'usage': body.get('usage', {})}


def returned(case, model, instruction):
    """The model's picture for this case, from the cache or from one paid call."""
    slug = model.replace('/', '_').replace(':', '_')
    path = os.path.join(OUT, f'{case["id"]}__{slug}.png')
    if os.path.exists(path):
        return Image.open(path), json.load(open(path.replace('.png', '.json')))
    if model.startswith('openai-api:'):
        img, meta = openai_edit(case['image'], instruction, model.split(':', 1)[1])
    else:
        img, meta = openrouter_edit.edit(case['image'], instruction, model, fmt='PNG')
    if img is None:
        print(f'    {model}: NO IMAGE -- {str(meta)[:200]}')
        return None, meta
    img.save(path)
    json.dump(meta, open(path.replace('.png', '.json'), 'w'), indent=2, default=str)
    return img, meta


def grade(case, edited, covered=None, returned_size=None):
    before, keep = case['image'], case['mask']
    d = BD.drift(before, edited, keep, covered)
    inside = np.asarray(edited.convert('RGB'))[np.asarray(keep) > 0.5]
    got = np.median(inside, axis=0).astype(np.uint8)
    row = {'visible': d['share_visible'], 'jnd': d['share_jnd'], 'blobs': d['blobs'],
           'largest_blob': d['largest_blob_px'],
           'moved': BD.intent(before, edited, keep, case['to_rgb'])['moved_share'],
           'median_colour': got.tolist(),
           'dE_to_asked': round(float(delta_e(got[None], np.array(case['to_rgb'], np.uint8)[None])[0]), 3)}
    if returned_size and returned_size != before.size:
        # what resizing to the returned size and back costs on its own: drift at or below
        # this is resampling, not the model changing content
        row['visible_resample_floor'] = BD.resample_floor(before, returned_size, keep)['share_visible']
    if 'data' in case:
        row.update(case['data'](edited))
    return row


def run(models):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for case in chart_cases() + photo_cases():
        before = case['image']
        instruction = INSTRUCTION.format(subject=case['subject'], words=case['words'])
        print(f'\n=== {case["id"]} {before.size[0]}x{before.size[1]}: "{instruction}"')
        if 'truth' in case:
            print(f'    original data: {case["truth"]}')
        if case['ours']:
            ours = case['ours'](before)
            rows.append({'case': case['id'], 'arm': 'figsurgeon only', **grade(case, ours)})
            print(f'    {"figsurgeon only":40s} {rows[-1]}')
        for model in models:
            got, meta = returned(case, model, instruction)
            if got is None:
                continue
            aligned, covered, notes = BD.align(before, got)
            raw = {'case': case['id'], 'arm': model,
                   **grade(case, aligned, covered, got.size),
                   'returned': list(got.size), 'cost': (meta.get('usage') or {}).get('cost')}
            pasted, _ = hybrid.paste(before, aligned, case['mask'], covered=covered)
            fixed = {'case': case['id'], 'arm': model + ' + figsurgeon',
                     **grade(case, pasted)}
            rows += [raw, fixed]
            for r in (raw, fixed):
                print(f'    {r["arm"]:40s} {r}')
            for n in notes:
                print(f'        {n}')
            BD.sheet(os.path.join(OUT, f'{case["id"]}__{model.replace("/", "_").replace(":", "_")}_sheet.png'),
                     before, aligned, case['mask'], model, covered)
    json.dump(rows, open(os.path.join(OUT, 'rows.json'), 'w'), indent=2, default=str)
    print(f'\nrows and sheets in {OUT} -- look at the sheets before trusting any number')
    return rows


if __name__ == '__main__':
    run(sys.argv[1:] or MODELS)
