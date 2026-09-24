"""Checks whether _EXTENT_PAD=0.3 holds up across object sizes, or whether the right pad
is size-dependent. Uses the 9 SCORED cases from evals/object_completeness.py (real
ground-truth SlimSAM masks, so 'full'/'half' recolour scenarios are exact) plus 6
SUPPLEMENTAL cases spanning other object sizes, using the segmenter's own mask as a
weaker stand-in ground truth (kept separate in the report).

For each case and pad in {0.0, 0.15, 0.3, 0.6, 'frame'}, computes completeness
k = |extent & changed| / |extent| for the 'full' and 'half' scenario, via
verify_photo._phrase_probability (or grounding._probability for pad='frame'). Results go
to $TMPDIR/audit_phrase/pad_sweep_results.json.
"""
import os, sys, json, time
import numpy as np
from PIL import Image

SP = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'audit_phrase')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'evals', 'segmentation_models'))

from evals import real_corpus
import cases as C_seg
from figsurgeon import grounding, objects
from figsurgeon.verify_photo import _phrase_probability, _PHRASE_THRESHOLD

HERE_SEG = os.path.join(ROOT, 'evals', 'segmentation_models')
COMPLETENESS_JSON = os.path.join(ROOT, 'evals', 'completeness_cases.json')
MASKS_SLIMSAM = os.path.join(HERE_SEG, 'masks_slimsam')
MASKS_OWN = os.path.join(ROOT, 'evals', 'masks_completeness')

SCORED = ['astronaut_suit', 'coffee_cup', 'car_red', 'portrait_studio', 'food_pizza',
          'flower_macro', 'shoe_white', 'night_tower', 'camera_man']

seg_cases = {c['id']: c for c in C_seg.load()}
own_cases = {c['id']: c for c in json.load(open(COMPLETENESS_JSON))}
OWN_IDS = set(own_cases)
all_cases = {**seg_cases, **own_cases}


def load_image_for(case_id):
    case = all_cases[case_id]
    if case_id not in OWN_IDS:
        return C_seg.image(case)
    if case['source'] == 'skimage':
        from skimage import data
        return Image.fromarray(getattr(data, case['name'])())
    img = real_corpus.load(case['name'])
    img.thumbnail((C_seg.WORKING, C_seg.WORKING), Image.LANCZOS)
    return img


def object_mask(case_id):
    path = (os.path.join(MASKS_OWN, case_id + '.png') if case_id in OWN_IDS
            else os.path.join(MASKS_SLIMSAM, case_id + '.png'))
    return np.asarray(Image.open(path)) > 127


def half(mask):
    ys, xs = np.nonzero(mask)
    out = mask.copy()
    if np.ptp(ys) >= np.ptp(xs):
        out[(ys.min() + ys.max()) // 2:, :] = False
    else:
        out[:, (xs.min() + xs.max()) // 2:] = False
    return out


PADS = [0.0, 0.15, 0.3, 0.6, 'frame']


def completeness_at(prob, box, changed):
    x0, y0, x1, y1 = box
    whole = prob > _PHRASE_THRESHOLD
    extent = np.zeros(prob.shape, bool)
    extent[y0:y1, x0:x1] = whole[y0:y1, x0:x1]
    n_extent = int(extent.sum())
    if n_extent < grounding.MIN_COVERAGE * prob.size:
        return None, None, n_extent
    hit = int((extent & changed).sum())
    n_changed = int(changed.sum())
    corroboration = hit / n_changed if n_changed else float('nan')
    k = hit / n_extent
    return k, corroboration, n_extent


def run_scored_case(case_id, results):
    case = all_cases[case_id]
    img = load_image_for(case_id)
    W, H = img.size
    box = tuple(int(v) for v in case['box'])
    x0, y0, x1, y1 = box
    whole_obj = object_mask(case_id)
    full_changed = np.zeros((H, W), bool)
    full_changed[y0:y1, x0:x1] = whole_obj[y0:y1, x0:x1]
    half_changed = np.zeros((H, W), bool)
    half_obj = half(whole_obj)
    half_changed[y0:y1, x0:x1] = half_obj[y0:y1, x0:x1]

    box_frac = (x1 - x0) * (y1 - y0) / float(W * H)
    print(f'{case_id}: box_frac={box_frac:.4f} phrase={case["what"]!r}', flush=True)

    for pad in PADS:
        if pad == 'frame':
            prob = grounding._probability(img.convert('RGB'), case['what'])
        else:
            prob = _phrase_probability(img.convert('RGB'), case['what'], box, pad=pad)
        kf, cf, nef = completeness_at(prob, box, full_changed)
        kh, ch, neh = completeness_at(prob, box, half_changed)
        results.append({'case': case_id, 'kind': 'scored', 'box_frac': box_frac,
                         'phrase': case['what'], 'pad': pad,
                         'k_full': kf, 'corrob_full': cf, 'n_extent_full': nef,
                         'k_half': kh, 'corrob_half': ch, 'n_extent_half': neh})
        print(f'  pad={pad!s:5} full k={kf} half k={kh}', flush=True)


SUPPLEMENTAL = ['boat_1', 'bike_2', 'flower_1', 'city_1', 'guitar_1', 'chairs_1']


def run_supplemental_case(cid, results):
    import importlib.util
    spec = importlib.util.spec_from_file_location('audit_cases', os.path.join(SP, 'cases.py'))
    C_aud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C_aud)
    row = next(r for r in C_aud.CASES if r[0] == cid)
    _, img_name, phrase, kind, expected, note = row
    img = C_aud.get_image(img_name)
    W, H = img.size

    # Segmenter's own mask as a stand-in "ground truth" -- weaker, same kind of model
    # being tested; kept separate in the report.
    prob_default = _phrase_probability(img.convert('RGB'), phrase, (0, 0, W, H), pad=0.0)
    hard = prob_default > 0.5
    if not hard.any():
        print(f'{cid}: SKIP, phrase matches nothing at pad=0', flush=True)
        return
    ys, xs = np.where(hard)
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    if bx1 - bx0 < 8 or by1 - by0 < 8:
        print(f'{cid}: SKIP, region too small', flush=True)
        return
    seg_mask, coverage = objects.segment_object(img, (bx0, by0, bx1, by1))
    whole_obj = np.asarray(seg_mask) > 0.5
    box = (bx0, by0, bx1, by1)
    full_changed = np.zeros((H, W), bool)
    full_changed[by0:by1, bx0:bx1] = whole_obj[by0:by1, bx0:bx1]
    half_changed = np.zeros((H, W), bool)
    half_obj = half(whole_obj[by0:by1, bx0:bx1])
    tmp = np.zeros((H, W), bool)
    tmp[by0:by1, bx0:bx1] = half_obj
    half_changed = tmp

    box_frac = (bx1 - bx0) * (by1 - by0) / float(W * H)
    print(f'{cid}: box_frac={box_frac:.4f} phrase={phrase!r} (supplemental, proxy truth)',
          flush=True)

    for pad in PADS:
        if pad == 'frame':
            prob = grounding._probability(img.convert('RGB'), phrase)
        else:
            prob = _phrase_probability(img.convert('RGB'), phrase, box, pad=pad)
        kf, cf, nef = completeness_at(prob, box, full_changed)
        kh, ch, neh = completeness_at(prob, box, half_changed)
        results.append({'case': cid, 'kind': 'supplemental', 'box_frac': box_frac,
                         'phrase': phrase, 'pad': pad,
                         'k_full': kf, 'corrob_full': cf, 'n_extent_full': nef,
                         'k_half': kh, 'corrob_half': ch, 'n_extent_half': neh})
        print(f'  pad={pad!s:5} full k={kf} half k={kh}', flush=True)


def main():
    results = []
    t0 = time.time()
    for case_id in SCORED:
        run_scored_case(case_id, results)
        json.dump(results, open(os.path.join(SP, 'pad_sweep_results.json'), 'w'), indent=2)
    for cid in SUPPLEMENTAL:
        run_supplemental_case(cid, results)
        json.dump(results, open(os.path.join(SP, 'pad_sweep_results.json'), 'w'), indent=2)
    print('pad sweep done in', time.time() - t0, 's')


if __name__ == '__main__':
    main()
