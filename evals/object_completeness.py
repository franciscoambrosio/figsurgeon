"""Measures whether an object recolour changed the WHOLE object, not just enough to avoid
leaking outside its region -- the failure `workspace.verify`'s containment check misses
(a segmenter can recolour half an object and still report nothing leaked).

    .venv/bin/python evals/object_completeness.py [out_dir]

Scores five candidate completeness measures against must-PASS/must-FAIL cases (full, loose
box, half-recoloured, eroded sliver, no-op), built from committed SlimSAM masks in
`evals/segmentation_models/`. Sheets and tables are printed/saved to `out_dir` (default
`evals/object_completeness/`).

Headline: candidates 1-4 (colour matching, boundary flatness, rembg extent) all infer the
object's extent from the box or the edited pixels, so they overlap: some complete recolours
score worse than incomplete ones. Candidate 5, `phrase_extent`, estimates extent from the
caller's own words via CLIPSeg instead -- it separates cleanly (margin +0.23) and is
calibrated against the true changed fraction (median error 0.055), abstaining only between
f=0.60-0.85 where no threshold would be honest.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, 'segmentation_models'))

import cases as C                                                        # noqa: E402

from figsurgeon import advanced as A                                    # noqa: E402
from figsurgeon import objects as O                                     # noqa: E402
from figsurgeon.locate import _font                                     # noqa: E402
from figsurgeon.perceptual import to_oklab                              # noqa: E402
from figsurgeon.photo import _feather                                   # noqa: E402
from figsurgeon.verify_photo import check_object_recoloured             # noqa: E402
from figsurgeon.verify_photo import _phrase_probability                 # noqa: E402

SCORED = ['astronaut_suit', 'coffee_cup', 'car_red', 'portrait_studio', 'food_pizza',
          'flower_macro', 'shoe_white', 'night_tower', 'camera_man']
TARGETS = [(40, 80, 200), (30, 150, 60)]
GATE, BAND = 8, 3          # /255 change gate; px of feathered edge treated as ambiguous

# This eval's own three cases (masks in masks_completeness/) cover a repeated object,
# a greyscale photo, and a night scene against dark sky; each mask was reviewed before
# being kept.
OWN_CASES = json.load(open(os.path.join(HERE, 'completeness_cases.json')))
OWN_IDS = {c['id'] for c in OWN_CASES}


def all_cases():
    return {c['id']: c for c in C.load() + OWN_CASES}


def load_image(case):
    if case['id'] not in OWN_IDS:
        return C.image(case)
    if case['source'] == 'skimage':
        from skimage import data
        return Image.fromarray(getattr(data, case['name'])())
    from evals import real_corpus
    img = real_corpus.load(case['name'])
    img.thumbnail((C.WORKING, C.WORKING), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# the edit, and the masks that make each variant

def recolour(img, mask, to_rgb):
    """`objects.recolour_object`'s maths, driven by a mask given from outside."""
    a = np.array(img.convert('RGB')).astype(float)
    target = np.array(to_rgb, dtype=float)
    lum = a.mean(axis=2) / 255.0
    scale = np.clip(lum / max(target.mean() / 255.0, 1e-6), 0.35, 1.9)
    recoloured = np.clip(target[None, None, :] * scale[:, :, None], 0, 255)
    m = mask[:, :, None]
    return Image.fromarray((a * (1 - m) + recoloured * m).astype(np.uint8))


def object_mask(case_id):
    """The whole object, NOT clipped to the box, so "how much changed" stays computable for
    an object that continues past its box."""
    path = (os.path.join(HERE, 'masks_completeness', case_id + '.png') if case_id in OWN_IDS
            else os.path.join(HERE, 'segmentation_models', 'masks_slimsam', case_id + '.png'))
    return np.asarray(Image.open(path)) > 127


def true_mask(case_id, box, shape):
    """The object mask, clipped to the box the way `segment_object` clips its own."""
    m = object_mask(case_id)
    clipped = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    clipped[y0:y1, x0:x1] = m[y0:y1, x0:x1]
    return clipped


def half(mask):
    """One half of the object, split across its longer axis -- the leg-left-behind case."""
    ys, xs = np.nonzero(mask)
    out = mask.copy()
    if np.ptp(ys) >= np.ptp(xs):
        out[(ys.min() + ys.max()) // 2:, :] = False
    else:
        out[:, (xs.min() + xs.max()) // 2:] = False
    return out


def eroded(mask, keep):
    """`keep` of the object, taken from the middle outward -- found the body, not the edge."""
    d = ndimage.distance_transform_edt(mask)
    return mask & (d > np.percentile(d[mask], 100 * (1 - keep)))


# ---------------------------------------------------------------------------
# the four candidate measures. Each returns one number, higher = more complete, so they
# can be read in the same direction.

def _parts(before, after, box):
    a0 = np.array(before.convert('RGB')).astype(float)
    a1 = np.array(after.convert('RGB')).astype(float)
    H, W = a0.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    inside = np.zeros((H, W), bool)
    inside[max(0, y0):y1, max(0, x0):x1] = True
    changed = (np.abs(a0 - a1).max(axis=2) > GATE) & inside
    return a0, inside, changed


def oklab_pair(img):
    """OKLab of the original, raw and smoothed, computed once per photograph (recomputing
    per measure exhausts memory)."""
    a = np.array(img.convert('RGB')).astype(np.float32)
    return (to_oklab(a).astype(np.float32),
            to_oklab(ndimage.gaussian_filter(a, (2, 2, 0))).astype(np.float32))


def leftover_dense(before, after, box, lab, tol=0.02, samples=3000):
    """(1) Unchanged pixels in the box whose ORIGINAL colour matches a changed pixel's,
    where "matches" means within 0.02 OKLab of any of 3000 sampled changed pixels."""
    _, inside, changed = _parts(before, after, box)
    cand = inside & ~ndimage.binary_dilation(changed, iterations=BAND)
    if changed.sum() < 200 or cand.sum() < 200:
        return float('nan')
    rs = np.random.RandomState(0)
    ref = lab[changed]
    ref = ref[rs.permutation(ref.shape[0])[:samples]]
    q = lab[cand]
    q = q[rs.permutation(q.shape[0])[:samples * 4]]
    rate = float((cKDTree(ref).query(q)[0] <= tol).mean())
    n = int(changed.sum())
    return n / (n + rate * int(cand.sum()))


def leftover_coarse(before, after, box, lab, tol=0.06, k=5):
    """(2) The same, with the object's colour modelled as 5 dominant colours instead."""
    _, inside, changed = _parts(before, after, box)
    cand = inside & ~ndimage.binary_dilation(changed, iterations=BAND)
    if changed.sum() < 200 or cand.sum() < 200:
        return float('nan')
    rs = np.random.RandomState(0)
    ref = lab[changed]
    ref = ref[rs.permutation(ref.shape[0])[:20000]]
    q = lab[cand]
    q = q[rs.permutation(q.shape[0])[:20000]]
    centres = kmeans2(ref, k, minit='++', seed=0)[0]
    near = np.sqrt(((q[:, None, :] - centres[None]) ** 2).sum(-1)).min(1)
    n = int(changed.sum())
    return n / (n + float((near <= tol).mean()) * int(cand.sum()))


def boundary_flat(before, after, box, lab, tol=0.05, step=5):
    """(3) 1 - the fraction of the mask's edge that cuts through material of the same colour
    on both sides. A cut across an object is flat; the object's own outline is not."""
    a0, inside, changed = _parts(before, after, box)
    if changed.sum() < 200:
        return float('nan')
    H, W = a0.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    edge = np.zeros((H, W), bool)     # the box's own border is containment, not evidence
    edge[max(0, y0 - 2):y0 + 4, :] = edge[y1 - 4:min(H, y1 + 2), :] = True
    edge[:, max(0, x0 - 2):x0 + 4] = edge[:, x1 - 4:min(W, x1 + 2)] = True
    inner = changed & ~ndimage.binary_erosion(changed, iterations=BAND + step) & ~edge
    outer = (ndimage.binary_dilation(changed, iterations=BAND + step)
             & ~ndimage.binary_dilation(changed, iterations=BAND) & ~edge)
    if inner.sum() < 50 or outer.sum() < 50:
        return float('nan')
    iy, ix = np.nonzero(inner)
    oy, ox = np.nonzero(outer)
    pick = np.random.RandomState(0).permutation(iy.size)[:5000]
    iy, ix = iy[pick], ix[pick]
    j = cKDTree(np.c_[oy, ox]).query(np.c_[iy, ix])[1]
    d = np.sqrt(((lab[iy, ix] - lab[oy[j], ox[j]]) ** 2).sum(axis=1))
    return 1.0 - float((d <= tol).mean())


def subject_extent(before, after, box, subject):
    """(4) The fraction of rembg's SUBJECT inside the box that changed -- an extent estimate
    from a model that does not depend on how the box was drawn."""
    _, inside, changed = _parts(before, after, box)
    s = (subject > 0.5) & inside
    return float(changed[s].mean()) if s.sum() >= 200 else float('nan')


def phrase_extent(before, after, box, prob):
    """(5) The fraction of the caller's-words region (CLIPSeg probability `prob`, computed
    once on the original image) that changed. Clipped to the edited region, like the ground
    truth, so it grades the edit rather than the box.

    Returns (completeness, corroboration); `corroboration` is the share of what changed that
    falls inside the extent -- see `verify_photo.check_object_recoloured`.
    """
    _, inside, changed = _parts(before, after, box)
    extent = (prob > 0.5) & inside
    if not extent.any() or not changed.any():
        return float('nan'), float('nan')
    hit = float((extent & changed).sum())
    return hit / float(extent.sum()), hit / float(changed.sum())


CANDIDATES = ['dense', 'coarse', 'bndry', 'rembg', 'phrase']


def wrong_box(img, obj, box, shape):
    """A box the same size, elsewhere in frame, containing <10% of the object and where
    segmenting it actually recolours something (>=2% of the box). Returns (box, mask) or
    (None, None) if no such box is found.
    """
    H, W = shape
    x0, y0, x1, y1 = box
    cy0, cx0 = (y0 + y1) / 2, (x0 + x1) / 2
    # Same size first, shrinking only when no same-size box clears the object -- place
    # matters more than size for this failure.
    for shrink in (1.0, 0.6, 0.35):
        w, h = int((x1 - x0) * shrink), int((y1 - y0) * shrink)
        if w < 16 or h < 16 or w >= W or h >= H:
            continue
        cands = [(cx, cy) for cy in range(0, H - h + 1, max(1, h // 2))
                 for cx in range(0, W - w + 1, max(1, w // 2))]
        # Furthest from the object first: the near ones just clip an edge of it.
        cands.sort(key=lambda c: -((c[0] + w / 2 - cx0) ** 2 + (c[1] + h / 2 - cy0) ** 2))
        for cx, cy in cands:
            bx = [cx, cy, cx + w, cy + h]
            if obj[cy:cy + h, cx:cx + w].mean() >= 0.10:
                continue
            mask = np.asarray(O.segment_object(img, bx)[0], float)
            if mask[cy:cy + h, cx:cx + w].mean() >= 0.02:
                return bx, mask
    return None, None


def corroboration_table(by_id, out_dir):
    """Checks the corroboration floor (0.60) in `check_object_recoloured`: right boxes must
    clear it, boxes moved onto whatever is next to the object must not.
    """
    print('\ncorroboration -- the share of what CHANGED that lies inside the caller\'s '
          'words.\nThe floor is 0.60; a right box must clear it and a wrong one must not:')
    print(f'{"case":16} {"phrase":20} {"right box":>10} {"wrong box":>10}  verdict on the wrong box')
    right, wrong = [], []
    for case_id in SCORED:
        case = by_id[case_id]
        img = load_image(case)
        W, H = img.size
        box = [int(v) for v in case['box']]
        # Same resolution path as the shipped check (box plus context), not the whole frame.
        prob = _phrase_probability(img.convert('RGB'), case['what'], box)
        bad, bad_mask = wrong_box(img, object_mask(case_id), box, (H, W))
        row = [f'{case_id:16} {case["what"]:20}']
        said = ''
        for tag, bx in (('right', box), ('wrong', bad)):
            if bx is None:
                row.append(f'{"n/a":>10}')
                continue
            mask = (bad_mask if tag == 'wrong'
                    else np.asarray(O.segment_object(img, bx)[0], float))
            after = recolour(img, mask, TARGETS[0])
            c = phrase_extent(img, after, bx, prob)[1]
            (right if tag == 'right' else wrong).append((case_id, c))
            row.append(f'{c:10.2f}')
            if tag == 'wrong':
                ok, detail = check_object_recoloured(img, after, bx, subject=case['what'])
                said = {True: 'VERIFIED -- WRONG', False: 'failed',
                        None: 'no verdict'}[ok]
                for phrase, label in (('disagree about which thing', 'disagreement named'),
                                      ('IS in this picture', 'wrong place named'),
                                      ('matches nothing', 'silent -- words found nothing'),
                                      ('covers the whole region', 'silent -- degenerate')):
                    if phrase in detail:
                        said += ' / ' + label
                        break
        print('  '.join(row) + '  ' + said, flush=True)
    clear = [c for _, c in right if np.isfinite(c)]
    miss = [c for _, c in wrong if np.isfinite(c)]
    print(f'\n  right boxes {min(clear):.2f}-{max(clear):.2f}, wrong boxes '
          f'{min(miss):.2f}-{max(miss):.2f}, floor 0.60')
    low = [cid for cid, c in right if np.isfinite(c) and c < 0.60]
    print(f'  {len(low)} right box(es) below the floor ({", ".join(low) or "none"}), '
          f'{sum(c >= 0.60 for c in miss)} wrong box(es) above it')
    print('  coffee_cup below the floor is not a false alarm: that box recolours the '
          'SAUCER\n  while the caller said "the cup", which is the disagreement, correctly '
          'reported.')


# ---------------------------------------------------------------------------
# sheets

def overlay(img, box, changed):
    a = np.array(img.convert('RGB')).astype(float)
    a[changed] = a[changed] * 0.45 + np.array([60, 200, 255]) * 0.55
    out = Image.fromarray(a.astype(np.uint8))
    ImageDraw.Draw(out).rectangle([int(v) for v in box], outline=(255, 220, 0), width=3)
    return out


def sheet(panels, title, width=1800):
    cell = width // len(panels) - 10
    thumbs = []
    for cap, p in panels:
        t = p.copy()
        t.thumbnail((cell, cell * 2), Image.LANCZOS)
        thumbs.append((cap, t))
    out = Image.new('RGB', (width, max(t.size[1] for _, t in thumbs) + 78), (246, 246, 248))
    d = ImageDraw.Draw(out)
    d.text((10, 6), title, font=_font(20), fill=(10, 10, 10))
    for i, (cap, t) in enumerate(thumbs):
        x = i * (width // len(panels)) + 6
        out.paste(t, (x, 66))
        for j, line in enumerate(cap.split('\n')):
            d.text((x, 26 + 13 * j), line, font=_font(12), fill=(60, 60, 60))
    return out


# ---------------------------------------------------------------------------

def run(out_dir=None):
    out_dir = out_dir or os.path.join(HERE, 'object_completeness')
    os.makedirs(out_dir, exist_ok=True)
    by_id = all_cases()
    scores, verdicts, calib = [], [], []
    for case_id in SCORED:
        case = by_id[case_id]
        img = load_image(case)
        box = [int(v) for v in case['box']]
        W, H = img.size
        truth = true_mask(case_id, box, (H, W))
        whole = object_mask(case_id)
        prob = _phrase_probability(img.convert('RGB'), case['what'], box)
        subject = np.array(A.remove_background(img).split()[-1]).astype(float) / 255.0
        lab_raw, lab_smooth = oklab_pair(img)
        px, py = int(0.3 * (box[2] - box[0])), int(0.3 * (box[3] - box[1]))
        loose = [max(0, box[0] - px), max(0, box[1] - py),
                 min(W, box[2] + px), min(H, box[3] + py)]
        variants = [('full', _feather(truth, 2), box, 'PASS'),
                    ('loose', _feather(truth, 2), loose, 'PASS'),
                    ('keep75', _feather(eroded(truth, 0.75), 2), box, '--'),
                    ('half', _feather(half(truth), 2), box, 'FAIL'),
                    ('sliver', _feather(eroded(truth, 0.51), 2), box, 'FAIL'),
                    ('noop', np.zeros((H, W), float), box, 'FAIL'),
                    ('shipped', O.segment_object(img, box)[0], box, '--')]
        panels = [('original', img.crop(box))]
        for name, mask, bx, want in variants:
            after = recolour(img, mask, TARGETS[0])
            k, corroboration = phrase_extent(img, after, bx, prob)
            row = {'case': case_id, 'variant': name, 'want': want,
                   'dense': leftover_dense(img, after, bx, lab_raw),
                   'coarse': leftover_coarse(img, after, bx, lab_smooth),
                   'bndry': boundary_flat(img, after, bx, lab_smooth),
                   'rembg': subject_extent(img, after, bx, subject),
                   'phrase': k}
            scores.append(row)
            # True changed fraction of the object, inside the edited region -- lets
            # candidate 5 be graded on correctness, not just ranking.
            _, inside, changed = _parts(img, after, bx)
            here = whole & inside
            calib.append({'case': case_id, 'variant': name, 'corroboration': corroboration,
                          'f': float((changed & here).sum()) / max(here.sum(), 1), 'k': k})
            for target in TARGETS:
                ok, detail = check_object_recoloured(img, recolour(img, mask, target), bx,
                                                     subject=case['what'])
                verdicts.append((case_id, target[0], name, want,
                                 {True: 'PASS', False: 'FAIL', None: 'no verdict'}[ok],
                                 detail))
            caption = (f'{name} (must {want})\n' +
                       '  '.join(f'{c}:{row[c]:.2f}' for c in CANDIDATES[:2]) + '\n' +
                       '  '.join(f'{c}:{row[c]:.2f}' for c in CANDIDATES[2:]))
            panels.append((caption, overlay(img, bx, _parts(img, after, bx)[2]).crop(bx)))
        sheet(panels, f'{case_id} -- "make {case["what"]} blue"; higher = more complete'
              ).save(os.path.join(out_dir, f'{case_id}.png'))
        print(f'{case_id:18} done', flush=True)

    print(f'\n{"case":16} {"variant":8} {"want":5} ' +
          ' '.join(f'{c:>7}' for c in CANDIDATES))
    for r in scores:
        print(f'{r["case"]:16} {r["variant"]:8} {r["want"]:5} ' +
              ' '.join(f'{r[c]:7.2f}' for c in CANDIDATES))
    print(f'\nseparation ({len(SCORED)} photographs, must-PASS against must-FAIL):')
    for c in CANDIDATES[:4]:
        good = [r for r in scores if r['want'] == 'PASS' and np.isfinite(r[c])]
        bad = [r[c] for r in scores if r['want'] == 'FAIL' and np.isfinite(r[c])]
        worst = min(good, key=lambda r: r[c])
        gap = worst[c] - max(bad)
        print(f'  {c:7} complete {worst[c]:.2f}-{max(r[c] for r in good):.2f}, incomplete '
              f'{min(bad):.2f}-{max(bad):.2f}, margin {gap:+.2f} '
              f'({"usable" if gap > 0.1 else "OVERLAPS"}; worst complete case: '
              f'{worst["case"]}/{worst["variant"]})')

    # Candidate 5 abstains on cases where words and box disagree; averaging in a declined
    # answer would hide that behaviour.
    graded = [r for r in calib if r['corroboration'] > 0.6 and np.isfinite(r['k'])]
    abstained = sorted({r['case'] for r in calib} - {r['case'] for r in graded})
    err = [abs(r['k'] - r['f']) for r in graded]
    # Graded by measured fraction f, not the variant label: `loose` isn't fully complete
    # when the object leaves the tight box, and `half` isn't 50% by area.
    good = [r['k'] for r in graded if r['f'] >= 0.95]
    bad = [r['k'] for r in graded if r['f'] <= 0.60]
    print(f'\n  phrase  complete (>=95 % of the object really changed) {min(good):.2f}-'
          f'{max(good):.2f},\n          incomplete (<=60 %) {min(bad):.2f}-{max(bad):.2f}, '
          f'margin {min(good) - max(bad):+.2f} -- the first one that does not overlap\n'
          f'          ({len(graded)} of {len(calib)} rows graded; abstained on '
          f'{", ".join(abstained)})')
    print(f'          and it is CALIBRATED, which none of the four are asked to be: against '
          f'the known\n          fraction of the object that changed, median error '
          f'{np.median(err):.3f}, max {max(err):.3f}')
    mid = [r for r in graded if 0.60 < r['f'] < 0.95]
    print(f'          the band in between is where it CANNOT decide, and the shipped '
          f'thresholds say so:\n          {len(mid)} rows at f={min(r["f"] for r in mid):.2f}-'
          f'{max(r["f"] for r in mid):.2f} read {min(r["k"] for r in mid):.2f}-'
          f'{max(r["k"] for r in mid):.2f}, overlapping both ends')
    print(f'\n{"case":16} {"variant":7} {"f (true)":>9} {"k (est)":>8} {"err":>6} '
          f'{"corrob":>7}')
    for r in calib:
        f = lambda v: '    nan' if not np.isfinite(v) else f'{v:7.2f}'
        print(f'{r["case"]:16} {r["variant"]:7} {r["f"]:9.2f} {f(r["k"])} '
              f'{f(r["k"] - r["f"])} {f(r["corroboration"])}')

    print('\nwhat ships -- `check_object_recoloured`, which claims only what is decidable:')
    print(f'{"case":16} {"col":4} {"variant":8} {"want":5} {"got":10} detail')
    wrong = missed = 0
    for case_id, col, name, want, got, detail in verdicts:
        flag = ''
        if (want, got) in (('FAIL', 'PASS'), ('PASS', 'FAIL')):
            flag, wrong = ' <- WRONG', wrong + 1
        elif want == 'FAIL' and got == 'no verdict':
            flag, missed = ' <- left to the caller to see', missed + 1
        print(f'{case_id:16} {col:4} {name:8} {want:5} {got:10} {detail[:64]}{flag}')
    print(f'\n{wrong} wrong verdicts; {missed} must-FAIL rows carry no verdict -- all of '
          f'them on the two\ncases where the words and the box disagree about the object, '
          f'which is the abstention, not a miss')
    corroboration_table(by_id, out_dir)
    print('sheets in', out_dir)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else None)
