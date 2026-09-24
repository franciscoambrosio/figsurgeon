import os, sys, json, time, traceback
import numpy as np

SP = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'audit_phrase')
sys.path.insert(0, SP)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cases import CASES, get_image
from figsurgeon import grounding, objects, locate

OUT = os.path.join(SP, 'overlays')
os.makedirs(OUT, exist_ok=True)
RESULTS_PATH = os.path.join(SP, 'results.json')


def bbox(hard):
    ys, xs = np.where(hard)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def run():
    results = []
    done = set()
    if os.path.exists(RESULTS_PATH):
        results = json.load(open(RESULTS_PATH))
        done = {r['id'] for r in results}
        print(f'resuming: {len(done)} cases already done')
    t0 = time.time()
    for i, (cid, img_name, phrase, kind, expected, note) in enumerate(CASES):
        if cid in done:
            continue
        row = {'id': cid, 'image': img_name, 'phrase': phrase, 'kind': kind,
               'expected_referent': expected, 'note': note}
        try:
            img = get_image(img_name)
            W, H = img.size
            mask, gnote = grounding.mask_for(img, phrase)
            row['grounding_note'] = gnote
            if mask is None:
                row['status'] = 'refused'
                row['iou'] = None
                results.append(row)
                print(f'[{i+1}/{len(CASES)}] {cid:14s} REFUSED  {gnote[:80]}')
                # still render a frame-probability overlay to look at, so a refusal can
                # be judged "correct refusal" vs "wrongly refused" by eye
                prob = grounding._probability(img.convert('RGB'), phrase)
                sheet = locate.mask_overlay(img, (prob > 0.3).astype(float))
                sheet.save(os.path.join(OUT, f'{cid}.jpg'), quality=82, optimize=True)
                continue

            hard = mask > 0.5
            frac = float(hard.mean())
            row['phrase_frac'] = frac
            box = bbox(hard)
            if box is None or (box[2]-box[0]) < 8 or (box[3]-box[1]) < 8:
                row['status'] = 'too_small'
                row['iou'] = None
                results.append(row)
                print(f'[{i+1}/{len(CASES)}] {cid:14s} TOO SMALL frac={frac:.4f}')
                continue

            seg_mask, coverage = objects.segment_object(img, box)
            seg_hard = np.asarray(seg_mask) > 0.5
            inter = float((hard & seg_hard).sum())
            union = float((hard | seg_hard).sum())
            iou = inter / union if union > 0 else 0.0
            row['status'] = 'measured'
            row['iou'] = iou
            row['seg_coverage'] = coverage
            row['box'] = box

            x0, y0, x1, y1 = box
            pad = max(20, int(0.15 * max(x1 - x0, y1 - y0)))
            crop_box = (max(0, x0 - pad), max(0, y0 - pad),
                        min(W, x1 + pad), min(H, y1 + pad))
            sheet = locate.mask_overlay(img, mask).crop(crop_box)
            sheet.save(os.path.join(OUT, f'{cid}.jpg'), quality=82, optimize=True)

            results.append(row)
            print(f'[{i+1}/{len(CASES)}] {cid:14s} frac={frac:.4f} iou={iou:.3f} '
                  f'cov={coverage:.2f} box={box}  ({time.time()-t0:.0f}s)')
        except Exception as e:
            row['status'] = 'error'
            row['error'] = f'{type(e).__name__}: {e}'
            results.append(row)
            print(f'[{i+1}/{len(CASES)}] {cid:14s} ERROR {e}')
            traceback.print_exc()

        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)

    print('done in', time.time() - t0, 's')


if __name__ == '__main__':
    run()
