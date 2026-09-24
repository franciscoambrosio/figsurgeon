"""Checks whether a signal separates degenerate segmentations (mask collapses to the box
rectangle, no object found) from correct ones, since the shipped check (`tools.py`
recolour_object, coverage > 0.95) cannot tell them apart alone.

    python evals/segmentation_models/degenerate_masks.py

For each hand-labelled case (`degenerate`, from overlays in `sheets_degenerate/`), runs
`objects.segment_object(backend='auto')` and records coverage, box-IoU, mask solidity,
component count, edge/gradient signal, and (SlimSAM only) predicted IoU, margin over the
runner-up, and stability under small box jitters. Edge/gradient and solidity do not
separate the two groups; the model-confidence columns are the remaining candidate.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

CORPUS = os.path.join(ROOT, 'evals', '_corpus')
SHEETS = os.path.join(HERE, 'sheets_degenerate')
WORKING = 1200

# name, source jpg, box (WORKING-thumbnail coords), hypothesis, degenerate (True/False/None)
CASES = [
    ('city_sky', 'city_wide.jpg', (0, 0, 700, 220), 'box wholly in sky gradient', True),
    ('city_water', 'city_wide.jpg', (0, 1000, 780, 1180), 'box wholly in rippled water', None),
    ('city_railing', 'city_wide.jpg', (0, 795, 780, 840), 'thin railing, low contrast', False),
    ('nightcity_sky', 'night_city.jpg', (0, 0, 300, 150), 'box wholly in sky gradient', True),
    ('nightcity_foliage', 'night_city.jpg', (0, 560, 280, 750), 'dark repeating foliage', None),
    ('flower_petal', 'flower_macro.jpg', (0, 400, 220, 700), 'box wholly in one petal', False),
    ('flower_seedhead', 'flower_macro.jpg', (650, 420, 950, 680), 'repeating seed-head bumps',
     False),
    ('flower_bokeh', 'flower_macro.jpg', (0, 0, 280, 160), 'out-of-focus background', None),
    ('shoe_carpet', 'product_white.jpg', (750, 0, 1020, 150), 'repeating carpet texture', True),
    ('shoe_body', 'product_white.jpg', (560, 140, 950, 640),
     'white shoe on similar-toned carpet, low contrast', False),
    ('shoe_blackbg', 'product_shoe.jpg', (0, 0, 380, 85), 'flat black studio background', True),
    ('umbrella_flat', 'portrait_studio.jpg', (0, 0, 280, 180), 'flat blue umbrella fabric',
     False),
    ('umbrella_spokes', 'portrait_studio.jpg', (0, 90, 700, 260),
     'thin spokes, mostly background', None),
    ('facade_glass', 'car_red.jpg', (300, 450, 750, 595), 'reflective/transparent shop glass',
     False),
    ('facade_slats', 'car_red.jpg', (300, 215, 900, 380), 'repeating vertical slats + plants',
     True),
    ('street_pair', 'street_people.jpg', (390, 380, 650, 730), 'two overlapping pedestrians',
     None),
    ('brick_wall', 'street_people.jpg', (900, 300, 1150, 700), 'repeating brick, wholly bg',
     True),
    ('chairs_water', 'eight_boxes/chairs_blue.jpg', (0, 0, 600, 150), 'flat dark water', True),
    ('chairs_wall', 'eight_boxes/chairs_blue.jpg', (700, 40, 1100, 150), 'flat white wall', True),
    ('chairs_railing', 'eight_boxes/chairs_blue.jpg', (100, 180, 650, 340), 'thin X railing',
     False),
    ('horse_grass', 'eight_boxes/horse_white.jpg', (0, 950, 800, 1150), 'flat grass texture',
     True),
    ('horse_legs', 'eight_boxes/horse_white.jpg', (280, 700, 600, 1080),
     'thin legs, mostly grass background', None),
]


def image(relpath):
    path = os.path.join(CORPUS, relpath)
    img = Image.open(path).convert('RGB')
    img.thumbnail((WORKING, WORKING), Image.LANCZOS)
    return img


def _solidity_and_components(binary):
    """(solidity, n_components) of a 0/1 mask, cropped to its own bounding box for speed."""
    from scipy import ndimage
    from skimage.morphology import convex_hull_image

    labels, n = ndimage.label(binary)
    if n == 0:
        return 0.0, 0
    ys, xs = np.nonzero(binary)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = binary[y0:y1, x0:x1].astype(bool)
    hull = convex_hull_image(crop)
    hull_area = float(hull.sum())
    solidity = float(crop.sum()) / hull_area if hull_area else 0.0
    return solidity, n


def _box_iou(binary, box):
    H, W = binary.shape
    x0, y0, x1, y1 = box
    box_mask = np.zeros_like(binary, dtype=bool)
    box_mask[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
    m = binary.astype(bool)
    inter = (m & box_mask).sum()
    union = (m | box_mask).sum()
    return float(inter) / float(union) if union else 0.0


def _edge_signal(gray, box):
    """(edge_frac, grad_mean) of the box's interior, before segmentation. Canny thresholds
    are set from the crop's own median intensity, so dark and bright crops are comparable."""
    H, W = gray.shape
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    crop = gray[y0:y1, x0:x1]
    crop_u8 = crop.astype(np.uint8)
    med = float(np.median(crop_u8))
    lo, hi = max(0, 0.66 * med), min(255, 1.33 * med)
    edges = cv2.Canny(crop_u8, lo, hi)
    edge_frac = float((edges > 0).mean())
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    grad_mean = float(np.sqrt(gx ** 2 + gy ** 2).mean())
    return edge_frac, grad_mean


def _slimsam_forward(model, proc, img, boxes):
    """One forward pass over several boxes on one photo, sharing the encoder. Returns
    (binaries, best_scores, margins) of raw masks (before `follow_object`), including the
    model's own predicted IoU per candidate, which the production path discards."""
    import torch
    rgb = img.convert('RGB')
    inputs = proc(images=rgb, input_boxes=[[[float(v) for v in b] for b in boxes]],
                  return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs['original_sizes'].cpu(),
        inputs['reshaped_input_sizes'].cpu())[0]
    scores = out.iou_scores.cpu().numpy()[0]
    binaries, best_scores, margins = [], [], []
    for i in range(len(boxes)):
        order = np.argsort(scores[i])[::-1]
        best, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        binaries.append(np.asarray(masks[i][best]).astype(np.uint8))
        best_scores.append(float(scores[i][best]))
        margins.append(float(scores[i][best] - scores[i][second]))
    return binaries, best_scores, margins


def _jittered_boxes(box, W, H, frac=0.06):
    """Four small perturbations of `box`: shrink, grow, shift right, shift down -- each by
    `frac` of the box's own width/height, clamped to the frame."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    fx, fy = frac * w, frac * h
    variants = [
        (x0 + fx, y0 + fy, x1 - fx, y1 - fy),   # shrink
        (x0 - fx, y0 - fy, x1 + fx, y1 + fy),   # grow
        (x0 + fx, y0, x1 + fx, y1),             # shift right
        (x0, y0 + fy, x1, y1 + fy),             # shift down
    ]
    return [(max(0, int(vx0)), max(0, int(vy0)), min(W, int(vx1)), min(H, int(vy1)))
            for vx0, vy0, vx1, vy1 in variants]


def _mask_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum()) / float(union) if union else 1.0


def run():
    from figsurgeon import locate, objects as O

    model, proc = O._slimsam()
    os.makedirs(SHEETS, exist_ok=True)
    rows = []
    print(f'{"case":18s} {"backend":8s} {"area_frac":>9s} {"iou_box":>8s} '
          f'{"solidity":>9s} {"n_comp":>6s} {"edge_frac":>9s} {"grad_mean":>9s} '
          f'{"score":>6s} {"margin":>7s} {"stab":>6s} '
          f'{"degen":>5s}   hypothesis')
    for name, relpath, box, hyp, degenerate in CASES:
        img = image(relpath)
        W, H = img.size
        backend = O.choose_backend(img)
        seg = O.segment_object(img, box=box, backend='auto')
        mask, coverage = seg
        binary_pre = (mask > 0.5).astype(np.uint8)  # after follow_object (as shipped)
        solidity, n_comp = _solidity_and_components(binary_pre)
        iou = _box_iou(binary_pre, box)
        gray = np.array(img.convert('L')).astype(np.float32)
        edge_frac, grad_mean = _edge_signal(gray, box)

        score = margin = stability = float('nan')
        if backend == 'slimsam':
            jittered = _jittered_boxes(box, W, H)
            binaries, scores, margins = _slimsam_forward(model, proc, img,
                                                          [box] + jittered)
            score, margin = scores[0], margins[0]
            base = binaries[0]
            stability = float(np.mean([_mask_iou(base, b) for b in binaries[1:]]))

        x0, y0, x1, y1 = box
        pad = 30
        sheet = locate.mask_overlay(img, mask).crop(
            (max(0, x0 - pad), max(0, y0 - pad),
             min(img.size[0], x1 + pad), min(img.size[1], y1 + pad)))
        sheet.save(os.path.join(SHEETS, f'{name}.jpg'), quality=85, optimize=True)

        rows.append(dict(name=name, backend=backend, area_frac=coverage, iou_box=iou,
                          solidity=solidity, n_comp=n_comp, edge_frac=edge_frac,
                          grad_mean=grad_mean, score=score, margin=margin,
                          stability=stability, degenerate=degenerate, hyp=hyp))
        degen_str = '?' if degenerate is None else ('YES' if degenerate else 'no')
        print(f'{name:18s} {backend:8s} {coverage:9.2f} {iou:8.2f} '
              f'{solidity:9.2f} {n_comp:6d} {edge_frac:9.3f} {grad_mean:9.2f} '
              f'{score:6.3f} {margin:7.3f} {stability:6.3f} '
              f'{degen_str:>5s}   {hyp}')

    scored = [r for r in rows if r['degenerate'] is not None]
    degen = [r for r in scored if r['degenerate']]
    correct = [r for r in scored if not r['degenerate']]

    def _report(label, key):
        if degen and correct:
            print(f'{label:10s} degenerate {min(r[key] for r in degen):.3f}-'
                  f'{max(r[key] for r in degen):.3f}  vs  correct '
                  f'{min(r[key] for r in correct):.3f}-'
                  f'{max(r[key] for r in correct):.3f}')

    print()
    _report('edge_frac', 'edge_frac')
    _report('grad_mean', 'grad_mean')
    _report('score', 'score')
    _report('margin', 'margin')
    _report('stability', 'stability')

    print(f'\nsheets in {SHEETS} -- LOOK AT THEM before trusting any number here.')
    return rows


if __name__ == '__main__':
    run()
