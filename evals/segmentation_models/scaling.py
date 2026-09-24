"""How each method's cost grows with image size, same box at three resolutions.
GrabCut's cost is driven by pixel count inside the box; a SAM encoder always sees
1024x1024, so its cost and mask are the same at every scale.

    python evals/segmentation_models/scaling.py grabcut    # the shipped backends
    python evals/segmentation_models/scaling.py slimsam     # (needs figsurgeon[grounding])
    /tmp/samenv/bin/python evals/segmentation_models/scaling.py facebook/sam2.1-hiera-tiny
"""
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as C                                                        # noqa: E402

BASE = {'id': 'food_pizza', 'source': 'corpus', 'name': 'food_pizza',
        'box': [40, 25, 1275, 975], 'what': 'the pizza'}


def run(model_id=None):
    full = C.image(BASE)                     # already thumbnailed to WORKING on the long side
    from evals import real_corpus
    full = real_corpus.load('food_pizza')
    for long_side in (700, 1400, 2400):
        img = full.copy()
        img.thumbnail((long_side, long_side), Image.LANCZOS)
        sx, sy = img.size[0] / 1400, img.size[1] / 1050
        box = [BASE['box'][0] * sx, BASE['box'][1] * sy,
               BASE['box'][2] * sx, BASE['box'][3] * sy]
        mp = img.size[0] * img.size[1] / 1e6
        if model_id and '/' in model_id:          # a raw checkpoint, in a torch-only env
            import run_models
            run_models.load(model_id)
            t = time.time()
            mask, _ = run_models.mask_from_box(img, box, model_id)
            print(f'{model_id:28} {mp:4.1f} MP  {time.time() - t:6.2f}s  '
                  f'mask {mask.mean():.1%}')
        else:
            from figsurgeon import objects as O
            backend = model_id or 'grabcut'
            O.segment_object(img, (0, 0, 8, 8), backend=backend)   # warm: load, don't time
            t = time.time()
            O.segment_object(img, tuple(int(v) for v in box), backend=backend)
            print(f'{backend:28} {mp:4.1f} MP  {time.time() - t:6.2f}s')


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else None)
