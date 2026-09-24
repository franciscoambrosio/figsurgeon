"""Demo: five openly-licensed photos (CC0 / public domain, via scikit-image's sample data),
each showing an effect actually suited to that photo.

Run: python examples/demo_photos.py   (requires: pip install scikit-image)
"""
import os
from skimage import data
from PIL import Image
import numpy as np
from figsurgeon import photo as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT, exist_ok=True)

jobs = [
    ('chelsea', 'colour pop the eyes',
     lambda im: P.isolate_colour(im, colour_name_or_rgb=0.11, hue_tol=0.022,
                                 sat_floor=0.2, feather=2, flatten=0.6)[0]),
    ('coffee', 'blur_background(focus=cup+saucer bbox)',
     lambda im: P.blur_background(im, focus=(70, 10, 420, 280), radius=16)),
    ('rocket', 'colour pop the warm lights',
     lambda im: P.isolate_colour(im, colour_name_or_rgb=0.10, hue_tol=0.09,
                                 sat_floor=0.12, feather=3, flatten=0.6)[0]),
    ('colorwheel', 'colour pop the red (stylised, flatten=0.6)',
     lambda im: P.isolate_colour(im, colour_name_or_rgb=(220, 30, 30), hue_tol=0.055,
                                 feather=4, flatten=0.6)[0]),
    ('chelsea', 'pencil sketch (blur_radius=10)',
     lambda im: P.sketch(im, blur_radius=10, contrast=1.15)),
]

for i, (name, label, fn) in enumerate(jobs):
    arr = getattr(data, name)()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    im = Image.fromarray(arr)
    out = fn(im)
    im.save(os.path.join(OUT, f'demo_{i}_{name}_orig.png'))
    out.save(os.path.join(OUT, f'demo_{i}_{name}_edit.png'))
    print(f'{name}: {label} -> saved demo_{i}_{name}_edit.png')
