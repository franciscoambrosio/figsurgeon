"""For each corpus image, fit every candidate matplotlib colormap against a sample of its
colourful pixels and report the best mean nearest-LUT distance. A far best fit (e.g. > 25)
means the figure uses a colormap not in CANDIDATES."""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from figsurgeon import rebrand  # noqa: E402

CACHE = os.path.join(ROOT, 'evals', '_corpus', 'audit_remap')

CANDIDATES = ['viridis', 'plasma', 'magma', 'inferno', 'cividis', 'turbo', 'jet', 'Greys',
              'coolwarm', 'RdBu_r', 'RdYlBu_r', 'Spectral_r']


def best_fit(path, max_samples=30000, seed=0):
    img = Image.open(path).convert('RGB')
    a = np.asarray(img).astype(float).reshape(-1, 3)
    colourful = (a.max(axis=1) - a.min(axis=1)) > 30
    pix = a[colourful]
    if len(pix) == 0:
        return None, {}
    rng = np.random.default_rng(seed)
    if len(pix) > max_samples:
        pix = pix[rng.choice(len(pix), max_samples, replace=False)]
    results = {}
    for cmap in CANDIDATES:
        lut = rebrand._cmap_lut(cmap)
        _, dist = rebrand._nearest_in_lut(pix, lut)
        results[cmap] = float(dist.mean())
    best = min(results, key=results.get)
    return best, results


if __name__ == '__main__':
    names = sorted(n[:-4] for n in os.listdir(CACHE) if n.endswith('.png'))
    for name in names:
        path = os.path.join(CACHE, name + '.png')
        best, results = best_fit(path)
        if best is None:
            print(f'{name:30s}  no colourful pixels')
            continue
        ranked = sorted(results.items(), key=lambda kv: kv[1])[:3]
        print(f'{name:30s}  best={best:10s} {results[best]:6.2f}   '
              + '  '.join(f'{k}={v:.1f}' for k, v in ranked))
