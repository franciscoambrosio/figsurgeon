"""Overlay the 'left on old scale' mask in magenta on the AFTER image (full frame, not
cropped) so the spatial pattern (a band vs scattered noise vs a whole region) can be judged
by eye without a huge unreadable crop."""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from figsurgeon import rebrand  # noqa: E402

CACHE = os.path.join(ROOT, 'evals', '_corpus', 'audit_remap')
OUT = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'audit_remap')


def overlay(name, fname, cmap, box=None, max_w=900):
    before = Image.open(os.path.join(CACHE, fname)).convert('RGB')
    after = Image.open(os.path.join(OUT, 'renders', name + '.png')).convert('RGB')
    a = np.asarray(before).astype(float)
    c = np.asarray(after).astype(float)
    if box is None:
        box = (0, 0, a.shape[1], a.shape[0])
    x0, y0, x1, y1 = box
    A, C = a[y0:y1, x0:x1], c[y0:y1, x0:x1]
    changed = np.abs(A - C).max(axis=2) > 2
    src = rebrand._cmap_lut(cmap)
    idx_before, dist_before = rebrand._nearest_in_lut(A.reshape(-1, 3), src)
    dist_before = dist_before.reshape(A.shape[:2])
    colourful = (A.max(axis=2) - A.min(axis=2)) > 30
    left = (~changed) & colourful & (dist_before < 60)
    out = c.copy()
    sub = out[y0:y1, x0:x1]
    sub[left] = [255, 0, 255]
    out[y0:y1, x0:x1] = sub
    img = Image.fromarray(np.clip(out, 0, 255).astype('uint8'))
    if img.width > max_w:
        h = int(img.height * max_w / img.width)
        img = img.resize((max_w, h))
    path = os.path.join(OUT, 'renders', name + '_overlay.png')
    img.save(path)
    print(name, 'n_left', int(left.sum()), '->', path)


if __name__ == '__main__':
    cases = [
        ('photon_jet', 'photon_jet.png', 'jet', None),
        ('photon_mfp_jet', 'photon_mfp_jet.png', 'jet', None),
        ('cloudless_desert_turbo', 'cloudless_desert_turbo.png', 'turbo', None),
        ('gliese_turbo', 'gliese_turbo.png', 'turbo', None),
        ('gw170817_spectro', 'gw170817_spectro.png', 'viridis', None),
        ('fertile_crescent_viridis', 'fertile_crescent_viridis.png', 'viridis', None),
        ('cretaceous_jet', 'cretaceous_jet.png', 'jet', None),
        ('radar_reflectivity_custom', 'radar_reflectivity_custom.png', 'jet', None),
        ('gw_transient_wrong', 'gw_transient_catalog.png', 'inferno', None),
        ('hr_diagram_no_cmap', 'hr_diagram.png', 'viridis', None),
        ('kangerlussuaq_jet', 'kangerlussuaq_jet.png', 'jet', None),
        ('wiki_depth_scatter', 'wiki_depth_scatter.png', 'viridis', None),
        ('rho_oph_scatter', 'rho_oph_scatter.png', 'viridis', None),
        ('hunter_gatherer_pop', 'hunter_gatherer_pop.png', 'viridis', None),
        ('rossmo_custom', 'rossmo_custom.png', 'viridis', None),
        ('gliese_profile_custom', 'gliese_profile_viridis.png', 'viridis', None),
        ('harappan_no_cmap', 'harappan_minerals.png', 'viridis', None),
    ]
    for name, fname, cmap, box in cases:
        overlay(name, fname, cmap, box)
