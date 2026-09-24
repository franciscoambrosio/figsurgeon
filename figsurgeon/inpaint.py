"""Fill a masked region with what was probably behind it. Two backends, one promise: pixels
outside the mask are never touched.

`cv2.inpaint` (TELEA) reconstructs a masked region by propagating colour inward from its
nearest unmasked neighbours. That works when the hole is thin and its surroundings are
uniform -- a mast against sky, a crack, a small speck. It cannot reconstruct an object:
the structure behind a horse is not present anywhere along its outline, so the fill is the
horse's own silhouette painted in surrounding colour. Measured on 24 real erases
(`evals/erase_field.py`), one came back clean -- a traffic cone on flat asphalt, the case
TELEA is for.

`backend='lama'` is a model that predicts the missing region instead of propagating into it,
so it can put grass, tiles or a wall where an object was. Weights are a ~200 MB TorchScript
file, Apache-2.0, cached under `~/.cache/figsurgeon` and downloaded on first use.

LaMa's failure mode is different: it can invent plausible structure that was never there --
erase a horse and it may reconstruct a bush behind it that did not exist. A fill that looks
like background is not evidence that it is the background; `verify_photo.check_object_erased`
has to catch both failure modes.

The raw LaMa export is handed the whole frame and returns a whole frame. Measured on four
photographs (horse, door, cone, boat), its return is byte-identical to the input outside the
mask, so it already composites internally. This module composites explicitly anyway rather
than rely on that undocumented property holding across model exports --
`tests/test_inpaint.py` fails if a future export stops preserving it.
"""
import os
import urllib.request

import numpy as np

# TorchScript export of big-lama (Suvorov et al., 2021), Apache-2.0. Hosted on the
# simple-lama-inpainting release page; the package itself is not a dependency -- it pins
# numpy and Pillow back several major versions and installs a second, conflicting OpenCV
# build. Loading the file is `torch.jit.load` plus a pad to a multiple of 8, so the four
# lines below are cheaper than the dependency.
MODEL_URL = ('https://github.com/enesmsahin/simple-lama-inpainting/releases/download/'
             'v0.1.0/big-lama.pt')
CACHE_DIR = os.path.expanduser('~/.cache/figsurgeon')
CACHE_PATH = os.path.join(CACHE_DIR, 'big-lama.pt')

_MODEL = []


def _require():
    try:
        import torch                                                     # noqa: F401
    except ImportError:
        raise ImportError(
            'Filling an erased region with predicted background needs a model. Install it '
            'with: pip install "figsurgeon[grounding]"  (torch, CPU is fine; the weights '
            'are ~200 MB and download on first use). Without it, erasing falls back to '
            'cv2 TELEA, which smears large objects rather than removing them.')


def available():
    """True when `backend='lama'` can run. `erase_object(backend='auto')` asks this."""
    try:
        _require()
        return True
    except ImportError:
        return False


def _weights():
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = CACHE_PATH + '.part'          # so an interrupted download is not cached
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, CACHE_PATH)
    return CACHE_PATH


def _model():
    if not _MODEL:
        _require()
        import torch
        _MODEL.append(torch.jit.load(_weights(), map_location='cpu').eval())
    return _MODEL[0]


def _pad8(a):
    """LaMa's convolutions need both sides to be a multiple of 8."""
    pad = [(0, (8 - a.shape[0] % 8) % 8), (0, (8 - a.shape[1] % 8) % 8)]
    return np.pad(a, pad + [(0, 0)] * (a.ndim - 2), mode='symmetric')


def lama(rgb, binary):
    """Predict what was behind `binary` in `rgb` (both numpy, uint8). Returns uint8 RGB.

    The return is `rgb` everywhere `binary` is 0 -- byte-identical, not merely similar.
    """
    import torch
    h, w = rgb.shape[:2]
    m = (binary > 0)
    if not m.any():
        return rgb.copy()
    image = torch.from_numpy(_pad8(rgb)).permute(2, 0, 1).float()[None] / 255.
    mask = torch.from_numpy(_pad8(m.astype(np.uint8)))[None, None].float()
    with torch.no_grad():
        out = _model()(image, mask)
    # The model's own range is [0, 1] despite the uint8-looking values some exports return;
    # clipping before the scale would flatten the whole frame to black.
    filled = np.clip(out[0].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
    result = rgb.copy()
    result[m] = filled[:h, :w][m]
    return result
