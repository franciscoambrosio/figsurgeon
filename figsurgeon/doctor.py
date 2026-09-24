"""`figsurgeon doctor` -- what optional machinery is present, and what its absence costs.

Everything the core package needs (numpy, scipy, Pillow) is a hard dependency and always
there if the package imports at all. Everything else is an extra: present or not depending
on how it was installed, and each absence changes what a caller can ask for rather than
crashing outright. This prints that state in one place instead of making someone discover it
tool call by tool call.
"""
import os
import shutil


def _try_import(*modules):
    try:
        for m in modules:
            __import__(m)
        return True
    except ImportError:
        return False


def checks():
    """List of (label, ok, cost_if_missing) -- ok is True/False, never raises."""
    from . import grounding, inpaint

    rows = []

    grounding_ok = grounding.available()
    rows.append((
        'grounding extra (torch + transformers)', grounding_ok,
        'phrases like "the sky" or "the flowers" are refused (no CLIPSeg), and '
        'segment_object / backend="auto" falls back to GrabCut instead of SlimSAM. '
        'Install: pip install "figsurgeon[grounding]"'))

    advanced_ok = _try_import('cv2') and _try_import('rembg')
    rows.append((
        'advanced extra (opencv + rembg)', advanced_ok,
        'remove_background, replace_background, remove_object, denoise are unavailable. '
        'Install: pip install "figsurgeon[advanced]"'))

    matplotlib_ok = _try_import('matplotlib')
    pytesseract_ok = _try_import('pytesseract')
    tesseract_bin = shutil.which('tesseract') is not None
    rebrand_ok = matplotlib_ok and pytesseract_ok and tesseract_bin
    missing = []
    if not (matplotlib_ok and pytesseract_ok):
        missing.append('pip install "figsurgeon[rebrand]"')
    if not tesseract_bin:
        missing.append('the tesseract BINARY (pip cannot install this) -- '
                        'on Debian/Ubuntu: apt-get install tesseract-ocr')
    rows.append((
        'rebrand extra (matplotlib + pytesseract + tesseract binary)', rebrand_ok,
        'colormap remapping and OCR-based font swap are unavailable. Need: '
        + '; and '.join(missing) if missing else ''))

    lama_weights = os.path.exists(inpaint.CACHE_PATH)
    lama_ok = inpaint.available() and lama_weights
    if not inpaint.available():
        lama_cost = 'erase_object falls back to cv2 TELEA, which smears large objects. ' + \
            'Install: pip install "figsurgeon[grounding]" (torch)'
    elif not lama_weights:
        lama_cost = ('erase_object falls back to cv2 TELEA until the ~200 MB weights '
                     'download on first use (automatic, no action needed)')
    else:
        lama_cost = ''
    rows.append(('LaMa inpainting weights (~200 MB, cached on first use)', lama_ok, lama_cost))

    key_ok = bool(os.environ.get('OPENROUTER_API_KEY'))
    rows.append((
        'OPENROUTER_API_KEY', key_ok,
        'generative.fill and erase_object\'s hosted fallback are unavailable. '
        'Set: export OPENROUTER_API_KEY=...'))

    return rows


def main():
    print('figsurgeon doctor\n')
    rows = checks()
    for label, ok, cost in rows:
        mark = 'OK  ' if ok else 'MISSING'
        print(f'[{mark}] {label}')
        if not ok and cost:
            print(f'         {cost}')
    n_missing = sum(1 for _, ok, _ in rows if not ok)
    print(f'\n{len(rows) - n_missing}/{len(rows)} present.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
