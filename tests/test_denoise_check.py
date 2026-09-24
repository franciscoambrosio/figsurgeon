"""Denoising is graded where noise lives, not over the whole frame: whole-frame
high-frequency energy charges a detailed photograph for the detail denoising is supposed to
keep, so the check is restricted to flat regions instead.

Paired, like every other check here: correct edits that must PASS, broken ones that must
FAIL, and the degenerate inputs that used to manufacture a verdict out of arithmetic.
"""
import numpy as np
from PIL import Image, ImageFilter
from skimage import data

from figsurgeon import advanced
from figsurgeon.verify_photo import check_denoise


def noisy(sigma=12, seed=0):
    base = np.asarray(Image.fromarray(data.astronaut()).resize((256, 256))).astype(float)
    rng = np.random.default_rng(seed)
    return Image.fromarray(np.clip(base + rng.normal(0, sigma, base.shape), 0, 255)
                           .astype(np.uint8))


def test_a_real_denoise_passes():
    img = noisy()
    ok, detail = check_denoise(img, advanced.denoise(img, strength=10))
    assert ok is True, detail


def test_a_gentle_denoise_on_a_detailed_photo_passes():
    """A photo with plenty of legitimate high-frequency detail and a modest amount of noise
    removed from its flat areas must still pass."""
    img = Image.fromarray(data.astronaut())
    ok, detail = check_denoise(img, advanced.denoise(img, strength=8))
    assert ok is True, detail


def test_a_silent_no_op_fails():
    img = noisy()
    ok, detail = check_denoise(img, img)
    assert ok is False, detail
    assert '100% of it left' in detail


def test_destroying_the_image_fails():
    img = noisy()
    ok, detail = check_denoise(img, img.filter(ImageFilter.GaussianBlur(25)))
    assert ok is False, detail
    assert 'structural correlation' in detail


def test_flat_regions_with_no_noise_get_no_verdict_rather_than_a_huge_one():
    """A rendered chart's flat areas are exactly flat, so the ratio divides by nothing and
    must not be reported as a huge score."""
    a = np.full((200, 200, 3), 250, np.uint8)
    a[40:160, 40:60] = (200, 60, 60)          # a bar: edges, but perfectly flat fills
    a[40:160, 90:110] = (60, 90, 200)
    chart = Image.fromarray(a)
    ok, detail = check_denoise(chart, advanced.denoise(chart, strength=8))
    assert ok is None, detail
    assert 'nothing there to remove' in detail


def test_a_constant_image_gets_no_verdict():
    flat = Image.new('RGB', (64, 64), (128, 128, 128))
    assert check_denoise(flat, flat)[0] is None
