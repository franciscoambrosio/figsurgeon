"""`inpaint.lama` and `erase_object(fill=...)`, driven without the 200 MB model.

The stub returns a flat colour at the frame's size, standing in for an inpainter that
re-renders everything it was shown rather than only the hole, so the composite is tested
even though the real weights never actually do this.
"""
import numpy as np
import pytest
from PIL import Image

from figsurgeon import inpaint, objects

STUB_COLOUR = 7          # near-black, unmissable if it leaks outside the mask


class _Stub:
    """Stands in for the TorchScript module, returning a flat frame in [0, 1], NCHW."""

    def __init__(self):
        self.calls = 0

    def __call__(self, image, mask):
        self.calls += 1
        import torch
        return torch.full_like(image, STUB_COLOUR / 255.)


@pytest.fixture
def stub_model(monkeypatch):
    pytest.importorskip('torch')
    stub = _Stub()
    monkeypatch.setattr(inpaint, '_model', lambda: stub)
    return stub


def _scene():
    a = np.zeros((64, 80, 3), np.uint8)
    a[:, :, 1] = 200                       # a green field
    a[20:40, 30:50] = (220, 30, 30)        # with a red thing in it
    mask = np.zeros((64, 80), np.uint8)
    mask[20:40, 30:50] = 255
    return a, mask


def test_fill_lands_only_inside_the_mask(stub_model):
    a, mask = _scene()
    out = inpaint.lama(a, mask)
    m = mask > 0
    assert (out[~m] == a[~m]).all(), 'a pixel outside the mask changed'
    assert (out[m] == STUB_COLOUR).all(), 'the mask was not filled with the model answer'
    assert stub_model.calls == 1


def test_empty_mask_changes_nothing_and_does_not_run_the_model(stub_model):
    a, _ = _scene()
    out = inpaint.lama(a, np.zeros(a.shape[:2], np.uint8))
    assert (out == a).all()
    assert stub_model.calls == 0


@pytest.mark.parametrize('shape', [(63, 79), (64, 80), (65, 81)])
def test_odd_sizes_come_back_at_their_own_size(stub_model, shape):
    """The frame is padded to a multiple of 8 for the model and cropped back afterwards."""
    a = np.zeros(shape + (3,), np.uint8)
    mask = np.zeros(shape, np.uint8)
    mask[1:3, 1:3] = 255
    assert inpaint.lama(a, mask).shape == (shape[0], shape[1], 3)


def test_erase_object_fill_telea_does_not_touch_the_model(stub_model):
    img = Image.fromarray(_scene()[0])
    objects.erase_object(img, box=(28, 18, 52, 42), fill='telea')
    assert stub_model.calls == 0, 'fill="telea" reached the LaMa path'


def test_erase_object_rejects_an_unknown_fill():
    img = Image.fromarray(_scene()[0])
    with pytest.raises(ValueError, match="fill must be"):
        objects.erase_object(img, box=(28, 18, 52, 42), fill='inpaint-please')
