"""The five demo phrasings must parse and apply against the demo time-series figure."""
import os

import pytest

from figsurgeon import apply_text, load

pytestmark = pytest.mark.filterwarnings('ignore')

DEMOS = [
    ("highlight the MLP line", 'highlight'),
    ("keep the MLP and Offline model lines", 'highlight'),
    ("grey out everything except the offline line", 'highlight'),
    ("make the MLP line red", 'recolour'),
    ("make the offline model line thicker", 'thicken'),
]


@pytest.fixture(scope='module')
def figure():
    from timeseries_demo import SPEC
    if not os.path.exists(SPEC.path):
        pytest.skip(f'demo figure not found at {SPEC.path}')
    return load(SPEC.path), SPEC


@pytest.mark.parametrize('text,expected_verb', DEMOS)
def test_demo_phrasing_parses_and_applies(figure, text, expected_verb):
    a, spec = figure
    out, verb, kwargs, info = apply_text(a, spec, text)
    assert verb == expected_verb, f'{text!r} routed to {verb}, expected {expected_verb}'
    assert out.shape == a.shape
    # An edit that changes nothing at all means the series was never matched.
    assert (out != a).any(), f'{text!r} left the figure byte-identical'


def test_recolour_leaves_the_protected_split_marker_alone(figure):
    """Recolouring `offline` must not repaint its dash-dot split marker (a ProtectRun):
    only where the line itself crosses the marker's columns may turn red."""
    from figsurgeon import edit
    a, spec = figure
    out, _ = edit.recolour_series(a, spec, 'offline', (200, 0, 0))
    strip = out[46:424, 609:620].astype(int)
    red = (strip[..., 0] - strip[..., 2]) > 60
    assert red.sum() < 200, f'{red.sum()} red pixels in the marker strip'
