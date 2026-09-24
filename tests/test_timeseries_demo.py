"""Acceptance test: the package must reproduce the reference outputs for the demo figure.

The reference images are the package's own verified outputs (`pkg_mlp.png`, `pkg_offline.png`),
shipped in the repo, so the test runs identically on every checkout.
"""
import os

import numpy as np
import pytest
from PIL import Image

from figsurgeon import assert_clean, load, recolour, report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def figure():
    from timeseries_demo import SPEC
    if not os.path.exists(SPEC.path):
        pytest.skip(f'demo figure not found at {SPEC.path}')
    return load(SPEC.path), SPEC


@pytest.mark.parametrize('keep', ['mlp', 'offline'])
def test_matches_reference_build(figure, keep):
    a, spec = figure
    ref_path = os.path.join(ROOT, 'tests', 'data', f'pkg_{keep}.png')
    if not os.path.exists(ref_path):
        pytest.skip(f'reference image {ref_path} not present')

    out, info = recolour(a, spec, keep)
    ref = np.array(Image.open(ref_path).convert('RGB')).astype(int)
    assert out.shape == ref.shape
    ndiff = int((np.abs(out.astype(int) - ref).max(axis=2) > 0).sum())
    assert ndiff == 0, f'keep={keep}: {ndiff} px differ from the reference build'


@pytest.mark.parametrize('keep', ['mlp', 'offline'])
def test_invariants_hold(figure, keep):
    a, spec = figure
    out, info = recolour(a, spec, keep)
    rep = report(a, out, spec, keep)
    assert_clean(rep)
    assert rep[f'series[{keep}]_survivors'] > 0, 'the kept series vanished entirely'
