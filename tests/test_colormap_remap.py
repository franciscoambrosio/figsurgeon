"""Tests for colormap remapping and its verification check.

Covers: anti-aliased overlay edges (gridlines, contours, annotations) must move to the new
palette while the overlay itself stays untouched; a band of values that doesn't match the
source colormap must be reported unless spatially rescued between confident matches on both
sides; leftover colormap content outside the box must be reported as one region, not as
scattered coincidence; alpha-blended figures must be detected and verified correctly.

All figures are constructed rather than fetched, so the suite stays offline and exact.
"""
import numpy as np
import pytest
from PIL import Image

from figsurgeon import rebrand, verify_photo

NEW = ('#00407A', '#52BDEC')
BOX = (0, 0, 240, 200)


def gradient(cmap='viridis', w=240, h=200):
    """A vertical value ramp drawn in `cmap`, the way an image plot renders one."""
    lut = rebrand._cmap_lut(cmap)
    pos = np.linspace(0, len(lut) - 1, h).astype(int)
    return np.repeat(lut[pos][:, None, :], w, axis=1)


def with_white_gridline(a, x=120, width=3):
    """A white line with a soft anti-aliased edge blending white with the colormap."""
    a = a.copy()
    a[:, x - 1] = 0.5 * a[:, x - 1] + 0.5 * 255
    a[:, x:x + width] = 255
    a[:, x + width] = 0.5 * a[:, x + width] + 0.5 * 255
    return a


def img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype('uint8'))


def test_the_soft_edge_of_a_gridline_is_remapped_with_everything_else():
    """PASS: the blended edge beside a white line moves to the new gradient. FAIL: the
    line's own core, which carries no colormap value, must stay untouched."""
    before = with_white_gridline(gradient())
    out, _, _ = rebrand.remap_colormap(img(before), BOX, new_colours=NEW)
    after = np.asarray(out).astype(float)

    src = rebrand._cmap_lut('viridis')
    new = rebrand.build_gradient(NEW)
    edge = after[:, 119]
    _, d_old = rebrand._nearest_in_lut(edge, src, key='viridis')
    _, d_new = rebrand._nearest_in_lut(edge, new)
    assert (d_new < d_old).mean() > 0.9, 'the soft edge still sits on the old colour scale'

    core = after[:, 120:123]
    assert core.min() == 255, 'the gridline itself was recoloured'


def test_a_colour_that_is_not_a_blend_is_left_alone():
    """PASS: a magenta marker that is not a blend of the colormap survives unchanged.
    FAIL: the ramp around it must still be remapped."""
    before = gradient()
    before[90:110, 50:70] = (255, 0, 255)
    out, _, _ = rebrand.remap_colormap(img(before), BOX, new_colours=NEW)
    after = np.asarray(out).astype(float)

    assert np.array_equal(after[95:105, 55:65], np.full((10, 10, 3), (255, 0, 255))), \
        'the magenta marker was remapped as if it were part of the colormap'
    assert not np.array_equal(after[95:105, 150:160], before[95:105, 150:160]), \
        'nothing was remapped at all'


def test_a_clean_remap_verifies():
    """PASS: a figure whose colormap really is the one named comes back verified, with
    nothing left on the old scale."""
    before = with_white_gridline(gradient())
    out, _, _ = rebrand.remap_colormap(img(before), BOX, new_colours=NEW)
    ok, detail = verify_photo.check_colormap_remapped(img(before), out, BOX,
                                                      source_cmap='viridis',
                                                      new_colours=NEW)
    assert ok is True, detail
    assert '0.0%' in detail


def test_a_band_that_did_not_match_is_reported_as_two_colour_scales():
    """FAIL (must be caught): a band of values outside `match_tolerance` survives the
    remap and is reported as two colour scales. Placed at the edge of the range, where it
    has no flank on one side and so cannot be rescued by spatial continuity (see
    `test_a_middle_of_range_band_is_rescued_by_spatial_continuity`)."""
    before = gradient()
    before[0:40] = np.clip(before[0:40] + (30, 0, 30), 0, 255)   # not quite viridis
    out, _, _ = rebrand.remap_colormap(img(before), BOX, new_colours=NEW)
    ok, detail = verify_photo.check_colormap_remapped(img(before), out, BOX,
                                                      source_cmap='viridis',
                                                      new_colours=NEW)
    assert ok is False, detail
    assert 'BAND' in detail and 'two colour scales' in detail


def test_a_middle_of_range_band_is_rescued_by_spatial_continuity():
    """PASS, where the same offset in the middle of the range fails without rescue: a band
    flanked by confidently-matched content on both sides is closed by spatial continuity.
    `calibrate=False` reproduces the old behaviour, to confirm the rescue is what changes
    the outcome rather than the check's own thresholds."""
    before = gradient()
    before[80:120] = np.clip(before[80:120] + (30, 0, 30), 0, 255)   # not quite viridis

    out_old, frac_old, _ = rebrand.remap_colormap(img(before), BOX, new_colours=NEW,
                                                  calibrate=False)
    assert frac_old < 0.85, f'expected the band to survive with calibrate=False: {frac_old}'

    out, frac, _ = rebrand.remap_colormap(img(before), BOX, new_colours=NEW)
    assert frac == 1.0, f'expected the band to be rescued with calibrate=True: {frac}'
    ok, detail = verify_photo.check_colormap_remapped(img(before), out, BOX,
                                                      source_cmap='viridis',
                                                      new_colours=NEW)
    assert ok is True, detail


def test_naming_the_wrong_colormap_fails_without_blaming_a_band():
    """A wrong source colormap must fail without describing a band, which would send the
    caller hunting for a stripe that is not there."""
    before = gradient('viridis')
    out, _, _ = rebrand.remap_colormap(img(before), BOX, source_cmap='plasma',
                                       new_colours=NEW)
    ok, detail = verify_photo.check_colormap_remapped(img(before), out, BOX,
                                                      source_cmap='plasma',
                                                      new_colours=NEW)
    assert ok is False, detail
    assert 'plasma' in detail and 'BAND' not in detail


def test_the_value_each_colour_encodes_survives_the_remap():
    """A pixel's position in the source colormap must equal its position in the new
    gradient, within the new gradient's own colour resolution; a real encoding error
    (an inverted or shuffled ramp) would move positions by an order of magnitude more.
    """
    before = gradient()
    out, _, _ = rebrand.remap_colormap(img(before), BOX, new_colours=NEW)
    after = np.asarray(out).astype(float)
    src, new = rebrand._cmap_lut('viridis'), rebrand.build_gradient(NEW)
    i_b, _ = rebrand._nearest_in_lut(before.reshape(-1, 3), src, key='viridis')
    i_a, _ = rebrand._nearest_in_lut(after.reshape(-1, 3), new)
    drift = np.abs(i_b / (len(src) - 1) - i_a / (len(new) - 1))
    assert drift.max() <= 0.02, f'value encoding moved by up to {drift.max():.3f}'
    assert np.median(drift) <= 0.01, f'median drift {np.median(drift):.4f}'


@pytest.mark.parametrize('cmap', ['coolwarm', 'RdBu_r'])
def test_a_diverging_colormap_is_still_refused(cmap):
    """A diverging colormap encodes sign and must be refused rather than remapped onto a
    two-stop gradient, which would make positive and negative indistinguishable."""
    with pytest.raises(ValueError) as e:
        rebrand.remap_colormap(img(gradient()), BOX, source_cmap=cmap, new_colours=NEW)
    assert 'DIVERGING' in str(e.value)


def figure_with_colorbar():
    """A plot area and a separate colorbar, the way a typical heatmap figure is drawn."""
    a = np.full((200, 300, 3), 255.0)
    a[:, 0:240] = gradient(w=240, h=200)
    a[:, 265:290] = gradient(w=25, h=200)
    return a


PLOT = (0, 0, 240, 200)
BAR = (265, 0, 290, 200)


def test_a_colorbar_left_behind_is_reported():
    """The check measures only inside the box, so remapping the cells and forgetting the
    key must still be caught. PASS: the note names the survivors and where they are.
    FAIL: the same call on a figure with no colorbar must stay quiet."""
    before = img(figure_with_colorbar())
    out, _, _ = rebrand.remap_colormap(before, PLOT, new_colours=NEW)
    ok, detail = verify_photo.check_colormap_remapped(before, out, PLOT,
                                                      source_cmap='viridis', new_colours=NEW)
    assert ok is True, detail                       # the work inside the box was right
    assert 'OUTSIDE the box' in detail and 'colorbar' in detail, detail

    # The control must have no colormap outside the box at all.
    bare = np.full((200, 300, 3), 255.0)
    bare[:, 0:240] = gradient(w=240, h=200)
    plain = img(bare)
    out2, _, _ = rebrand.remap_colormap(plain, PLOT, new_colours=NEW)
    _, detail2 = verify_photo.check_colormap_remapped(plain, out2, PLOT,
                                                      source_cmap='viridis', new_colours=NEW)
    assert 'OUTSIDE the box' not in detail2, detail2


def test_finishing_the_job_is_not_reported_as_leaving_it_undone():
    """The second call of a two-step re-theme must come back clean: an anti-aliased
    gridline fringe must not be mistaken for a surviving colorbar, which carries a range
    of colours rather than one."""
    before = img(with_white_gridline(figure_with_colorbar()))
    step1, _, _ = rebrand.remap_colormap(before, PLOT, new_colours=NEW)
    step2, _, _ = rebrand.remap_colormap(step1, BAR, new_colours=NEW)
    ok, detail = verify_photo.check_colormap_remapped(step1, step2, BAR,
                                                      source_cmap='viridis', new_colours=NEW)
    assert 'OUTSIDE the box' not in detail, detail


def figure_with_two_overlays():
    """A ramp crossed by anti-aliased grey gridlines and a magenta annotation line, where
    the two cross each other."""
    a = np.repeat(_cmap()[np.linspace(0, 511, 300).astype(int)][:, None, :], 420, axis=1)
    for x in (100, 200, 300):
        a[:, x - 1] = 0.5 * a[:, x - 1] + 0.5 * np.array([128, 128, 128])
        a[:, x:x + 2] = 128
        a[:, x + 2] = 0.5 * a[:, x + 2] + 0.5 * np.array([128, 128, 128])
    for y in (80, 180):
        a[y - 1] = 0.5 * a[y - 1] + 0.5 * np.array([255, 0, 255])
        a[y:y + 2] = (255, 0, 255)
        a[y + 2] = 0.5 * a[y + 2] + 0.5 * np.array([255, 0, 255])
    return a


def _cmap():
    return rebrand._cmap_lut('viridis')


def test_the_edge_pass_does_not_assume_the_overlay_is_white_or_black():
    """PASS: the soft edge of a grey and of a magenta overlay both leave the old palette,
    since the overlay's colour is measured rather than assumed. FAIL: each line's core,
    carrying no colormap value, must come through untouched."""
    before = figure_with_two_overlays()
    out, _, _ = rebrand.remap_colormap(img(before), new_colours=NEW)
    after = np.asarray(out).astype(float)

    for label, rows, cols in (('grey', slice(None), [99, 102]),
                              ('magenta', [79, 82], slice(None))):
        moved = np.abs(before[rows, cols] - after[rows, cols]).max(axis=-1).mean()
        assert moved > 5, f'the {label} edge kept the old palette (moved {moved:.1f})'

    assert np.array_equal(after[150, 100], [128, 128, 128]), 'the grey line was recoloured'
    assert np.array_equal(after[80, 200], [255, 0, 255]), 'the magenta line was recoloured'


def test_no_box_means_the_whole_figure_including_the_colorbar():
    """The operation selects by colour, so it defaults to the whole figure with no box,
    covering the colorbar along with the plot in one call. PASS: with no box the colorbar
    is remapped too and nothing is reported outside. FAIL: with a box around the cells
    only, the colorbar stays on the old scale."""
    before = img(figure_with_colorbar())

    whole, _, _ = rebrand.remap_colormap(before, new_colours=NEW)
    ok, detail = verify_photo.check_colormap_remapped(before, whole, None,
                                                      source_cmap='viridis', new_colours=NEW)
    assert ok is True, detail
    assert 'OUTSIDE the box' not in detail, detail
    bar = np.asarray(whole).astype(float)[:, 265:290]
    _, d_new = rebrand._nearest_in_lut(bar.reshape(-1, 3), rebrand.build_gradient(NEW))
    assert d_new.mean() < 15, 'the colorbar was left on the old scale'

    cells_only, _, _ = rebrand.remap_colormap(before, PLOT, new_colours=NEW)
    _, detail2 = verify_photo.check_colormap_remapped(before, cells_only, PLOT,
                                                      source_cmap='viridis', new_colours=NEW)
    assert 'OUTSIDE the box' in detail2, detail2


def with_contour_lines(a, colour=(255, 255, 255)):
    """A ramp crossed by many thin anti-aliased lines, as a filled-contour plot is."""
    a = a.copy()
    for y in range(12, a.shape[0] - 12, 17):
        a[y - 1] = 0.5 * a[y - 1] + 0.5 * np.array(colour)
        a[y] = colour
        a[y + 1] = 0.5 * a[y + 1] + 0.5 * np.array(colour)
    return a


def test_contour_lines_are_not_mistaken_for_a_surviving_band():
    """PASS: a clean remap of a contour-style figure verifies, since the line-work's own
    fringe is not mistaken for a surviving band. FAIL: a real band left behind still fails,
    so this is not passing by ignoring everything."""
    clean = img(with_contour_lines(gradient()))
    out, _, _ = rebrand.remap_colormap(clean, new_colours=NEW)
    ok, detail = verify_photo.check_colormap_remapped(clean, out, None, new_colours=NEW)
    assert ok is True, detail

    banded = with_contour_lines(gradient())
    # At the range's edge, not the middle -- see the note on the same choice above.
    banded[0:40] = np.clip(banded[0:40] + (30, 0, 30), 0, 255)     # not quite viridis
    before = img(banded)
    out2, _, _ = rebrand.remap_colormap(before, new_colours=NEW)
    ok2, detail2 = verify_photo.check_colormap_remapped(before, out2, None, new_colours=NEW)
    assert ok2 is False and 'BAND' in detail2, detail2


def test_the_outside_note_needs_one_region_not_scattered_coincidence():
    """PASS: scattered coincidental matches on an unrelated scale are not reported. FAIL: a
    real colorbar, which is one contiguous region, still is."""
    scattered = np.full((200, 300, 3), 255.0)
    scattered[:, 0:240] = gradient(w=240, h=200)
    rng = np.random.default_rng(0)
    lut = rebrand._cmap_lut('viridis')
    for _ in range(400):                       # confetti of colormap-coloured specks
        y, x = rng.integers(0, 190), rng.integers(245, 292)
        scattered[y:y + 3, x:x + 3] = lut[rng.integers(0, len(lut))]
    before = img(scattered)
    out, _, _ = rebrand.remap_colormap(before, PLOT, new_colours=NEW)
    _, detail = verify_photo.check_colormap_remapped(before, out, PLOT, new_colours=NEW)
    assert 'OUTSIDE the box' not in detail, detail

    bar = img(figure_with_colorbar())
    out2, _, _ = rebrand.remap_colormap(bar, PLOT, new_colours=NEW)
    _, detail2 = verify_photo.check_colormap_remapped(bar, out2, PLOT, new_colours=NEW)
    assert 'OUTSIDE the box' in detail2 and 'in one region' in detail2, detail2


# Below: a figure that is mostly not colormap -- a scatter plot where the markers and
# colorbar are a small fraction of a white page.

def sparse_figure(alpha=0.8, w=600, h=400, bar=(560, 40, 585, 360)):
    """A white page with ONE small colormap region on it, blended at `alpha`."""
    a = np.full((h, w, 3), 255.0)
    x0, y0, x1, y1 = bar
    ramp = gradient(w=x1 - x0, h=y1 - y0)
    a[y0:y1, x0:x1] = ramp * alpha + 255.0 * (1 - alpha)
    return a


def test_the_alpha_search_is_not_decided_by_the_background():
    """White background pixels un-blend to white at any alpha, so the alpha search must
    not be decided by them rather than by the colormap region itself."""
    a = sparse_figure(alpha=0.8)
    found, _err = rebrand.detect_colormap_alpha(img(a), (0, 0, a.shape[1], a.shape[0]))
    assert abs(found - 0.8) < 0.05, found


def test_an_alpha_is_not_invented_for_a_figure_with_no_colormap_in_it():
    """A page with no colormap content must come back opaque, not with an alpha fitted to
    whatever happened to land nearest the LUT."""
    a = np.full((400, 600, 3), 255.0)
    a[100:300, 100:400] = (40, 40, 40)                 # a dark grey panel, not a colormap
    found, _err = rebrand.detect_colormap_alpha(img(a), (0, 0, 600, 400))
    assert found == pytest.approx(1.0), found


def test_a_correct_remap_of_a_blended_figure_is_not_failed_for_drifting():
    """The output is re-blended at the same alpha on purpose, so its pixels are not ON the
    new gradient; measuring them against the pure gradient must not fail correct work."""
    a = sparse_figure(alpha=0.8)
    before = img(a)
    after, frac, used = rebrand.remap_colormap(before, new_colours=('#0000FF', '#FF0000'))
    assert used < 0.9, f'alpha not detected: {used}'
    ok, detail = verify_photo.check_colormap_remapped(
        before, after, new_colours=('#0000FF', '#FF0000'))
    assert 'moved by 0.00' in detail, detail
    assert ok is not False, detail


def test_a_remap_that_really_does_move_the_values_still_fails():
    """The direction that must still fail: same figure, same alpha, but the output's scale
    is reversed, so every colour means the opposite of what it meant."""
    a = sparse_figure(alpha=0.8)
    before = img(a)
    after, _frac, _used = rebrand.remap_colormap(before, new_colours=('#0000FF', '#FF0000'))
    flipped = np.asarray(after.convert('RGB')).copy()
    x0, y0, x1, y1 = 560, 40, 585, 360
    flipped[y0:y1, x0:x1] = flipped[y0:y1, x0:x1][::-1]
    ok, detail = verify_photo.check_colormap_remapped(
        before, img(flipped), new_colours=('#0000FF', '#FF0000'))
    assert ok is False, detail
    assert 'did NOT preserve the value encoding' in detail, detail


def test_an_opaque_colorbar_beside_a_blended_plot_still_counts_as_the_old_scale():
    """An opaque colorbar beside a blended plot must still be recognised as the old scale:
    leftovers are matched against either rendering, even though drift is measured only in
    blended space."""
    a = np.full((260, 400, 3), 255.0)
    a[20:240, 20:300] = gradient(w=280, h=220) * 0.72 + 255.0 * 0.28    # the plot, blended
    a[20:240, 330:360] = gradient(w=30, h=220)                          # the key, opaque
    before = img(a)
    after = np.asarray(before.convert('RGB')).astype(float).copy()
    remapped, _frac, used = rebrand.remap_colormap(img(a[20:240, 20:300]), new_colours=NEW)
    after[20:240, 20:300] = np.asarray(remapped.convert('RGB'), dtype=float)
    assert used < 0.9, f'alpha not detected on the blended plot: {used}'

    ok, detail = verify_photo.check_colormap_remapped(before, img(after), new_colours=NEW)
    assert ok is False, detail
    assert 'left on the old scale' in detail, detail


def test_the_value_range_search_finds_the_same_distances_as_a_distance_transform():
    """`_spatial_rescue`'s k-d tree search over value ranges must find the same nearest-
    confident-pixel distances as a per-value distance transform."""
    from scipy.ndimage import distance_transform_edt
    from figsurgeon.rebrand import _nearest_valued
    rng = np.random.default_rng(0)
    H, W = 90, 130
    pos = rng.integers(0, 40, (H, W)).astype(float)
    conf = rng.random((H, W)) < 0.3
    cy, cx = np.nonzero(~conf)
    confy, confx = np.nonzero(conf)
    for side, keep in (('below', np.less), ('above', np.greater)):
        d, i = _nearest_valued(confy, confx, pos[confy, confx], cy, cx, pos[cy, cx], side, 25)
        for v in np.unique(pos[cy, cx]):
            s = pos[cy, cx] == v
            allowed = conf & keep(pos, v)
            if not allowed.any():
                assert np.isinf(d[s]).all()
                continue
            ref = distance_transform_edt(~allowed)[cy[s], cx[s]]
            within = ref <= 25
            assert np.array_equal(d[s] <= 25, within), (side, v)
            assert np.allclose(d[s][within], ref[within]), (side, v)
            assert keep(pos[confy, confx][i[s][within]], v).all(), (side, v)
