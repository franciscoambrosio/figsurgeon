"""Re-theme a matplotlib chart export to match a brand style guide -- font, label size,
and colour palette -- from the rendered PNG alone.

This is a different problem from photo.py/compose.py: those separate or recolour discrete
series. Here the hard case is a continuous colormap (a heatmap or hexbin background, plus
its colorbar) that encodes real data values in colour. Recolouring it isn't picking a mask
and flattening it -- every pixel's colour has to be reinterpreted as a position along the
original colormap, then re-rendered through a new gradient at that same position, or the
chart stops meaning what it meant.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .composite import _arr, _clamp
# matplotlib and pytesseract are deliberately not imported at module level -- same reasoning
# as advanced.py's cv2/rembg: importing figsurgeon for chart or photo work alone shouldn't
# require either of these. Each function imports what it needs.

LIBERATION_SANS = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
LIBERATION_SANS_BOLD = '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'


def _cmap_lut(name, n=512):
    import matplotlib
    cmap = matplotlib.colormaps[name]
    t = np.linspace(0, 1, n)
    return (np.array([cmap(x)[:3] for x in t]) * 255)


def build_gradient(colours, n=512):
    """A linear-interpolated gradient LUT through `colours` (hex strings or RGB tuples)."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    rgbs = [to_rgb(c) for c in colours]
    cmap = LinearSegmentedColormap.from_list('brand', rgbs, N=n)
    t = np.linspace(0, 1, n)
    return (np.array([cmap(x)[:3] for x in t]) * 255)


_LUT_TREES = {}


def _nearest_in_lut(values, lut, key=None):
    """Nearest LUT entry for each RGB value: (index, distance).

    A KD-tree over 512 LUT entries, not an all-pairs distance array: for a 633x249 plot
    area that array is 157k x 512 x 3 floats (about 1.9 GB), and `detect_colormap_alpha`
    rebuilds it once per alpha step, 31 times. The KD-tree brings a call from around two
    minutes to under a second, needed for an interactive editing loop.

    Distances match the brute-force result exactly (max difference 0.0). Where the chosen
    index differs it is always a tie -- 256 of viridis's 512 LUT rows are exact duplicates
    once quantised to 8-bit, so both indices name the same source colour, and the largest
    resulting output difference is 0.24/255, i.e. the same byte.
    """
    from scipy.spatial import cKDTree
    if key is not None:
        tree = _LUT_TREES.get(key)
        if tree is None:
            tree = _LUT_TREES[key] = cKDTree(lut)
    else:
        tree = cKDTree(lut)
    dist, idx = tree.query(values, workers=-1)
    return idx, dist


def _unblend(pixels, bg, alpha):
    """Invert `alpha*colour + (1-alpha)*bg`: the on-screen colour a colormap pixel would be
    if it were fully opaque. Same formula in `detect_colormap_alpha`'s alpha search and
    `remap_colormap`'s own match, written once so the two cannot drift apart."""
    return (np.asarray(pixels, dtype=float) - (1 - alpha) * np.asarray(bg, dtype=float)) / alpha


def _self_calibrated_lut(unblended, canonical_lut, nearest, best_dist, tolerance,
                         min_confident=200, max_gap=48):
    """An LUT traced from this image's own pixels, for rescuing a band matplotlib's own
    reference LUT cannot see.

    A published figure's colormap is very often not exactly matplotlib's: a different
    renderer, a colour-managed export, a JPEG re-save -- something applies a small, smooth
    shift to every colour. `remap_colormap`'s ordinary match handles that up to
    `tolerance`, but a large or uneven shift pushes part of the gradient's own range
    outside it. Widening the tolerance would recolour content that was never in the
    colormap at all (see `remap_colormap`'s docstring), so instead this asks the image what
    its colours actually are, at the positions already known with confidence.

    A pixel matching the canonical LUT within half the caller's tolerance is confident
    evidence of both its position and its true on-screen colour there. Pooling those by
    position (one bin per canonical LUT step) gives samples of the real curve wherever there
    is evidence for it; a position with no confident sample of its own is filled by linear
    interpolation only when its two flanking confident bins are within `max_gap` of each
    other -- the colormap is continuous, so two confident samples a short way apart describe
    one smooth stretch between them, but a real chart very often only uses part of its
    colormap's range at all, so nothing here requires every position to have nearby
    evidence. Positions outside the confident span, or inside a wider gap, keep the
    canonical colour untouched and are left for the ordinary pass to decide.

    Calibration requires the confident coverage to reach both flanks of a gap of up to
    `max_gap` bins, not a fixed fraction of the whole 512-step LUT -- a real chart often
    uses only part of a colormap's range (`evals/chart_remap.py`'s `photon_jet` covers 23%
    of jet's range), and a global coverage fraction would refuse to calibrate at all in
    that case.

    This pass alone does not rescue every band: where the renderer's colormap is genuinely
    flatter over some stretch than matplotlib's rather than shifted, the true colour there
    is further than `tolerance` from every one of the 512 canonical entries and no traced
    or interpolated colour places it (`photon_jet`'s band, closed instead by
    `_spatial_rescue`). This pass is for a genuinely smooth global shift -- on a synthetic
    gamma-shifted jet gradient built with that property, it raises the match from 83.5% to
    96.5%.

    Returns None when there is too little to calibrate from at all (too few confident
    pixels to trust, or fewer than two distinct confident positions).
    """
    n = len(canonical_lut)
    confident = best_dist < (tolerance * 0.5)
    if confident.sum() < min_confident:
        return None
    idx = nearest[confident]
    sums = np.zeros((n, 3))
    counts = np.zeros(n)
    np.add.at(sums, idx, unblended[confident])
    np.add.at(counts, idx, 1)
    covered = counts > 0
    covered_bins = np.where(covered)[0]
    if len(covered_bins) < 2:
        return None
    observed = canonical_lut.copy()
    observed[covered] = sums[covered] / counts[covered, None]
    gap_starts = np.where(np.diff(covered_bins) > 1)[0]
    for g in gap_starts:
        lo, hi = int(covered_bins[g]), int(covered_bins[g + 1])
        if hi - lo > max_gap:
            continue
        span = np.arange(lo + 1, hi)
        for c in range(3):
            observed[span, c] = np.interp(span, [lo, hi], [observed[lo, c], observed[hi, c]])
    return observed


# Loose enough to catch a genuinely-part-of-the-colormap pixel a renderer difference or a
# flattened stretch pushed past `match_tolerance`, tight enough to still exclude ordinary
# figure content -- the same "colourful, not obviously something else" bound this package's
# own eval (`evals/chart_remap.py`'s `old_scale_left`) already uses to tell old-scale
# survivors from axis lines, text and background.
_SPATIAL_BAND_TOLERANCE = 60


def _spatial_rescue(nearest, best_dist, confident, candidate, shape, max_reach=250):
    """Position a candidate pixel -- visually part of the colormap but not a confident
    colour match, even against `_self_calibrated_lut` -- from where it sits between the
    nearest confidently-matched pixel below it in position and the nearest one above.

    Exists for a failure neither the canonical LUT nor a calibrated one can see: a real
    published figure's colormap can be genuinely flatter over some stretch than
    matplotlib's own definition -- several distinct data values render as almost the same
    colour, because that renderer's version of the colormap has a wider plateau there, not
    because of noise. On the case this exists for (`evals/chart_remap.py`'s `photon_jet`),
    the true pixel colour in the band is further than 15 from every one of the 512
    canonical LUT entries, so no recalibration of colour matching can place it -- the
    position information is not in the colour there at all. It is still in the image: a
    chart is usually a spatially continuous field, so a pixel sandwiched between a
    lower-positioned and a higher-positioned confidently-matched neighbour is almost always
    continuing the trend between them, whatever its own colour says.

    A candidate's flanks are the nearest confident pixel valued strictly below its own value
    and the nearest valued strictly above it. Inside a genuine band -- a contiguous run of
    positions with no confident pixel anywhere in the image -- that reaches past the band to
    its real boundaries, and a sparse, incidentally-correct confident pixel inside the band
    cannot be a flank, because it is not in the band's value range. On a scanline through
    `photon_jet`'s band, the flat run of one repeated canonical index becomes a smooth,
    monotonic progression through it.

    Computed with k-d trees over value ranges (`_nearest_valued`) rather than a distance
    transform per gap and per residual value, which does not scale: on six real audit
    charts this is 10-12x faster (jan_rain 68 s -> 5.7 s) for the same Euclidean distances
    and the same pixels within reach. Where two confident pixels are exactly equidistant
    the two methods can pick different ones (up to 3.4% of pixels on fertile_crescent,
    isolated single pixels), but `check_colormap_remapped` gives the same verdict and the
    same value drift on all six.

    A candidate whose own value already carries a confident pixel somewhere else in the
    image (two regions can legitimately share a data value, one rendered faithfully and one
    not) sits inside no gap at all, and the same rule applies: that pixel has neither a
    lower nor a higher value, so it is not a flank.

    `max_reach`: a candidate further than this from a confident pixel of either polarity is
    left alone -- a large empty reach is not continuity, it is an absence of it.

    Returns (positions, rescued): `positions` is `nearest.astype(float)` with rescued
    entries replaced; `rescued` is which entries changed.
    """
    H, W = shape
    pos = nearest.reshape(H, W).astype(float)
    out_pos = pos.copy()
    rescued2d = np.zeros((H, W), bool)

    # Both groupings above come down to one rule per candidate of value v: blend between the
    # nearest confident pixel valued below v and the nearest valued above it. (Inside a gap
    # no confident pixel has a value in (lo, hi), so `pos <= lo` is `pos < v`.) Answered
    # with k-d trees over value ranges, giving the same Euclidean distances a per-gap
    # distance transform would, much faster.
    cy, cx = np.nonzero(candidate.reshape(H, W))
    confy, confx = np.nonzero(confident.reshape(H, W))
    if cy.size and confy.size:
        v = pos[cy, cx]
        cval = pos[confy, confx]
        dl, il = _nearest_valued(confy, confx, cval, cy, cx, v, 'below', max_reach)
        dh, ih = _nearest_valued(confy, confx, cval, cy, cx, v, 'above', max_reach)
        ok = (dl <= max_reach) & (dh <= max_reach)
        w = dl[ok] / np.maximum(dl[ok] + dh[ok], 1e-6)
        val_low, val_high = cval[il[ok]], cval[ih[ok]]
        out_pos[cy[ok], cx[ok]] = val_low + w * (val_high - val_low)
        rescued2d[cy[ok], cx[ok]] = True

    return out_pos.reshape(-1), rescued2d.reshape(-1)


def _nearest_valued(py, px, pval, qy, qx, qval, side, max_reach):
    """For each query pixel, the nearest point whose value is strictly below (`side='below'`)
    or above its own: (distance, index into the points), distance inf where none lies
    within `max_reach`.

    The allowed points for a query are a prefix of the points sorted by value, so the
    prefix is split into power-of-two blocks of distinct values -- the blocks of the binary
    representation of its length -- and each block gets one k-d tree, shared by every query
    that needs it. Around a dozen trees are queried per pixel instead of one transform
    over the whole frame per distinct value.
    """
    from scipy.spatial import cKDTree
    if side == 'above':                       # "above v" is "below -v"
        pval, qval = -pval, -qval
    bins = np.unique(pval)
    rank = np.searchsorted(bins, pval)
    order = np.argsort(rank, kind='stable')
    sorted_rank = rank[order]
    pts = np.column_stack([py, px])[order].astype(float)
    k = np.searchsorted(bins, qval, side='left')      # how many distinct values lie below
    q = np.column_stack([qy, qx]).astype(float)
    best_d = np.full(len(q), np.inf)
    best_i = np.zeros(len(q), int)
    for j in range(int(k.max()).bit_length() if k.size and k.max() > 0 else 0):
        uses = ((k >> j) & 1).astype(bool)
        starts = (k >> (j + 1)) << (j + 1)
        for start in np.unique(starts[uses]):
            a, b = np.searchsorted(sorted_rank, [start, start + (1 << j)])
            if a == b:
                continue
            sel = np.nonzero(uses & (starts == start))[0]
            d, i = cKDTree(pts[a:b]).query(q[sel], distance_upper_bound=max_reach + 1e-9,
                                             workers=-1)
            better = d < best_d[sel]
            best_d[sel[better]] = d[better]
            best_i[sel[better]] = order[a + i[better]]
    return best_d, best_i


# How much better the winning alpha has to be than full opacity before it is believed, and
# how much of the region it must actually match. Without both, a figure carrying no
# colormap at all would get an alpha fitted to noise.
_ALPHA_EVIDENCE = 1.25
_ALPHA_MIN_SHARE = 0.002


def detect_colormap_alpha(img, box, source_cmap='viridis', max_samples=40000,
                          tolerance=15):
    """Grid-search the alpha a colormap region was blended with over white.

    Matters because matplotlib content is very often drawn with alpha < 1 (density plots,
    overlapping hexbins, a deliberate stylistic choice) -- matching pixels against the pure
    colormap LUT without accounting for this fails badly. On a real chart, mean nearest-LUT
    distance was 50-85 assuming full opacity; searching alpha found 0.76, bringing the
    residual to 3.6.

    Scored by how many pixels match, not by the mean residual. The mean is taken over the
    whole box, and on a scatter plot the colormap is a few per cent of it -- 100 markers and
    a colorbar against a white page. Every alpha leaves the background exactly where it was
    (un-blending white over white returns white), so those pixels contribute the same large
    constant to every step of the search, and the winner is decided by noise in the
    remainder. On `evals/_corpus/audit_remap/rho_oph_scatter.png`, a viridis scatter drawn
    at 0.80, the mean picks 1.00 and matches 0.1% of the figure -- leaving the colorbar and
    every mid-value point on the old scale -- while maximising the matched fraction picks
    0.82 and matches 2.4%, which is all of the colormap content there is.

    On the 16 figures of that corpus with a known colormap the two objectives agree exactly
    on 12; the four they differ on, the matched fraction is the same or better (rho_oph
    0.1 -> 2.4%, wiki_depth 3.2 -> 3.2%, hr_diagram 0.5 -> 0.6%, kangerlussuaq
    42.9 -> 43.2%). A tie keeps the largest alpha, because claiming a blend that is not
    there costs more than assuming opacity.

    `max_samples` subsamples the region: this estimates one scalar from hundreds of
    thousands of pixels, and a strided sample of tens of thousands moves the estimate by
    less than the 0.02 grid step.

    Returns (alpha, mean nearest-LUT distance at that alpha).
    """
    a = _arr(img)
    x0, y0, x1, y1 = box
    region = a[y0:y1, x0:x1].reshape(-1, 3)
    if len(region) > max_samples:
        region = region[::max(1, len(region) // max_samples)]
    lut = _cmap_lut(source_cmap)
    scored = []
    for alpha in np.arange(0.4, 1.001, 0.02):
        d = _nearest_in_lut(_unblend(region, (255, 255, 255), alpha), lut,
                            key=source_cmap)[1]
        scored.append((float(alpha), float((d < tolerance).mean()), float(d.mean())))

    opaque = max(scored, key=lambda r: r[0])
    best_share = max(r[1] for r in scored)
    # Ties go to the largest alpha: the grid is fine enough that several steps can match
    # the same pixels, and the least surprising of those is the one closest to opaque.
    alpha, share, err = max((r for r in scored if r[1] >= best_share - 1e-9),
                            key=lambda r: r[0])
    if share < _ALPHA_MIN_SHARE or share < opaque[1] * _ALPHA_EVIDENCE:
        return float(opaque[0]), float(opaque[2])
    return float(alpha), float(err)


def _remap_fringe(region, match, src_obs, new_obs, nearest, tolerance):
    """Recolour the anti-aliased edge where an overlay crosses the colormap.

    A gridline, a contour, a marker or a letter drawn over a colormap does not have a hard
    edge: its border pixels are blends of the overlay's colour and the colormap colour under
    it. Those blends are not in the colormap's LUT, so the main match skips them -- correctly,
    by its own rule -- and they keep the old palette. Against the new one they read as a
    coloured outline: on a published viridis contour plot, every white contour line came
    back with a yellow-green halo (30,535 px, 4.3% of the plot box), and on a viridis
    heatmap every white gridline came back green, yellow or purple (19,160 px, 7.7%). Both
    are invisible at figure scale and unmistakable at 1:1.

    An edge pixel is `p = C + t*(O - C)`: C the colormap colour as it appears on screen, O
    the overlay's colour, t how much of the pixel the overlay covers. The output is the same
    t against the new colour, `C' + t*(O - C')`, so the edge keeps its softness and the value
    C encoded is unchanged. Both ends are sampled from the image itself:

      * C comes from the nearest remapped pixel -- the colormap colour under a line is the
        one beside it. (Searching all 512 LUT entries for the best (C, t) instead cost 3.3 s
        on a 250 kpx box against 0.5 s for the whole rest of the operation, and left more
        leftovers, because a free search also fits edges that are not edges.)
      * O comes from the overlay itself: the nearest pixel that is unmatched and not on an
        edge -- the core of the line, letter or marker running through here. This must be
        sampled rather than assumed to be white or black: a figure with grey gridlines and
        a magenta annotation line needs both, and a per-component vote is also wrong, since
        overlays cross (a grid and an annotation line are one component, and one vote cannot
        describe both). The nearest core pixel is local, needs no assumption about colour,
        and handles a figure carrying several different overlays at once.

    Two guards, because the alternative is silently recolouring content that was never part
    of the chart: only pixels adjacent to a remapped one are considered, and `t <= 0.9` with
    a fit inside `tolerance`, so a pixel that is essentially pure overlay -- carrying no
    colormap value to preserve -- is left alone.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, distance_transform_edt
    cand = binary_dilation(match, iterations=1) & ~match
    if not cand.any() or not match.any():
        return None, 0

    # C: the colormap colour beside each edge pixel, and its replacement.
    _, (iy, ix) = distance_transform_edt(~match, return_indices=True)
    idx2 = nearest.reshape(match.shape)
    C = src_obs[idx2[iy[cand], ix[cand]]]
    Cn = new_obs[idx2[iy[cand], ix[cand]]]
    p = region[cand]

    # O: the overlay running through here, sampled from its own core -- the nearest pixel
    # that is unmatched and not itself an edge. A line thin enough to be all edge has no
    # core and no colormap value to preserve either; the `t <= 0.9` guard below is what
    # leaves those alone.
    core = ~match & ~cand
    if not core.any():
        return None, 0
    _, (cy, cx) = distance_transform_edt(~core, return_indices=True)
    O = region[cy[cand], cx[cand]]

    d = O - C
    denom = (d * d).sum(axis=1)
    denom[denom < 1e-6] = 1e-6
    t = np.clip(((p - C) * d).sum(axis=1) / denom, 0.0, 1.0)
    res = np.linalg.norm(p - (C + t[:, None] * d), axis=1)
    take = (res < tolerance) & (t <= 0.9)
    if not take.any():
        return None, 0
    out = Cn[take] + t[take][:, None] * (O[take] - Cn[take])
    where = np.zeros_like(cand)
    where[cand] = take
    return (where, out), int(take.sum())


def remap_colormap(img, box=None, source_cmap='viridis', new_colours=('#00407A', '#52BDEC'),
                   match_tolerance=15, alpha=None, background=(255, 255, 255),
                   allow_diverging=False, calibrate=True):
    """Remap every pixel that matches `source_cmap` to the equivalent position on a new
    brand gradient, preserving the value ordering the original colormap encoded.

    `box` is optional, and leaving it out is usually right. This operation selects by
    colour -- a pixel is remapped because it matches the source colormap, not because of
    where it sits -- so a box only ever restricts where that selection is allowed to apply.
    Defaulting it to the whole figure is what makes the operation generic across layouts:
    the colorbar, a second panel, an inset, a legend patch and a scale drawn down the side
    all carry the same colormap and all go together, without the caller having to know the
    geometry.

    Pass a box only when part of the figure must be protected -- a second heatmap in the
    same colormap that should keep it, or a photograph pasted into the page.

    Refuses diverging colormaps by default (`allow_diverging=False`). A diverging map
    (RdBu, coolwarm, bwr, seismic, PiYG...) encodes sign, not just magnitude: red is
    positive, blue is negative, and the pale midpoint is zero. Remapping it onto a
    sequential two-colour gradient technically succeeds -- on a real RdBu_r heatmap, 99% of
    pixels matched -- while destroying the one thing the chart was for: in the output you
    can no longer tell a positive cell from a negative one. That is worse than refusing,
    because the result looks clean and is silently misleading. Pass a diverging pair of
    `new_colours` (a dark colour, a pale midpoint, a second dark colour) and
    `allow_diverging=True` if you genuinely want to re-theme one.

    How a pixel is reinterpreted: un-blend it from `background` at `alpha` (auto-detected
    via `detect_colormap_alpha` if not given -- matplotlib content is very often drawn at
    alpha<1, and matching without correcting for this fails badly, see that function's
    docstring), then find its nearest match in the source colormap's LUT (512-step,
    matplotlib's own definition). That match's position (0..1) is the data value it
    represents. The new colour is the brand gradient sampled at that same position,
    re-blended at the same alpha so the chart's visual style (not just its hue) survives.

    A pixel far from the whole source LUT even after un-blending (further than
    `match_tolerance`) is left untouched -- it's not part of the colormap (axis lines,
    markers, text), and forcing a nearest-match on it would silently recolour content that
    was never in the colormap to begin with.

    `calibrate=True` (default) adds two rescue passes for a pixel the ordinary match
    leaves stranded outside `match_tolerance` -- not by widening the tolerance (see above:
    that recolours content that was never in the colormap), but by using evidence the
    ordinary pass never looks at.

    A published figure's colormap is very often not exactly matplotlib's reference one --
    a different renderer, a colour profile, a JPEG re-save -- and when that difference is a
    small, smooth shift applied everywhere, `_self_calibrated_lut` traces the image's own
    on-screen colour from pixels already confidently matched and re-matches against that
    instead. On a synthetic gamma-shifted jet gradient built to have exactly this property,
    match rises from 83.5% to 96.5%, with no pixels of unrelated content (a plain
    background, axis text) ever admitted.

    A band can also survive because the renderer's colormap is genuinely flatter over some
    stretch than matplotlib's -- several distinct values render as almost the same colour,
    which is not a shift any recalibration of colour can see through, because the position
    information is not in the colour there at all. `_spatial_rescue` closes that one, from
    the image's own layout rather than its colour: on the real published figure this whole
    mechanism exists for (`evals/chart_remap.py`'s `photon_jet`, a jet field whose jet is
    not matplotlib's), the survivor band drops from 3.15% to 0.00% of the box's colormap
    content with no false grab of the figure's axis text or background. `_self_calibrated_lut`
    alone leaves that exact band untouched (see its own docstring for why).

    Both passes are additive only -- a confident canonical match is never revised -- and
    each declines outright rather than acting on too little evidence (see their own
    docstrings).

    Returns (result_image, match_fraction, alpha_used).
    """
    DIVERGING = {'rdbu', 'rdbu_r', 'coolwarm', 'coolwarm_r', 'bwr', 'bwr_r', 'seismic',
                 'seismic_r', 'piyg', 'piyg_r', 'prgn', 'prgn_r', 'brbg', 'brbg_r',
                 'puor', 'puor_r', 'rdgy', 'rdgy_r', 'rdylbu', 'rdylbu_r', 'spectral',
                 'spectral_r', 'rdylgn', 'rdylgn_r'}
    if source_cmap.lower() in DIVERGING and not allow_diverging:
        raise ValueError(
            f'{source_cmap!r} is a DIVERGING colormap: it encodes sign (one colour for '
            f'positive, another for negative, pale at zero). Remapping it onto the '
            f'sequential gradient {tuple(new_colours)} would discard that distinction and '
            f'produce a chart where positive and negative values are indistinguishable. '
            f'Either keep the original colormap, or pass a 3-stop diverging replacement '
            f'(dark, pale, dark) with allow_diverging=True.')

    a = _arr(img)
    H, W = a.shape[:2]
    if box is None:
        box = (0, 0, W, H)
    x0, y0, x1, y1 = _clamp(box, W, H)
    if x1 <= x0 or y1 <= y0:
        # An empty region gives a mean over nothing, which would report "remapped nan% of
        # pixels" -- a success-shaped report of an operation that could not run.
        raise ValueError(
            f'box {tuple(box)} selects no pixels of the {W}x{H} image; '
            f'pass a box inside its bounds')
    region = a[y0:y1, x0:x1]
    if alpha is None:
        alpha, _ = detect_colormap_alpha(img, box, source_cmap,
                                         tolerance=match_tolerance)

    src_lut = _cmap_lut(source_cmap)
    new_lut = build_gradient(new_colours)
    bg = np.array(background, float)

    flat = region.reshape(-1, 3)
    unblended = _unblend(flat, bg, alpha)
    nearest, best_dist = _nearest_in_lut(unblended, src_lut, key=source_cmap)

    if calibrate:
        calibrated_lut = _self_calibrated_lut(unblended, src_lut, nearest, best_dist,
                                              match_tolerance)
        if calibrated_lut is not None:
            nearest2, best_dist2 = _nearest_in_lut(unblended, calibrated_lut, key=None)
            # Additive only: a pixel the canonical pass already matched keeps that match
            # (and the position it named) untouched. This only ever rescues a pixel the
            # canonical pass missed, and only when the image's own traced curve confirms it.
            rescue = (best_dist >= match_tolerance) & (best_dist2 < match_tolerance)
            nearest = np.where(rescue, nearest2, nearest)
            best_dist = np.where(rescue, best_dist2, best_dist)

        # A pixel the calibrated LUT still cannot place is not necessarily unrelated
        # content -- its true colour may sit further than `match_tolerance` from every one
        # of the 512 canonical entries, which no recalibration of what counts as a colour
        # match can fix (see `_spatial_rescue`'s docstring). Try the image's own layout
        # instead, restricted to pixels loose colour distance still says are plausibly part
        # of the colormap.
        confident = best_dist < match_tolerance
        candidate = (~confident) & (best_dist < _SPATIAL_BAND_TOLERANCE)
        if candidate.any() and confident.any():
            positions, rescued = _spatial_rescue(nearest, best_dist, confident, candidate,
                                                 region.shape[:2])
            new_idx = np.clip(positions.round(), 0, len(src_lut) - 1).astype(int)
            nearest = np.where(rescued, new_idx, nearest)
            best_dist = np.where(rescued, 0.0, best_dist)   # positioned by layout, not
                                                             # colour -- treat as matched

    match = best_dist < match_tolerance
    out_flat = flat.copy()
    src_obs = alpha * src_lut + (1 - alpha) * bg        # the LUT as it appears on screen
    new_obs = alpha * new_lut + (1 - alpha) * bg
    reblended = new_obs[nearest]
    out_flat[match] = reblended[match]
    out_region = out_flat.reshape(region.shape)

    # The anti-aliased edge where a gridline, contour or letter crosses the colormap keeps
    # the old palette otherwise, and reads as a coloured outline against the new one.
    fringe, n_fringe = _remap_fringe(region, match.reshape(region.shape[:2]),
                                     src_obs, new_obs, nearest, match_tolerance)
    if fringe is not None:
        where, values = fringe
        out_region[where] = values

    out = a.copy()
    out[y0:y1, x0:x1] = out_region
    result = Image.fromarray(np.clip(out, 0, 255).astype('uint8'))
    return result, float(match.mean() + n_fringe / match.size), alpha


def find_text_regions(img, min_confidence=40, group_lines=True, exclude_boxes=()):
    """OCR every text element: returns [(text, (x0,y0,x1,y1), confidence), ...].

    Uses tesseract via pytesseract. Confidence is tesseract's own 0-100 score -- filter on
    it before trusting a detected string, since OCR on small matplotlib tick labels (often
    8-10px tall) is genuinely error-prone, not a solved problem to paper over.

    `exclude_boxes`: words overlapping any of these are dropped before line-grouping, not
    after -- filtering afterwards is too late. A rotated axis title sits only a few px from
    the adjacent tick-number column, so even once that title is correctly redrawn, a later
    full-image scan still finds it as text and the grouping below merges its fragments into
    nearby tick numbers ('6.25' became '9 6.25', confidence dragged from 96 down to 59,
    silently falling under the threshold and leaving that one tick label unswapped).
    Excluding at the word level, before any merging, prevents the merge.

    `group_lines=True` (default) merges words tesseract assigns to the same line into one
    entry, spanning from the first word's left edge to the last word's right edge, with the
    combined text joined by spaces. This matters for redrawing in a different font: redrawing
    each word independently at its own original x-position leaves visibly wide, unnatural
    gaps between words, since Liberation Sans renders narrower than the original DejaVu and
    words anchored to their old positions drift apart with nothing to fill the space a wider
    font used to occupy. A low-confidence word inside an otherwise-good line still pulls its
    text in (better than a gap), but does not extend the line's confidence-based inclusion
    on its own.
    """
    import pytesseract
    try:
        data = pytesseract.image_to_data(img.convert('RGB'),
                                         output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as e:
        # This is missing the tesseract binary, not the pytesseract wrapper: `pip install
        # pytesseract` succeeds and this still fails. pytesseract's own message ends "See
        # README file for more information" -- its readme, which the caller does not have
        # open -- and names neither the package nor the command to install it.
        # Re-raised as a plain OSError, which is what TesseractNotFoundError already is
        # (it subclasses EnvironmentError); pytesseract's own class takes no message
        # argument at all, so the message has to travel on a different exception.
        raise OSError(
            'the tesseract OCR binary is not installed or not on PATH. pip cannot install '
            'it: apt-get install tesseract-ocr (Debian/Ubuntu), brew install tesseract '
            '(macOS). Everything in figsurgeon except find_text_regions / replace_font '
            'works without it.') from e
    words = []
    for i, txt in enumerate(data['text']):
        txt = txt.strip()
        conf = float(data['conf'][i])
        if not txt or conf < min_confidence:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        box = (x, y, x + w, y + h)
        if any(box[0] < ex1 and box[2] > ex0 and box[1] < ey1 and box[3] > ey0
               for ex0, ey0, ex1, ey1 in exclude_boxes):
            continue
        words.append({'text': txt, 'box': box, 'conf': conf,
                     'line': (data['block_num'][i], data['par_num'][i], data['line_num'][i])})
    if not group_lines:
        return [(w['text'], w['box'], w['conf']) for w in words]

    lines = {}
    for w in words:
        lines.setdefault(w['line'], []).append(w)
    out = []
    for line_words in lines.values():
        line_words.sort(key=lambda w: w['box'][0])
        # Split a tesseract "line" further wherever the horizontal gap between consecutive
        # words is large relative to their height: tesseract assigns the same line_num to
        # widely-separated content that happens to sit at the same y (all six x-axis tick
        # numbers spread across the full axis width; both columns of a two-column legend),
        # which without this split gets joined into one nonsensical string
        # ("10 12 14 16 18 20", "Feasible region A Sample (in band, n=54)") drawn from a
        # single anchor point -- destroying the layout, not just the wording.
        clusters, cur = [], [line_words[0]]
        for w in line_words[1:]:
            gap = w['box'][0] - cur[-1]['box'][2]
            h = cur[-1]['box'][3] - cur[-1]['box'][1]
            if gap > max(15, h * 2.5):
                clusters.append(cur)
                cur = [w]
            else:
                cur.append(w)
        clusters.append(cur)

        for cluster in clusters:
            text = ' '.join(w['text'] for w in cluster)
            x0 = min(w['box'][0] for w in cluster)
            y0 = min(w['box'][1] for w in cluster)
            x1 = max(w['box'][2] for w in cluster)
            y1 = max(w['box'][3] for w in cluster)
            conf = min(w['conf'] for w in cluster)
            out.append((text, (x0, y0, x1, y1), conf))
    return out


def replace_font(img, font_path=LIBERATION_SANS, size_scale=1.0, min_confidence=40,
                 fill_colour=None, text_colour=(0, 0, 0), rotated_regions=()):
    """Detect every text element via OCR, blank it, and redraw it in `font_path`.

    `font_path` defaults to Liberation Sans, a free font metric-compatible with Arial --
    not Arial itself, which is proprietary and not present in this environment. This is the
    standard substitute; it is not pixel-identical to Arial.

    `fill_colour`, if None, is sampled locally around each text box (the colour just
    outside it) rather than assumed to be a flat white -- so this also works on text sitting
    over a light background tint, within reason. It will not work well on text sitting over
    a busy, varying background (a colour gradient, a photo), since there is no single
    correct fill colour to blank it with in that case.

    `rotated_regions`: plain OCR fails on rotated text -- on a real chart, a 90-degree
    y-axis label and colorbar label read as garbage ('ov', '\u20ac', '&', 'avvyv') while
    every horizontal label read correctly at 85%+ confidence. Automatic rotation-detection
    is a harder, separate problem this doesn't attempt; instead, like
    `advanced.remove_object`'s explicit region, the caller supplies
    `[(box, angle_degrees), ...]` for any label they can see is rotated. Each region is
    rotated by -angle before OCR (making it horizontal), redrawn, then rotated back.

    Returns (result_image, [(original_text, box, redrawn_size), ...]) so the caller can
    verify what OCR actually read before trusting the output.
    """
    out = img.convert('RGB').copy()
    log = []
    handled_boxes = []

    for box, angle in rotated_regions:
        x0, y0, x1, y1 = box
        handled_boxes.append(box)
        patch = out.crop((x0, y0, x1, y1))
        rotated = patch.rotate(-angle, expand=True, fillcolor=(255, 255, 255))
        cleaned_rot, sub_log = replace_font(rotated, font_path=font_path,
                                            size_scale=size_scale,
                                            min_confidence=min_confidence,
                                            fill_colour=fill_colour,
                                            text_colour=text_colour)
        back = cleaned_rot.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        # centre-paste back, since expand=True changes canvas size on each rotation
        bx = (back.width - patch.width) // 2
        by = (back.height - patch.height) // 2
        back_cropped = back.crop((bx, by, bx + patch.width, by + patch.height))
        out.paste(back_cropped, (x0, y0))
        for txt, sbox, sz in sub_log:
            log.append((txt, box, sz))   # report in terms of the outer region

    draw = ImageDraw.Draw(out)
    a = np.array(out)
    # Detect on the unblanked image, then exclude handled regions at the word level, rather
    # than blanking them on a scratch copy before OCR: painting large white rectangles
    # changes tesseract's own page-layout segmentation, and can drop tick labels at the
    # extreme ends of an axis from detection entirely rather than just leaving them
    # unswapped. Excluding by coordinate after detection avoids perturbing layout analysis.
    regions = find_text_regions(out, min_confidence=min_confidence,
                                exclude_boxes=handled_boxes)

    # Second pass over the strip beside each handled region. Even after a rotated title is
    # correctly redrawn and excluded by coordinate, its tall glyphs still sit in the same
    # OCR "line" as the adjacent tick-label column and perturb segmentation there, which can
    # drop a nearby tick label from every full-image pass. OCR'ing a narrow strip that
    # contains only the tick column recovers it, since there is no large neighbouring text
    # left to merge with. Results are merged by position, so a label found in both passes is
    # not drawn twice.
    seen = {(r[1][0], r[1][1]) for r in regions}
    for hx0, hy0, hx1, hy1 in handled_boxes:
        strip = (hx1, max(0, hy0 - 10), min(out.width, hx1 + 90), min(out.height, hy1 + 10))
        if strip[2] - strip[0] < 10:
            continue
        sub_img = out.crop(strip)
        for txt, (sx0, sy0, sx1, sy1), conf in find_text_regions(
                sub_img, min_confidence=min_confidence):
            gx0, gy0 = sx0 + strip[0], sy0 + strip[1]
            gx1, gy1 = sx1 + strip[0], sy1 + strip[1]
            if any(abs(gx0 - px) < 12 and abs(gy0 - py) < 8 for px, py in seen):
                continue
            seen.add((gx0, gy0))
            regions.append((txt, (gx0, gy0, gx1, gy1), conf))

    measured = []
    for txt, (x0, y0, x1, y1), conf in regions:
        # Exclusion of already-rotated-and-fixed regions happens inside find_text_regions,
        # at the word level before line-grouping (see its docstring), so a handled region's
        # fragments never get merged into a neighbouring tick label here.

        # Strip leading characters that are actually marker glyphs misread as text. A legend
        # entry is "<marker> <label>", and the marker sits inside the text line tesseract
        # returns -- on a real chart, an up-triangle marker came back as 'A', a short line
        # as '——', a filled dot as '@'. Redrawing those as characters both destroys the real
        # marker (it gets blanked) and prints a wrong glyph in its place. Only a leading
        # token is stripped, and only when it is a single non-alphanumeric symbol or a lone
        # capital with no lowercase following -- so real words survive.
        cleaned = txt
        parts = txt.split(' ', 1)
        if len(parts) == 2:
            head, rest = parts
            looks_like_marker = (
                not any(c.isalnum() for c in head)          # '——', '@', '*'
                or (len(head) == 1 and head.isupper())      # 'A' from a triangle
            )
            if looks_like_marker and rest:
                cleaned = rest
                # keep the marker's pixels: shrink the blanking box to start where the
                # actual label text begins, so the real glyph is never painted over
                shift = int((x1 - x0) * (len(head) + 1) / max(len(txt), 1))
                x0 = min(x0 + shift, x1 - 1)

        # Re-measure a tight bounding box of actual dark pixels within tesseract's reported
        # box, rather than trusting its height directly: a stray gridline sliver pulled into
        # the OCR box can report a label taller than its neighbours, and sizing naively
        # would render it visibly larger than the rest. Falls back to the raw box if no dark
        # pixels are found (shouldn't normally happen for real text).
        # Size from a reference height, not this label's own measured height. Measuring each
        # label independently makes size depend on which characters it happens to contain --
        # a label with no ascender and no digits measures shorter than one with a capital,
        # an 'f' and a 'g', even from the same legend at the same true font size, which
        # would reproduce the "legend entries have different font sizes" complaint. Instead,
        # measure the x-height-independent cap height: use the tallest dark run in the box,
        # then scale so a typical mixed-case label lands consistently.
        sub = a[y0:y1, x0:x1]
        dark_rows = np.where((sub.max(axis=2) < 150).any(axis=1))[0]
        if len(dark_rows):
            h = int(dark_rows.max() - dark_rows.min()) + 1
        else:
            h = y1 - y0
        # Characters that reach neither above the x-height nor below the baseline make a
        # label measure short; nudge those up toward the group norm rather than shrinking
        # them. Digits and capitals are full-height references.
        has_tall = any(c.isupper() or c.isdigit() or c in 'bdfhklt' for c in cleaned)
        has_descender = any(c in 'gjpqy,' for c in cleaned)
        if not has_tall:
            h = int(h * 1.35)
        elif not has_descender:
            h = int(h * 1.05)
        measured.append({'txt': cleaned, 'box': (x0, y0, x1, y1), 'h': h,
                        'fill_box': None})

    # Normalise sizes across labels that clearly belong to the same group (same legend, same
    # tick axis). Per-label measurement is unavoidably noisy: an adjacent marker glyph can
    # distort a label's box geometry before any measurement happens, so no amount of careful
    # pixel-measuring inside a wrong box recovers it. Labels whose vertical centres fall
    # within one text height of each other are treated as one group and take the group's
    # median height.
    if measured:
        heights = [m['h'] for m in measured]
        for m in measured:
            cy = (m['box'][1] + m['box'][3]) / 2
            cx0 = m['box'][0]
            # Peers are labels in the same row (a legend line, an x-axis tick row) or the
            # same column (a y-axis tick column, a colorbar tick column). Row-only grouping
            # is not enough: y-ticks are stacked vertically, so each one would only ever peer
            # with its immediate vertical neighbours, leaving labels recovered by the strip
            # second-pass measured against a different local group and rendered visibly
            # larger than the rest of the same axis.
            peers = [o['h'] for o in measured
                     if (abs(((o['box'][1] + o['box'][3]) / 2) - cy) < max(heights) * 2.2
                         and abs(o['box'][0] - cx0) < 900)
                     or abs(o['box'][0] - cx0) < 15]
            if len(peers) >= 2:
                m['h'] = int(np.median(peers))

    for m in measured:
        cleaned = m['txt']
        x0, y0, x1, y1 = m['box']
        h = m['h']
        pad = max(1, h // 4)
        bx0, by0 = max(0, x0 - pad), max(0, y0 - pad)
        bx1, by1 = min(a.shape[1], x1 + pad), min(a.shape[0], y1 + pad)

        if fill_colour is None:
            ring = []
            if by0 > 0:
                ring.append(a[max(0, by0 - 3):by0, bx0:bx1].reshape(-1, 3))
            if by1 < a.shape[0]:
                ring.append(a[by1:min(a.shape[0], by1 + 3), bx0:bx1].reshape(-1, 3))
            ring = np.concatenate(ring) if ring else np.array([[255, 255, 255]])
            local_fill = tuple(int(v) for v in np.median(ring, axis=0))
        else:
            local_fill = fill_colour

        draw.rectangle([bx0, by0, bx1, by1], fill=local_fill)
        font_size = max(6, int(h * size_scale))
        font = ImageFont.truetype(font_path, font_size)
        draw.text((x0, y0 - max(0, (font_size - h) // 2)), cleaned, font=font, fill=text_colour)
        log.append((cleaned, (x0, y0, x1, y1), font_size))

    return out, log
