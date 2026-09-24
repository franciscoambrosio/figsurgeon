"""Object-level editing: segmentation from a caller-supplied box.

Everything else here separates pixels by colour (`photo.py`) or subject-vs-background
(`advanced.remove_background`); neither helps when the target shares a hue with its
surroundings and isn't the whole subject ("the suit, not the flag"). Given a rough box,
this returns a real object-shaped mask.

Two backends: SlimSAM (`nielsr/slimsam-77-uniform`, ~40 MB) is better on photographs;
GrabCut is better on flat-colour figures, where SlimSAM tends to return the background
inside the box instead. `backend='auto'` (`choose_backend`) picks between them on a
flatness statistic; neither backend is deprecated, and either can be named explicitly.
Neither backend fixes a bad box -- both just segment whatever dominates it.

Needs a box (or points) from the caller; there is no detection here. GrabCut degrades
when the box covers nearly the whole frame (nothing left to model as background), and
neither backend disambiguates a second object also sitting inside the box.

`segment_object` reseeds OpenCV's global RNG before each GrabCut call: GrabCut's k-means
init otherwise isn't reproducible across identical calls.
"""
import numpy as np
from PIL import Image

from .advanced import _require
from .composite import _arr, _blend

MODEL_ID = 'nielsr/slimsam-77-uniform'   # Apache-2.0, ~40 MB, the checkpoint grounding.py
_SAM = {}                                # sharpens with. Loaded once, not once per call.


def available():
    """True when the SlimSAM backend can run. `backend='auto'` asks before choosing."""
    try:
        import torch                                                     # noqa: F401
        import transformers                                              # noqa: F401
        return True
    except ImportError:
        return False


def _slimsam():
    """The SlimSAM model and processor, loaded once and kept: a per-call load would roughly
    double the cost of every object edit."""
    if MODEL_ID not in _SAM:
        try:
            import torch                                                 # noqa: F401
            from transformers import AutoProcessor, SamModel
        except ImportError:
            raise ImportError(
                'the SlimSAM segmentation backend needs torch + transformers. Install '
                'them with: pip install "figsurgeon[grounding]"  (CPU torch is fine; the '
                'model is Apache-2.0, ~40 MB, and downloads on first use). Or pass '
                "backend='grabcut' for the OpenCV segmenter, which needs neither.")
        _SAM[MODEL_ID] = (SamModel.from_pretrained(MODEL_ID).eval(),
                          AutoProcessor.from_pretrained(MODEL_ID))
    return _SAM[MODEL_ID]


FLAT_FIGURE = 0.6      # above this share of exactly-flat pixels, treat it as a figure


def _flat_fraction(img, cap=400_000):
    """Share of pixels identical to both their right and lower neighbour.

    A rendered figure is mostly exactly-flat fill; a photograph almost never is, since
    sensor noise reaches every pixel. Subsampled to ~400k pixels, so it costs a few
    milliseconds on any image.
    """
    a = np.asarray(img.convert('RGB')).astype(np.int16)
    H, W = a.shape[:2]
    step = max(1, int((H * W / float(cap)) ** 0.5))
    a = a[::step, ::step]
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    right = np.abs(np.diff(a, axis=1)).max(axis=2)
    down = np.abs(np.diff(a, axis=0)).max(axis=2)
    return float(((right[:-1, :] == 0) & (down[:, :-1] == 0)).mean())


def choose_backend(img):
    """What `backend='auto'` picks, and why it is not simply "SlimSAM if installed".

    SlimSAM is better on photographs, but on a flat-colour figure with a loose box it does
    the opposite of what was asked: it returns the background inside the box and excludes
    the element, sometimes IoU 0.00 where GrabCut scores 1.00. There is no mask-quality
    check to catch that, so the failure is silent, and charts are what this package is for.
    GrabCut's model -- colours inside the box against colours outside it -- is the right
    one for flat fills, and its cost is negligible at figure sizes. See
    `evals/segmentation_models/` for the measurements behind this.
    """
    if not available():
        return 'grabcut'
    return 'grabcut' if _flat_fraction(img) > FLAT_FIGURE else 'slimsam'


def _normalise_points(points):
    """Accept (x, y), [(x, y), ...] or [[x, y], ...] and return a list of int pairs."""
    if points is None:
        return None
    pts = [points] if (len(points) == 2 and not hasattr(points[0], '__len__')) else list(points)
    out = []
    for p in pts:
        if len(p) != 2:
            raise ValueError(f'a point is (x, y); got {p!r}')
        out.append((int(p[0]), int(p[1])))
    return out


def _slimsam_points_binary(img, points):
    """SlimSAM's mask for a list of points on the object, as a 0/1 array.

    Answers the one failure no segmenter fixes from a box: a box that is mostly
    background. A point is not a correction to a box -- this checkpoint's box prompt
    dominates a point rather than being refined by it, so a point replaces the box
    entirely, and passing both is an error rather than a silently ignored argument.
    """
    model, proc = _slimsam()
    import torch
    inputs = proc(images=img.convert('RGB'),
                  input_points=[[[[float(x), float(y)] for x, y in points]]],
                  input_labels=[[[1] * len(points)]], return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs['original_sizes'].cpu(),
        inputs['reshaped_input_sizes'].cpu())[0]
    masks = masks[0] if masks.ndim == 4 else masks
    scores = np.ravel(out.iou_scores.cpu().numpy())
    return np.asarray(masks[int(np.argmax(scores))]).astype(np.uint8)


def _slimsam_binary(img, box):
    """SlimSAM's mask for `box`, as a full-resolution 0/1 array. Cost is nearly flat across
    image sizes, since a SAM encoder always sees 1024x1024; GrabCut's cost is not.
    """
    model, proc = _slimsam()      # first, so a missing torch names the install line
    import torch
    rgb = img.convert('RGB')
    inputs = proc(images=rgb, input_boxes=[[[float(v) for v in box]]], return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs['original_sizes'].cpu(),
        inputs['reshaped_input_sizes'].cpu())[0]
    masks = masks[0] if masks.ndim == 4 else masks
    scores = np.ravel(out.iou_scores.cpu().numpy())
    return np.asarray(masks[int(np.argmax(scores))]).astype(np.uint8)


def _slimsam_binaries(img, boxes):
    """SlimSAM's masks for several boxes on one photograph, in a single forward pass.

    A SAM call's cost is the image encoder, which doesn't depend on the prompt, so batching
    several boxes on one photograph is much cheaper than calling `_slimsam_binary` per box.
    The decoder still runs per prompt, so each mask matches the one-at-a-time result to
    within a boundary pixel, not bit-exactly.
    """
    model, proc = _slimsam()      # first, so a missing torch names the install line
    import torch
    rgb = img.convert('RGB')
    inputs = proc(images=rgb, input_boxes=[[[float(v) for v in b] for b in boxes]],
                  return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs['original_sizes'].cpu(),
        inputs['reshaped_input_sizes'].cpu())[0]          # (nb_boxes, nb_masks, H, W)
    scores = out.iou_scores.cpu().numpy()[0]              # (nb_boxes, nb_masks)
    return [np.asarray(masks[i][int(np.argmax(scores[i]))]).astype(np.uint8)
            for i in range(len(boxes))]


def segment_object(img, box=None, iterations=3, feather=2, max_pixels=6_000_000,
                   backend='auto', follow_object=True, points=None, _binary=None):
    """Segment the object inside `box`. Returns (mask_0_to_1, coverage), plus `.region`.

    `mask` is a float array the size of `img`, feathered at the edges rather than hard 0/1.
    `coverage` is the fraction of the box (not the whole image) classified as foreground; a
    value near 1.0 usually means the box was tight around a busy/textured region and the
    segmenter gave up and kept everything, not that it found a clean object.

    `backend` is 'slimsam', 'grabcut', or 'auto' (see `choose_backend`); 'auto' also falls
    back to GrabCut whenever torch + transformers aren't importable.

    `follow_object=True` (default) lets a SAM mask extend past the prompt box to the
    object's real extent (e.g. a sleeve below the drawn box), kept only where connected to
    what's inside the box -- the box still decides which thing is edited, not where it
    ends. The region actually touched comes back as `.region`. `follow_object=False` clips
    the mask to the box instead.
    """
    points = _normalise_points(points)
    if points and box is not None:
        raise ValueError(
            'give a box OR points, not both: measured on the rocket photograph, this '
            'checkpoint ignores points when a box is present (mask 50.7% of the frame '
            'with a point against 49.5% without, the same sky either way), so accepting '
            'both would silently drop the point the caller relied on.')
    if not points and box is None:
        raise ValueError('segment_object needs a box, or points on the object')
    if backend == 'auto':
        backend = 'slimsam' if points else choose_backend(img)
    if backend not in ('slimsam', 'grabcut'):
        raise ValueError(f"backend must be 'slimsam', 'grabcut' or 'auto', not {backend!r}")
    if points and backend != 'slimsam':
        raise ValueError('points need the SlimSAM backend (GrabCut segments from a box '
                         'only). Install figsurgeon[grounding], or pass a box.')

    if points:
        binary = _slimsam_points_binary(img, points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        # The "box" a point prompt is measured against is the point itself: coverage is
        # meaningless here, and the region comes from the mask, reported as always.
        x0, y0, x1, y1 = min(xs), min(ys), max(xs) + 1, max(ys) + 1
    elif backend == 'slimsam':
        x0, y0, x1, y1 = (int(v) for v in box)
        # `_binary` is the already-computed mask for this box, when several boxes on the
        # same photograph were segmented in one encoder pass (`segment_objects`).
        binary = _slimsam_binary(img, (x0, y0, x1, y1)) if _binary is None else _binary
    else:
        x0, y0, x1, y1 = (int(v) for v in box)
        binary = _grabcut_binary(img, box, iterations, max_pixels)

    if points:
        coverage = float(binary.mean())          # of the frame: there is no box to divide by
    else:
        box_pixels = binary[y0:y1, x0:x1]
        coverage = float(box_pixels.mean()) if box_pixels.size else 0.0

    if follow_object:
        binary = _connected_to_box(binary, (x0, y0, x1, y1))
    if points:
        # A point prompt has no box to grow from: the region IS the mask, and is reported.
        from .photo import _feather
        soft = _feather(binary, feather)
        return Located((soft, coverage), _mask_region(soft, (x0, y0, x1, y1), soft.shape), 0.0)

    # Feather the edge outward only -- see photo._feather. A blur applied to the mask
    # itself weakens thin parts of the object as well as softening its boundary, which on a
    # narrow object means the recolour or cutout comes out at partial strength.
    from .photo import _feather
    soft = _feather(binary, feather)
    if not follow_object:
        # Clip the feathered mask back to the box: without this, edge softening alone
        # spreads the mask outside the region the caller named.
        clipped = np.zeros_like(soft)
        clipped[y0:y1, x0:x1] = soft[y0:y1, x0:x1]
        return Located((clipped, coverage), (x0, y0, x1, y1), 0.0)

    region = _mask_region(soft, (x0, y0, x1, y1), soft.shape)
    total = float((soft > 0).sum())
    inside = float((soft[y0:y1, x0:x1] > 0).sum())
    outside_frac = 0.0 if total == 0 else 1.0 - inside / total
    return Located((soft, coverage), region, outside_frac)


class Located(tuple):
    """What a box-driven operation returns, plus where it actually reached.

    A tuple subclass, like `tools.ToolResult`: every existing `mask, coverage =
    segment_object(...)` keeps working unchanged, with two extra facts attached --

      `region`       the box the edit may touch: the caller's box unioned with the extent
                     of the object found in it. `composite.enforce_region` is given this
                     rather than the raw box, so localisation is checked against what the
                     tool actually claims instead of being kept by cutting the object in
                     half.
      `outside_frac` how much of the mask fell outside the caller's box -- the signal that
                     the box was drawn too tight.
    """
    def __new__(cls, values, region, outside_frac):
        self = super().__new__(cls, tuple(values))
        self.region = tuple(int(v) for v in region)
        self.outside_frac = float(outside_frac)
        return self


def _connected_to_box(binary, box):
    """Mask components that touch the box, at full extent; everything else dropped.

    Keeps the box meaningful once the mask is allowed to leave it: a component that never
    touches the box is not what the caller pointed at, even if the segmenter also found it.
    """
    from scipy import ndimage
    x0, y0, x1, y1 = box
    labels, n = ndimage.label(binary)
    if n == 0:
        return binary
    inside = np.unique(labels[y0:y1, x0:x1])
    inside = inside[inside > 0]
    if inside.size == 0:
        return np.zeros_like(binary)
    return np.isin(labels, inside).astype(binary.dtype)


def _mask_region(mask, box, shape):
    """The caller's box unioned with the bounding box of the mask, clamped to the image."""
    H, W = shape[:2]
    ys, xs = np.nonzero(mask > 0)
    x0, y0, x1, y1 = box
    if xs.size:
        x0, y0 = min(x0, int(xs.min())), min(y0, int(ys.min()))
        x1, y1 = max(x1, int(xs.max()) + 1), max(y1, int(ys.max()) + 1)
    return (max(0, x0), max(0, y0), min(W, x1), min(H, y1))


def _grabcut_binary(img, box, iterations=3, max_pixels=6_000_000):
    """GrabCut's mask for `box`, as a full-resolution 0/1 array."""
    cv2 = _require('cv2', 'advanced')
    a = np.array(img.convert('RGB'))
    H, W = a.shape[:2]
    x0, y0, x1, y1 = box
    box_area = max(0, x1 - x0) * max(0, y1 - y0)
    if box_area >= 0.98 * W * H:
        raise ValueError(
            f'box {box} covers {box_area / (W * H):.0%} of the {W}x{H} image -- GrabCut '
            f'builds its background colour model from pixels OUTSIDE the box, so a box '
            f'this large leaves nothing to learn "background" from. Pass a box that '
            f'excludes a real margin around the object.')

    # GrabCut is slow at default settings on real (multi-megapixel) photographs. Two levers
    # bring the cost down safely: fewer iterations (used here; costs little accuracy) and
    # downscaling large frames (used here, with a high cap, since a small subject can
    # disappear entirely at low resolution). Cropping to the box was rejected: it changes
    # the answer, since GrabCut's background model needs the whole frame's surroundings,
    # not just a local crop. For interactive work on a large image, resize first.
    scale = 1.0
    work = a
    if H * W > max_pixels:
        scale = (max_pixels / float(H * W)) ** 0.5
        work = np.array(img.convert('RGB').resize(
            (max(1, int(W * scale)), max(1, int(H * scale))), Image.BILINEAR))
    wH, wW = work.shape[:2]

    rx0 = int(min(max(0, round(x0 * scale)), wW - 2))
    ry0 = int(min(max(0, round(y0 * scale)), wH - 2))
    rx1 = max(rx0 + 1, int(min(round(x1 * scale), wW)))
    ry1 = max(ry0 + 1, int(min(round(y1 * scale), wH)))

    mask = np.zeros((wH, wW), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (rx0, ry0, rx1 - rx0, ry1 - ry0)
    # cv2.grabCut's k-means init draws from cv2's global RNG, which isn't reset per call, so
    # identical calls can land in a degenerate init and return a different result.
    # Reseeding before every call makes it reproducible.
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(np.ascontiguousarray(work), mask, rect, bgd_model, fgd_model,
                    iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        raise ValueError(f'GrabCut failed on box {box}: {e}. Usually means the box is too '
                         f'small or too close to the image edge to sample a background '
                         f'model from -- try a box with more margin around the object.')

    binary = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    if scale != 1.0:
        # Bilinear then thresholded, not nearest-neighbour: nearest-neighbour upscaling of a
        # binary mask leaves visibly stepped edges.
        binary = np.array(Image.fromarray(binary * 255).resize((W, H), Image.BILINEAR))
        binary = (binary > 127).astype(np.uint8)
    return binary


def segment_objects(img, boxes=None, iterations=3, feather=2, backend='auto',
                    follow_object=True, points=None):
    """Segment several boxes and return the union of their masks, plus per-box coverage.

    "Colour pop the eyes", "recolour both headlights", "cut out the two logos" -- these are
    ordinary requests, and running `segment_object` once per box and compositing separately
    is wrong for the isolating operations specifically: the second pass desaturates
    everything outside its box, including the object the first pass just kept. The union has
    to be built before the edit, not after.

    Coverage is reported per box rather than pooled, so a caller can tell which region failed
    to segment -- a pooled average hides one box finding nothing behind another finding
    everything.
    """
    if not boxes and points is None:
        raise ValueError('no boxes given')
    if points is not None:
        seg = segment_object(img, None, iterations=iterations, feather=feather,
                             backend=backend, follow_object=follow_object, points=points)
        return Located((seg[0], [seg[1]]), seg.region, seg.outside_frac)
    # One encoder pass for the whole photograph when SlimSAM is what will run: the boxes
    # are prompts against a shared encoding, and paying for it per box is the whole of the
    # extra cost. GrabCut has no shared work, and one box has nothing to share.
    resolved = choose_backend(img) if backend == 'auto' else backend
    binaries = (_slimsam_binaries(img, [tuple(int(v) for v in b) for b in boxes])
                if resolved == 'slimsam' and len(boxes) > 1 else [None] * len(boxes))

    masks, coverages, regions, outside = [], [], [], []
    for box, binary in zip(boxes, binaries):
        seg = segment_object(img, box, iterations=iterations, feather=feather,
                             backend=resolved, follow_object=follow_object, _binary=binary)
        masks.append(seg[0])
        coverages.append(seg[1])
        regions.append(seg.region)
        outside.append(seg.outside_frac)
    region = (min(r[0] for r in regions), min(r[1] for r in regions),
              max(r[2] for r in regions), max(r[3] for r in regions))
    return Located((np.maximum.reduce(masks), coverages), region, max(outside))


def _as_boxes(box, boxes, points=None):
    """Accept either a single `box`, a list of `boxes`, or points instead of both."""
    if boxes:
        return [tuple(int(v) for v in b) for b in boxes]
    if box is None:
        if points is not None:
            return []                     # the prompt is the points; there is no box
        raise ValueError('a box (or boxes, or points on the object) is required')
    return [tuple(int(v) for v in box)]


def recolour_object(img, box=None, to_rgb=None, preserve_shading=True, iterations=3,
                    feather=2, boxes=None, backend='auto', follow_object=True,
                    points=None):
    """Recolour the segmented object to `to_rgb`, keeping its own shading.

    Unlike `photo.replace_colour` (hue-based -- recolours every pixel near a hue anywhere in
    the frame), this is spatial: it recolours only the object found inside `box`, regardless
    of same-hue content elsewhere (a red spacesuit against a flag with red stripes).

    `preserve_shading=True` scales the target colour by each pixel's own relative luminance
    (clipped to [0.35, 1.9]) instead of flattening to a flat fill, so folds and highlights
    survive -- same technique as `photo.replace_colour`.
    """
    regions = _as_boxes(box, boxes, points)
    seg = segment_objects(img, regions, iterations=iterations, feather=feather,
                          backend=backend, follow_object=follow_object, points=points)
    mask, coverages = seg
    coverage = coverages[0] if len(coverages) == 1 else float(np.mean(coverages))
    a = _arr(img)
    target = np.array(to_rgb, dtype=float)

    if preserve_shading:
        lum = a.mean(axis=2) / 255.0
        target_lum = target.mean() / 255.0
        target_lum = max(target_lum, 1e-6)
        scale = np.clip(lum / target_lum, 0.35, 1.9)
        recoloured = target[None, None, :] * scale[:, :, None]
    else:
        recoloured = np.broadcast_to(target, a.shape)

    out = _blend(a, np.clip(recoloured, 0, 255), mask)
    return Located((Image.fromarray(out.astype(np.uint8)), coverage),
                   seg.region, seg.outside_frac)


def isolate_object(img, box=None, flatten=0.5, iterations=3, feather=2, boxes=None,
                   backend='auto', follow_object=True, points=None):
    """Keep the segmented object in colour, desaturate everything else.

    The spatial counterpart to `photo.isolate_colour`, and works where that provably fails:
    when the subject and background share a hue, no hue tolerance separates them, but a box
    around the subject still lets the segmenter find its shape.

    `flatten` behaves like `photo.isolate_colour`'s: 0 = true per-pixel luminance in the
    desaturated area, 1 = flat mid-grey. Same reason it exists there -- accurate desaturation
    of a naturally bright region can look barely different from the original.
    """
    regions = _as_boxes(box, boxes, points)
    seg = segment_objects(img, regions, iterations=iterations, feather=feather,
                          backend=backend, follow_object=follow_object, points=points)
    mask, coverages = seg
    coverage = coverages[0] if len(coverages) == 1 else float(np.mean(coverages))
    a = _arr(img)
    lum = a.mean(axis=2, keepdims=True)
    grey_true = np.repeat(lum, 3, axis=2)
    grey_flat = np.full_like(a, 128.0)
    grey = grey_true * (1 - flatten) + grey_flat * flatten

    out = _blend(grey, a, mask)
    return Located((Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), coverage),
                   seg.region, seg.outside_frac)


def _inpaint_coverage_warning(frac, subject):
    """Shared caution for `erase_object` and `scale_object`'s shrink path: inpainting has no
    information about what was behind a large region and interpolates surrounding texture
    instead of reconstructing it. Returns None under the 8% frame-coverage cut this is
    calibrated at.
    """
    if frac <= 0.08:
        return None
    return (f'{subject} covers {frac:.0%} of the frame -- inpainting has no information '
            f'about what was actually behind it and interpolates surrounding texture '
            f'instead of reconstructing it. Measured: convincing at ~1% of frame (a thin '
            f'mast against sky), smears visibly at ~33% (a cup on a table). Inspect the '
            f'result rather than trusting it.')


def _lama_coverage_warning(frac, subject):
    """The same caution as `_inpaint_coverage_warning`, for the opposite failure: TELEA
    smears visibly when it lacks information, while LaMa instead predicts a plausible fill
    that can look convincing yet be entirely invented. Same 8% cut.
    """
    if frac <= 0.08:
        return None
    return (f'{subject} covers {frac:.0%} of the frame -- at this size the fill is PREDICTED '
            f'rather than interpolated, and a prediction that looks like the background is '
            f'not the background. Measured: erasing a horse at 23% of frame reconstructed a '
            f'hedge behind it that was never in the picture. Inspect the result rather than '
            f'trusting it.')


def erase_object(img, box=None, radius=6, grow=3, iterations=3, feather=2, backend='auto',
                 follow_object=True, points=None, fill='auto'):
    """Segment the object inside `box` and inpaint it out. Returns (img, coverage, warning).

    Dilates the segmented mask by `grow` pixels first so the object's own anti-aliased
    edge doesn't survive as a faint outline.

    `backend` chooses the segmenter (what to erase); `fill` chooses the inpainter (what to
    put there): 'auto' (default) uses LaMa when `inpaint.available()`, else TELEA; 'lama'
    always LaMa; 'telea' always `cv2.inpaint`, right for a thin hole in a uniform surround
    (a mast against sky) where there's real structure to propagate from.

    `warning` fires once the segmented region exceeds ~8% of the frame, since inpainting
    has no information about what was actually behind a large region -- TELEA then smears
    visibly, LaMa invents plausible-looking content instead. `verify_photo.check_object_erased`
    grades the result.
    """
    cv2 = _require('cv2', 'advanced')
    seg = segment_object(img, box, iterations=iterations, feather=feather,
                         backend=backend, follow_object=follow_object, points=points)
    mask, coverage = seg
    a = np.array(img.convert('RGB'))
    H, W = a.shape[:2]

    binary = (mask > 0.5).astype(np.uint8) * 255
    if grow > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
        binary = cv2.dilate(binary, kernel)

    frame_frac = float((binary > 0).mean())
    if fill not in ('auto', 'lama', 'telea'):
        raise ValueError(f"fill must be 'auto', 'lama' or 'telea', not {fill!r}")
    from . import inpaint as I
    use_lama = fill == 'lama' or (fill == 'auto' and I.available())
    if use_lama:
        out = I.lama(a, binary)
        warning = _lama_coverage_warning(frame_frac, 'segmented region')
    else:
        out = cv2.inpaint(a, binary, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        warning = _inpaint_coverage_warning(frame_frac, 'segmented region')
    return Located((Image.fromarray(out), frame_frac, warning), seg.region,
                   seg.outside_frac)


def scale_object(img, box=None, scale=1.0, iterations=3, feather=2, backend='auto',
                 points=None):
    """Segment the object inside `box` and resize it, in place, about its own centre.

    Unlike `tools.resize` (the whole frame), this resizes one object in place. Scaled about
    the mask's centroid rather than the box's corner or centre, since the segmented mask
    routinely isn't centred in the box the caller drew (a sleeve extending past one edge),
    and the centroid is the point that keeps the object's weight in place as it resizes.

    Shrinking inpaints the whole original footprint, not just the newly-exposed ring:
    inpainting only the ring biases the reconstruction toward the object's own colour,
    since the object's still-full-colour interior is the nearest unmasked neighbour on the
    ring's inner edge. Growing (`scale > 1`) needs no inpainting, since the larger object is
    pasted directly over the original pixels.

    Returns `(image, frame_frac, warning)` via `Located`, with `.region` the union of where
    the object was and where it ended up (for `scale > 1` that reaches past the caller's
    box). `warning` reuses `erase_object`'s 8% frame-coverage threshold, since it's the same
    inpainting call on the same kind of ring.
    """
    if scale <= 0:
        raise ValueError(f'scale must be > 0, got {scale}')
    cv2 = _require('cv2', 'advanced')
    seg = segment_object(img, box, iterations=iterations, feather=feather,
                         backend=backend, points=points)
    mask, coverage = seg
    a = np.array(img.convert('RGB'))
    H, W = a.shape[:2]

    binary = mask > 0.5
    ys, xs = np.nonzero(binary)
    if not xs.size:
        # No object to scale. `tools.py` turns frame_frac == 0 into the same
        # no-op-reporting-success guard `erase_object` already has.
        return Located((img, 0.0, None), seg.region, seg.outside_frac)

    cy, cx = float(ys.mean()), float(xs.mean())
    # Scale about (cx, cy): p' = centre + scale * (p - centre).
    M = np.array([[scale, 0.0, cx * (1.0 - scale)],
                  [0.0, scale, cy * (1.0 - scale)]])

    scaled_rgb = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    scaled_mask = cv2.warpAffine(mask.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    scaled_binary = scaled_mask > 0.5

    plate = a
    warning = None
    if scale < 1.0:
        grow = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
        old = cv2.dilate(binary.astype(np.uint8) * 255, kernel)
        if old.any():
            plate = cv2.inpaint(a, old, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
            vacated_frac = float((old > 0).mean())
            warning = _inpaint_coverage_warning(vacated_frac, "the object's original footprint")

    out = _blend(plate, scaled_rgb, scaled_mask)
    out_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    frame_frac = float(binary.mean())
    sys_, sxs = np.nonzero(scaled_binary)
    rx0, ry0, rx1, ry1 = seg.region
    if sxs.size:
        rx0, ry0 = min(rx0, int(sxs.min())), min(ry0, int(sys_.min()))
        rx1, ry1 = max(rx1, int(sxs.max()) + 1), max(ry1, int(sys_.max()) + 1)
    region = (max(0, rx0), max(0, ry0), min(W, rx1), min(H, ry1))

    return Located((out_img, frame_frac, warning), region, seg.outside_frac)


def extract_object(img, box=None, iterations=3, feather=2, backend='auto',
                   follow_object=True, points=None, _seg=None):
    """Segment the object inside `box` and return it as an RGBA cutout.

    `_seg` is a `segment_object` result the caller already has -- the tool dispatch
    segments first to check the coverage, and segmenting again here cost a second full
    pass (9.1 s against 6.0 s on a 1600x1200 photograph)."""
    seg = _seg if _seg is not None else segment_object(
        img, box, iterations=iterations, feather=feather, backend=backend,
        follow_object=follow_object, points=points)
    mask, coverage = seg
    a = np.array(img.convert('RGB'))
    alpha = np.clip(mask * 255, 0, 255).astype(np.uint8)
    rgba = np.dstack([a, alpha])
    return Image.fromarray(rgba)
