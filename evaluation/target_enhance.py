""" Target (building/vehicle/aircraft) visual-identifiability enhancement for
CUT outputs (fake_B) — stages A (CFAR + saliency detection) and B
(saliency-guided local enhancement) of docs/TARGET_ENHANCEMENT_SPEC.md.

Design principles (see the spec for full literature grounding):
  - Detection (stage A) uses CFAR, the SAR-standard adaptive local threshold,
    instead of a single global Otsu threshold (evaluation/rectify.py's
    approach) — far more robust on heterogeneous backgrounds (urban/coastal
    clutter), corroborated by a brightness-saliency floor to suppress
    isolated noisy-pixel false alarms, then cleaned up with morphology and
    split into connected components.
  - Enhancement (stage B) is ALWAYS blended by a smooth per-pixel saliency
    weight, never applied globally: every cited paper on this topic agrees
    that indiscriminate global sharpening/contrast just amplifies SAR-derived
    speckle in the background. Only pixels the detector considers
    target-like get sharpened/contrast-boosted; the rest stay close to the
    original.
  - Pure NumPy/Pillow for detection + unsharp/guided-filter enhancement (runs
    anywhere, no hard dependency). Masked CLAHE needs opencv and degrades with
    a clear message when unavailable, exactly like evaluation/rectify.py.

This module intentionally does NOT do stage C (class-aware shape/confidence
annotation) — that extends evaluation/rectify.py and is a separate task.
"""

import os
from collections import deque

import numpy as np

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Shared numeric helpers (self-contained on purpose — mirrors the small
# _box_filter/_local_stats utilities already duplicated across
# preprocessing/steps.py and preprocessing/img_metrics.py, so this module has
# no coupling to preprocessing/ internals).
# --------------------------------------------------------------------------- #

def _box_filter(x, w):
    """Mean over a w x w window (reflect pad) via integral image. Pure NumPy."""
    w = int(w)
    if w <= 1:
        return x.astype(np.float64, copy=True)
    if w % 2 == 0:
        w += 1
    r = w // 2
    xpad = np.pad(x, r, mode='reflect')
    cs = np.cumsum(np.cumsum(xpad, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode='constant')
    H, W = x.shape
    S = (cs[w:w + H, w:w + W] - cs[0:H, w:w + W]
         - cs[w:w + H, 0:W] + cs[0:H, 0:W])
    return S / float(w * w)


def _annulus_stats(x, guard, train):
    """Local mean/std over a TRAINING ANNULUS (outer window minus an inner
    guard window) around each pixel — the standard CFAR "reference cells"
    construction: the guard ring is excluded so the target itself (which may
    span a few pixels) doesn't leak into its own background estimate.

    guard: guard-ring half-width in pixels (>=0).
    train: training-ring half-width in pixels, OUTSIDE the guard ring (>=1).
    Returns (mean, std, n_annulus) — n_annulus is the (constant) cell count.
    """
    guard, train = int(guard), int(train)
    outer = guard + train
    w_outer, w_inner = 2 * outer + 1, 2 * guard + 1
    n_outer, n_inner = w_outer * w_outer, w_inner * w_inner
    n_annulus = n_outer - n_inner
    if n_annulus <= 0:
        raise ValueError('train ring must be larger than the guard ring')

    sum_outer = _box_filter(x, w_outer) * n_outer
    sum_inner = _box_filter(x, w_inner) * n_inner
    sq_outer = _box_filter(x * x, w_outer) * n_outer
    sq_inner = _box_filter(x * x, w_inner) * n_inner

    annulus_sum = sum_outer - sum_inner
    annulus_sq = sq_outer - sq_inner
    mean = annulus_sum / n_annulus
    var = np.maximum(annulus_sq / n_annulus - mean * mean, 0.0)
    return mean, np.sqrt(var), n_annulus


def to_luminance01(image):
    """HxW or HxWx3 uint8/float array -> HxW float64 luminance in [0,1]."""
    a = np.asarray(image).astype(np.float64)
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    if a.max() > 1.0:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Pure-NumPy binary morphology + connected components (no scipy/cv2 hard dep;
# cv2 is used opportunistically for connected components when available,
# since it's much faster, but a correct pure-Python fallback always works).
# --------------------------------------------------------------------------- #

def _binary_dilate(mask, iterations=1):
    m = mask
    for _ in range(max(0, int(iterations))):
        p = np.pad(m, 1, mode='constant', constant_values=False)
        m = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | p[1:-1, 1:-1])
    return m


def _binary_erode(mask, iterations=1):
    m = mask
    for _ in range(max(0, int(iterations))):
        p = np.pad(m, 1, mode='constant', constant_values=False)
        m = (p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:] & p[1:-1, 1:-1])
    return m


def binary_open(mask, iterations=1):
    """Erode then dilate: removes small isolated noise specks without
    shrinking the surviving (larger) target blobs."""
    return _binary_dilate(_binary_erode(mask, iterations), iterations)


def binary_close(mask, iterations=1):
    """Dilate then erode: fills small holes/gaps inside target blobs."""
    return _binary_erode(_binary_dilate(mask, iterations), iterations)


def _label_connected_components(mask):
    """4-connected component labeling. Uses cv2 when available (fast); falls
    back to a pure-Python BFS flood fill otherwise (always correct, fine for
    the sparse target masks typical of this use case). Returns (labels HxW
    int32, n_components) with background = 0."""
    mask = np.asarray(mask, dtype=bool)
    try:
        import cv2
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        return labels.astype(np.int32), int(n - 1)
    except Exception:
        pass

    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    visited = np.zeros((H, W), dtype=bool)
    current = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not visited[y, x]:
                current += 1
                visited[y, x] = True
                q = deque([(y, x)])
                while q:
                    cy, cx = q.popleft()
                    labels[cy, cx] = current
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
    return labels, current


# --------------------------------------------------------------------------- #
# Stage A: CFAR + saliency target detection
# --------------------------------------------------------------------------- #

def cfar_mask(lum01, guard=2, train=9, method='sigma', k_sigma=3.0, pfa=1e-3):
    """CFAR-style adaptive local threshold over a training annulus.

    method='sigma' (default): flag a pixel if it exceeds the LOCAL annulus
        mean by k_sigma local standard deviations. Distribution-free — the
        right choice for a rendered fake_B (RGB "optical-like" CUT output),
        where the classical exponential-clutter assumption behind CA-CFAR
        doesn't strictly hold (that assumption is derived for calibrated raw
        SAR intensity, not 8-bit rendered RGB).
    method='ca': classical Cell-Averaging CFAR threshold for exponentially
        distributed clutter power, T = N*(pfa^(-1/N) - 1), threshold =
        local_mean * (1+T). Provided for users applying this to real/
        calibrated SAR intensity (e.g. real_A) rather than rendered fake_B.

    lum01: HxW float in [0,1]. Returns a bool HxW mask.
    """
    mean, std, n = _annulus_stats(lum01, guard, train)
    if method == 'ca':
        pfa = float(np.clip(pfa, 1e-9, 0.5))
        t = n * (pfa ** (-1.0 / n) - 1.0)
        threshold = mean * (1.0 + t)
    elif method == 'sigma':
        threshold = mean + float(k_sigma) * std
    else:
        raise ValueError(f"unknown CFAR method: {method!r} (expected 'sigma' or 'ca')")
    return lum01 > threshold


def brightness_saliency01(image, window=5):
    """Local-contrast saliency in [0,1]: highlights small, LOCALLY bright
    compact regions (candidate strong reflectors: ships/vehicles/building
    corners) relative to their surroundings. Pure-NumPy port of
    models/losses_extra.py's reflector_saliency_map (torch, training-time),
    renormalised to [0,1] here for direct use as a stage-B blend weight
    rather than [1, 1+boost] loss-weight scale.
    """
    lum = to_luminance01(image)
    local_mean = _box_filter(lum, window)
    contrast = np.clip(lum - local_mean, 0.0, None)
    ref = float(np.quantile(contrast, 0.995))
    ref = max(ref, EPS)
    return np.clip(contrast / ref, 0.0, 1.0)


def detect_targets(image, guard=2, train=9, method='sigma', k_sigma=3.0, pfa=1e-3,
                   saliency_window=5, saliency_floor=0.15,
                   min_area=9, max_area_frac=0.2, morph_iterations=1):
    """Stage A end-to-end: CFAR detection, corroborated by a brightness-
    saliency floor (suppresses isolated single-pixel CFAR false alarms that
    aren't backed by genuine local contrast), morphological open+close
    cleanup, then connected-component + area filtering (same min/max-area
    philosophy as evaluation/rectify.py's detect_candidate_regions).

    Returns (target_mask, saliency_map, regions):
      target_mask  -- bool HxW: the final detection mask.
      saliency_map -- float HxW in [0,1]: smooth "target-likeness"
                      (brightness_saliency01), independent of the hard mask —
                      this is what stage B uses as its blend weight, so
                      enhancement fades smoothly at target edges instead of
                      having a hard seam at the mask boundary.
      regions      -- list of dicts, one per surviving connected component:
                      {'bbox': (x, y, w, h), 'area': int,
                       'centroid': (cx, cy), 'mean_saliency': float}.
    """
    lum = to_luminance01(image)
    cfar = cfar_mask(lum, guard=guard, train=train, method=method,
                     k_sigma=k_sigma, pfa=pfa)
    saliency = brightness_saliency01(image, window=saliency_window)
    corroborated = cfar & (saliency > float(saliency_floor))
    cleaned = binary_close(binary_open(corroborated, morph_iterations), morph_iterations)

    labels, n = _label_connected_components(cleaned)
    H, W = lum.shape
    max_area = float(max_area_frac) * H * W
    final_mask = np.zeros_like(cleaned)
    regions = []
    for i in range(1, n + 1):
        comp = labels == i
        area = int(comp.sum())
        if area < min_area or area > max_area:
            continue
        ys, xs = np.nonzero(comp)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        final_mask |= comp
        regions.append({
            'bbox': (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
            'area': area,
            'centroid': (float(xs.mean()), float(ys.mean())),
            'mean_saliency': float(saliency[comp].mean()),
        })
    return final_mask, saliency, regions


# --------------------------------------------------------------------------- #
# Stage B: saliency-guided local enhancement
# --------------------------------------------------------------------------- #

def _apply_per_channel(image, fn):
    """Run fn(channel_float64_0_255) -> channel_float64_0_255 on each channel
    of an HxW or HxWx3 uint8/float image; returns uint8, same shape."""
    arr = np.asarray(image).astype(np.float64)
    if arr.ndim == 2:
        return np.clip(fn(arr), 0, 255).astype(np.uint8)
    out = np.stack([fn(arr[..., c]) for c in range(arr.shape[-1])], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def unsharp_mask(image, weight_map, amount=0.6, radius=3):
    """Adaptive unsharp masking: sharpening strength at each pixel is
    proportional to weight_map (e.g. target saliency, in [0,1]) — targets get
    sharpened while the background stays close to the original, so SAR-
    derived speckle in non-target regions is not amplified.

    image: HxW or HxWx3 uint8. weight_map: HxW float in [0,1] (same H,W).
    amount: sharpening strength at weight_map=1 (0=no effect; ~0.3-1.0 is a
        reasonable range; values much above 1 start creating visible halos).
    radius: box-blur radius defining "detail" (high-frequency = original -
        blurred); larger radius picks up coarser structure.
    """
    w = np.asarray(weight_map, dtype=np.float64)

    def fn(ch):
        blurred = _box_filter(ch, 2 * radius + 1)
        detail = ch - blurred
        return ch + float(amount) * detail * w

    return _apply_per_channel(image, fn)


def guided_filter(p, guide=None, radius=5, eps=1e-2):
    """Edge-preserving guided filter (He, Sun & Tang, 2010), self-guided (a.k.a.
    "guided filter smoothing") when `guide` is None. Unlike a plain box/
    Gaussian blur, it smooths flat regions while preserving strong edges
    (locally-high-variance regions keep more of their original value) — so
    base/detail decomposition around this filter avoids the halo artefacts a
    naive blur causes near sharp object boundaries.

    p, guide: HxW float arrays, SAME scale (e.g. both in [0,1]). Returns an
    array the same shape as p.
    """
    if guide is None:
        guide = p
    w = 2 * int(radius) + 1
    mean_I = _box_filter(guide, w)
    mean_p = _box_filter(p, w)
    mean_Ip = _box_filter(guide * p, w)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = _box_filter(guide * guide, w)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _box_filter(a, w)
    mean_b = _box_filter(b, w)
    return mean_a * guide + mean_b


def guided_detail_boost(image, weight_map, boost=1.5, radius=5, eps=1e-2):
    """Base/detail decomposition via the (edge-preserving) guided filter,
    with weight_map-modulated MULTIPLICATIVE detail amplification —
    complements unsharp_mask: larger radius, edge-aware base, so this
    reinforces coarser target structure (e.g. a building's outline) rather
    than fine texture.

    image: HxW or HxWx3 uint8. weight_map: HxW float in [0,1].
    boost: detail gain at weight_map=1 (1.0 = no change; ~1.3-2.0 typical).
    """
    w = np.asarray(weight_map, dtype=np.float64)
    gain = 1.0 + (float(boost) - 1.0) * w

    def fn(ch01_source):
        # guided_filter expects a consistent scale; operate in [0,1] internally
        ch01 = ch01_source / 255.0
        base = guided_filter(ch01, radius=radius, eps=eps)
        detail = ch01 - base
        boosted = base + detail * gain
        return boosted * 255.0

    return _apply_per_channel(image, fn)


def masked_clahe(image, weight_map, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE (contrast-limited adaptive histogram equalisation) applied on the
    L channel (Lab colour space, to avoid the hue/saturation shifts naive
    per-RGB-channel CLAHE causes), then blended back in proportional to
    weight_map — so contrast enhancement only takes effect in high-saliency
    (target) regions instead of amplifying background speckle contrast
    everywhere. Requires opencv; raises ImportError with a clear message when
    unavailable (same fail-fast pattern as evaluation/rectify.py).
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "masked_clahe에는 opencv가 필요합니다: pip install opencv-python "
            "(또는 opencv-python-headless)") from exc

    arr = np.asarray(image)
    w = np.asarray(weight_map, dtype=np.float64)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=tuple(int(v) for v in tile_grid_size))
    if arr.ndim == 3:
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = clahe.apply(l_ch)
        l_blend = (w * l_eq.astype(np.float64) + (1 - w) * l_ch.astype(np.float64))
        l_blend = np.clip(l_blend, 0, 255).astype(np.uint8)
        merged = cv2.merge([l_blend, a_ch, b_ch])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    else:
        eq = clahe.apply(arr.astype(np.uint8))
        blend = w * eq.astype(np.float64) + (1 - w) * arr.astype(np.float64)
        return np.clip(blend, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# One-call pipeline (stage A -> stage B) + batch/GUI entry point
# --------------------------------------------------------------------------- #

def enhance_targets(image, methods=('unsharp', 'guided'), detect_kwargs=None,
                    unsharp_kwargs=None, guided_kwargs=None, clahe_kwargs=None,
                    return_detection=False):
    """Detect target saliency (stage A) then apply the requested stage-B
    local-enhancement method(s), each blended by the saliency map so only
    target-like regions are affected.

    methods: subset/order of {'unsharp', 'guided', 'clahe'}, applied in that
        order (each sees the previous stage's output).
    Returns the enhanced uint8 image, or (image, detection_info) if
    return_detection=True, where detection_info = {'mask', 'saliency', 'regions'}.
    """
    mask, saliency, regions = detect_targets(image, **(detect_kwargs or {}))
    out = np.asarray(image).copy()
    for m in methods:
        if m == 'unsharp':
            out = unsharp_mask(out, saliency, **(unsharp_kwargs or {}))
        elif m == 'guided':
            out = guided_detail_boost(out, saliency, **(guided_kwargs or {}))
        elif m == 'clahe':
            out = masked_clahe(out, saliency, **(clahe_kwargs or {}))
        else:
            raise ValueError(f"unknown enhancement method: {m!r}")
    if return_detection:
        return out, {'mask': mask, 'saliency': saliency, 'regions': regions}
    return out


def enhance_folder(input_dir, output_dir, methods=('unsharp', 'guided'),
                   detect_kwargs=None, unsharp_kwargs=None, guided_kwargs=None,
                   clahe_kwargs=None, log=None, recursive=False, max_items=0):
    """Run enhance_targets on every image in input_dir; saves the enhanced
    image per file to output_dir plus a summary CSV of detected target
    regions (image, cx, cy, w, h, area, mean_saliency).

    Mirrors evaluation/rectify.py::rectify_folder's conventions: fails fast
    ONCE (not per-file) if 'clahe' is requested without opencv, counts and
    reports per-file failures instead of silently swallowing them, and
    returns (csv_path, n_regions, n_processed, n_failed, failures).
    """
    import csv
    from PIL import Image
    from preprocessing.pipeline import scan_images

    if 'clahe' in methods:
        try:
            import cv2  # noqa
        except ImportError as exc:
            raise ImportError(
                "'clahe' 방법에는 opencv가 필요합니다: pip install opencv-python "
                "(또는 opencv-python-headless)") from exc

    os.makedirs(output_dir, exist_ok=True)
    files = scan_images(input_dir, recursive=recursive, shuffle=False, seed=42,
                        max_items=int(max_items or 0))
    csv_path = os.path.join(output_dir, 'target_regions.csv')
    n_regions = 0
    n_processed = 0
    failures = []
    if not files:
        if log:
            log(f'⚠️ 입력 폴더에서 이미지를 찾지 못했습니다: {input_dir}')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['image', 'cx', 'cy', 'width', 'height', 'area', 'mean_saliency'])
        return csv_path, 0, 0, 0, []

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'cx', 'cy', 'width', 'height', 'area', 'mean_saliency'])
        for i, p in enumerate(files):
            name = os.path.basename(p)
            try:
                img = np.asarray(Image.open(p).convert('RGB'))
                enhanced, info = enhance_targets(
                    img, methods=methods, detect_kwargs=detect_kwargs,
                    unsharp_kwargs=unsharp_kwargs, guided_kwargs=guided_kwargs,
                    clahe_kwargs=clahe_kwargs, return_detection=True)
                Image.fromarray(enhanced).save(os.path.join(output_dir, name))
                for r in info['regions']:
                    x, y, rw, rh = r['bbox']
                    cx, cy = r['centroid']
                    w.writerow([name, round(cx, 1), round(cy, 1), rw, rh,
                               r['area'], round(r['mean_saliency'], 3)])
                    n_regions += 1
                n_processed += 1
            except Exception as exc:
                failures.append((name, str(exc)))
                if log and len(failures) <= 5:
                    log(f'⚠️ {name} 처리 실패: {exc}')
                continue
            if log and ((i + 1) % 20 == 0 or i == len(files) - 1):
                log(f'강조 처리 중 {i+1}/{len(files)}  (성공 {n_processed}, 실패 {len(failures)}, '
                    f'검출된 표적 누적 {n_regions}개)')
    return csv_path, n_regions, n_processed, len(failures), failures[:5]
