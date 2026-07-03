""" Deterministic shape rectification: snap candidate rigid-object regions in a
CUT output (fake_B) to straight-sided rectangles/polygons via classical CV
(contours + cv2.minAreaRect / approxPolyDP), instead of relying on the GAN to
learn exact geometry from scratch.

This is a DIFFERENT tool from coherence_loss (models/losses_extra.py):
  - coherence_loss (training-time) nudges the generator to prefer crisp,
    locally-straight edges over blob-like smearing, but does not (and cannot,
    from a loss alone) guarantee exact right angles.
  - rectify_image (this module, post-hoc / inference-time) guarantees exact
    straight edges by construction: it detects candidate object regions and
    replaces their outline with a fitted rotated rectangle or a simplified
    polygon. It trades "photorealistic, learned" for "geometrically exact".

Use this when the downstream need is geometric (footprint extraction, extent
measurement, "does this look unambiguously like a building/ship") rather than
purely photorealistic rendering. Requires opencv (``pip install opencv-python``
or ``opencv-python-headless``); degrades with a clear message if unavailable.
"""

import os

import numpy as np


def _require_cv2():
    try:
        import cv2
        return cv2
    except Exception as exc:
        raise ImportError(
            "opencv가 필요합니다: pip install opencv-python (또는 opencv-python-headless)"
        ) from exc


def detect_candidate_regions(image, min_area=16, max_area_frac=0.25, threshold=None):
    """Find candidate rigid-object regions in a grayscale/RGB uint8 image via
    adaptive thresholding + contour extraction.

    Parameters:
        image        -- HxW or HxWx3 uint8 array
        min_area     -- discard contours smaller than this many pixels (noise)
        max_area_frac -- discard contours larger than this fraction of the
                         image area (background/terrain, not a discrete object)
        threshold    -- fixed 0-255 threshold; None = Otsu's automatic method

    Returns a list of contours (each an (N,1,2) int32 array, OpenCV format).
    """
    cv2 = _require_cv2()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    gray = gray.astype(np.uint8)

    if threshold is None:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    max_area = max_area_frac * h * w
    return [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]


def rectify_regions(contours, poly_epsilon_frac=0.02):
    """For each contour, fit BOTH a minimum-area rotated rectangle and a
    simplified polygon (Douglas-Peucker). Returns a list of dicts:
        {'contour', 'rect': (center, size, angle_deg), 'rect_box' (4x2 int),
         'polygon' (Nx2 int, simplified), 'area', 'rectangularity'}
    ``rectangularity`` = contour_area / rect_area in [0,1]; close to 1 means
    the object's true outline is already close to a rectangle (high
    confidence the min-area-rect snap is a faithful geometric summary, not a
    distortion — useful to filter out non-rectangular objects, e.g. round
    tanks/silos, before rectifying).
    """
    cv2 = _require_cv2()
    out = []
    for c in contours:
        rect = cv2.minAreaRect(c)                      # ((cx,cy),(w,h),angle)
        box = cv2.boxPoints(rect).astype(np.int32)      # 4x2 corner points
        area = float(cv2.contourArea(c))
        rect_area = max(rect[1][0] * rect[1][1], 1e-6)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, poly_epsilon_frac * peri, True).reshape(-1, 2)
        out.append({
            'contour': c, 'rect': rect, 'rect_box': box, 'polygon': approx,
            'area': area, 'rectangularity': float(np.clip(area / rect_area, 0, 1)),
        })
    return out


def rectify_image(image_path_or_array, min_area=16, max_area_frac=0.25,
                  min_rectangularity=0.85, poly_epsilon_frac=0.02, threshold=None):
    """High-level entry point: detect candidate regions in an image and fit
    rectangles/polygons to them.

    Returns (overlay_rgb, regions):
        overlay_rgb -- uint8 HxWx3 image with detected rectangles (green) and
                       simplified polygons (red) drawn over the original.
        regions     -- list of dicts from `rectify_regions`, filtered to
                       rectangularity >= min_rectangularity (i.e. objects whose
                       true outline is close enough to a rectangle that
                       snapping to one is a faithful summary, not a
                       distortion — round/irregular blobs are left alone).

    ``min_rectangularity`` default is 0.85: a circle's area is exactly pi/4
    (~0.785) of its bounding square's area, so anything at or below ~0.79 will
    let round blobs (clouds, silos, storage tanks) through as "rectangular" —
    the threshold must clear that ceiling to actually discriminate shape.
    """
    cv2 = _require_cv2()
    if isinstance(image_path_or_array, str):
        from PIL import Image
        img = np.asarray(Image.open(image_path_or_array).convert('RGB'))
    else:
        img = np.asarray(image_path_or_array)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
    img = img.astype(np.uint8)

    contours = detect_candidate_regions(img, min_area=min_area,
                                        max_area_frac=max_area_frac, threshold=threshold)
    regions = rectify_regions(contours, poly_epsilon_frac=poly_epsilon_frac)
    regions = [r for r in regions if r['rectangularity'] >= min_rectangularity]

    overlay = img.copy()
    for r in regions:
        cv2.drawContours(overlay, [r['rect_box']], 0, (0, 255, 0), 1)     # fitted rectangle
        poly = r['polygon'].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(overlay, [poly], True, (255, 0, 0), 1)              # simplified polygon
    return overlay, regions


def rectify_folder(input_dir, output_dir, min_area=16, max_area_frac=0.25,
                   min_rectangularity=0.85, poly_epsilon_frac=0.02, log=None):
    """Run rectify_image on every image in input_dir; saves an overlay PNG per
    image to output_dir plus a summary CSV of detected rectangle geometries
    (image, cx, cy, w, h, angle_deg, rectangularity). Returns the CSV path.
    """
    import csv
    from PIL import Image
    from preprocessing.pipeline import scan_images

    os.makedirs(output_dir, exist_ok=True)
    files = scan_images(input_dir, recursive=False, shuffle=False, seed=42)
    csv_path = os.path.join(output_dir, 'rectangles.csv')
    n_regions = 0
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'cx', 'cy', 'width', 'height', 'angle_deg', 'rectangularity'])
        for i, p in enumerate(files):
            try:
                overlay, regions = rectify_image(
                    p, min_area=min_area, max_area_frac=max_area_frac,
                    min_rectangularity=min_rectangularity, poly_epsilon_frac=poly_epsilon_frac)
                name = os.path.basename(p)
                Image.fromarray(overlay).save(os.path.join(output_dir, name))
                for r in regions:
                    (cx, cy), (rw, rh), ang = r['rect']
                    w.writerow([name, round(cx, 1), round(cy, 1), round(rw, 1), round(rh, 1),
                               round(ang, 1), round(r['rectangularity'], 3)])
                    n_regions += 1
            except Exception:
                continue
            if log and ((i + 1) % 20 == 0 or i == len(files) - 1):
                log(f'후처리 {i+1}/{len(files)}  (검출된 사각형 누적 {n_regions}개)')
    return csv_path, n_regions
