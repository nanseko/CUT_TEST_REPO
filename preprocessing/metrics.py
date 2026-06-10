""" No-reference SAR preprocessing quality metrics.

Computed on the preprocessed output images and averaged over the whole output
set, so you can compare preprocessing configurations objectively.

Metrics (per image, then averaged):
  - mean            : overall brightness (전체 평균 밝기)
  - std             : global contrast / spread (대비)
  - speckle_index   : sigma/mean coefficient of variation — LOWER = less speckle
  - enl             : (mean/std)^2 Equivalent Number of Looks — HIGHER = better
                      speckle suppression on homogeneous areas
  - avg_gradient    : mean spatial gradient — HIGHER = sharper / more detail/edges
  - entropy         : Shannon entropy of the intensity histogram — information

ENL / speckle index are most meaningful on homogeneous regions; the global
(whole-image) values used here are a practical dataset-level proxy.
"""

import numpy as np

from preprocessing.pipeline import scan_images

EPS = 1e-8

# (key, label, higher_is_better or None)
METRIC_INFO = [
    ('mean', '평균 밝기 (Mean)', None),
    ('std', '대비 (Std)', None),
    ('speckle_index', 'Speckle Index (σ/μ)', False),
    ('enl', 'ENL ((μ/σ)²)', True),
    ('avg_gradient', '선명도 (Avg Gradient)', True),
    ('entropy', '정보량 (Entropy)', True),
]
METRIC_KEYS = [k for k, _, _ in METRIC_INFO]


def _gray01(path):
    from PIL import Image
    im = np.asarray(Image.open(path).convert('L')).astype(np.float64)
    return im / 255.0


def image_metrics(g):
    """Per-image metrics for a single-channel image in [0, 1]."""
    mean = float(g.mean())
    std = float(g.std())
    si = std / (mean + EPS)
    enl = (mean * mean) / (std * std + EPS)
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    ag = float((gx.mean() + gy.mean()) / 2.0)
    hist, _ = np.histogram(g, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / (hist.sum() + EPS)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    return {'mean': mean, 'std': std, 'speckle_index': si,
            'enl': enl, 'avg_gradient': ag, 'entropy': entropy}


def compute_dataset_metrics(image_dir, max_items=0, recursive=True):
    """Average the per-image metrics over all images under ``image_dir``.

    Returns {'count', 'avg': {...}, 'std': {...}} or None if no images found.
    """
    files = scan_images(image_dir, recursive=recursive, shuffle=False,
                        seed=42, max_items=max_items)
    if not files:
        return None
    acc = {k: [] for k in METRIC_KEYS}
    n = 0
    for p in files:
        try:
            g = _gray01(p)
        except Exception:
            continue
        m = image_metrics(g)
        for k in METRIC_KEYS:
            acc[k].append(m[k])
        n += 1
    if n == 0:
        return None
    avg = {k: float(np.mean(acc[k])) for k in METRIC_KEYS}
    std = {k: float(np.std(acc[k])) for k in METRIC_KEYS}
    return {'count': n, 'avg': avg, 'std': std}


def format_metrics(result):
    """Human-readable summary (Korean) for the GUI / CLI."""
    if not result:
        return '지표를 계산할 이미지가 없습니다. 출력(images) 폴더 경로를 확인하세요.'
    lines = [f'전처리 성능 지표 — 이미지 {result["count"]}장 평균', '']
    for key, label, better in METRIC_INFO:
        a = result['avg'][key]
        s = result['std'][key]
        hint = ''
        if better is True:
            hint = '  (↑ 높을수록 좋음)'
        elif better is False:
            hint = '  (↓ 낮을수록 좋음)'
        lines.append(f'- {label:<22}: {a:8.4f}  ± {s:.4f}{hint}')
    return '\n'.join(lines)
