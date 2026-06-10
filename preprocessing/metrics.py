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

import os
import csv
import json
import datetime

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


def compute_dataset_metrics(image_dir, max_items=0, recursive=True, save_dir=None):
    """Average the per-image metrics over all images under ``image_dir``.

    Returns {'count', 'avg', 'std', 'per_image', 'saved'} or None if no images.
    If ``save_dir`` is given, a per-image CSV + summary TXT/JSON log is written to
    ``save_dir/metrics_logs/`` for later analysis.
    """
    files = scan_images(image_dir, recursive=recursive, shuffle=False,
                        seed=42, max_items=max_items)
    if not files:
        return None
    acc = {k: [] for k in METRIC_KEYS}
    per_image = []
    n = 0
    for p in files:
        try:
            g = _gray01(p)
        except Exception:
            continue
        m = image_metrics(g)
        for k in METRIC_KEYS:
            acc[k].append(m[k])
        per_image.append((os.path.basename(p), m))
        n += 1
    if n == 0:
        return None
    avg = {k: float(np.mean(acc[k])) for k in METRIC_KEYS}
    std = {k: float(np.std(acc[k])) for k in METRIC_KEYS}
    result = {'count': n, 'avg': avg, 'std': std, 'per_image': per_image, 'saved': None}
    if save_dir:
        try:
            result['saved'] = save_metrics_log(save_dir, result, image_dir)
        except Exception:
            result['saved'] = None
    return result


def save_metrics_log(save_dir, result, image_dir=''):
    """Write per-image CSV + summary TXT/JSON under save_dir/metrics_logs/.

    Returns the per-image CSV path.
    """
    log_dir = os.path.join(save_dir, 'metrics_logs')
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(log_dir, f'metrics_{stamp}.csv')
    txt_path = os.path.join(log_dir, f'metrics_{stamp}.txt')
    json_path = os.path.join(log_dir, f'metrics_{stamp}.json')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image'] + METRIC_KEYS)
        for name, m in result.get('per_image', []):
            w.writerow([name] + [round(m[k], 6) for k in METRIC_KEYS])
        # average / std rows at the end for quick reading
        w.writerow(['__AVG__'] + [round(result['avg'][k], 6) for k in METRIC_KEYS])
        w.writerow(['__STD__'] + [round(result['std'][k], 6) for k in METRIC_KEYS])

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f'image_dir: {image_dir}\n')
        f.write(format_metrics(result) + '\n')

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'count': result['count'], 'avg': result['avg'],
                       'std': result['std'], 'image_dir': image_dir},
                      f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return csv_path


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
