""" SAR preprocessing pipeline runner.

Scans an input folder, runs the configured (ordered, toggleable) steps on each
image, saves results + previews + manifest + log, and can export a CUT-style
trainA/testA layout. Pure NumPy/Pillow. See docs/README_pipeline.md.
"""

import os
import csv
import glob
import json
import time
import random
import datetime
import traceback

import numpy as np

from preprocessing.steps import (
    STEP_REGISTRY, DEFAULT_STEP_ORDER, optical_like_v1_cdf,
    SPECKLE_METHODS, HISTOGRAM_MODES, INTENSITY_MODES,
)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #

def scan_images(input_dir, recursive=True, shuffle=False, seed=42, max_items=0):
    if not input_dir or not os.path.isdir(input_dir):
        return []
    files = []
    if recursive:
        for root, _, names in os.walk(input_dir):
            for n in names:
                if n.lower().endswith(IMAGE_EXTS) and not n.startswith('.'):
                    files.append(os.path.join(root, n))
    else:
        for n in os.listdir(input_dir):
            if n.lower().endswith(IMAGE_EXTS) and not n.startswith('.'):
                files.append(os.path.join(input_dir, n))
    files = sorted(files)
    if shuffle:
        random.Random(seed).shuffle(files)
    if max_items and int(max_items) > 0:
        files = files[:int(max_items)]
    return files


def _load_gray(path):
    from PIL import Image
    im = Image.open(path)
    return np.asarray(im).astype(np.float32)


def build_optical_reference_cdf(optical_dir, bins=1024, max_items=200, seed=42):
    """Target CDF from the luminance histogram of an optical folder."""
    files = scan_images(optical_dir, recursive=True, shuffle=True,
                        seed=seed, max_items=max_items)
    if not files:
        return None
    from PIL import Image
    hist = np.zeros(bins, dtype=np.float64)
    for p in files:
        try:
            im = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
            y = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
            h, _ = np.histogram(y.ravel(), bins=bins, range=(0.0, 1.0))
            hist += h
        except Exception:
            continue
    if hist.sum() <= 0:
        return None
    cdf = np.cumsum(hist) / hist.sum()
    x = np.linspace(0, 1, bins)
    return x, cdf


# --------------------------------------------------------------------------- #
# Config / presets
# --------------------------------------------------------------------------- #

def default_config():
    return {
        'io': {
            'input_dir': './datasets/M4-SAR/raw_sar',
            'output_dir': './datasets/M4-SAR-preprocessed',
            'recursive': True, 'shuffle': False, 'seed': 42,
            'max_items': 0, 'save_format': 'png',
        },
        'pipeline': {'steps': [
            {'name': 'validate_image', 'enabled': True,
             'params': {'drop_empty': True, 'handle_nan': 'zero'}},
            {'name': 'sar_intensity_transform', 'enabled': True,
             'params': {'mode': 'log1p', 'eps': 1e-6}},
            {'name': 'speckle_filter', 'enabled': True,
             'params': {'method': 'refined_lee', 'window_size': 7, 'enl': 'auto',
                        'damping_factor': 2.0}},
            {'name': 'outlier_clipping', 'enabled': True,
             'params': {'min_percentile': 0.2, 'max_percentile': 99.8,
                        'ignore_zero': True}},
            {'name': 'histogram_mapping', 'enabled': True,
             'params': {'mode': 'sar_only', 'bins': 1024,
                        'optical_reference_dir': None,
                        'clahe': {'enabled': False, 'clip_limit': 2.0,
                                  'tile_grid_size': [8, 8]}}},
            {'name': 'resize_or_tile', 'enabled': True,
             'params': {'mode': 'resize', 'image_size': 256}},
            {'name': 'channel_adapter', 'enabled': True,
             'params': {'output_channels': 3, 'strategy': 'repeat_gray'}},
            {'name': 'normalize_for_cut', 'enabled': True,
             'params': {'output_range': 'uint8'}},
        ]},
    }


def build_steps(step_cfgs):
    steps = []
    for sc in step_cfgs:
        name = sc['name']
        if name not in STEP_REGISTRY:
            continue
        steps.append(STEP_REGISTRY[name](enabled=sc.get('enabled', True),
                                         **(sc.get('params') or {})))
    return steps


# --------------------------------------------------------------------------- #
# Runner (generator -> yields (log_text, preview_paths))
# --------------------------------------------------------------------------- #

def run_pipeline(config, max_preview=12):
    from PIL import Image

    io = config['io']
    out_dir = io['output_dir']
    img_dir = os.path.join(out_dir, 'images')
    prev_dir = os.path.join(out_dir, 'preview')
    log_dir = os.path.join(out_dir, 'logs')
    for d in (img_dir, prev_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'preprocess_{stamp}.log')
    manifest_path = os.path.join(out_dir, 'manifest.csv')
    logs = []
    previews = []

    def log(msg):
        line = f'[{datetime.datetime.now().strftime("%H:%M:%S")}] {msg}'
        logs.append(line)
        try:
            with open(log_path, 'a') as f:
                f.write(line + '\n')
        except Exception:
            pass
        return '\n'.join(logs[-200:])

    files = scan_images(io['input_dir'], io.get('recursive', True),
                        io.get('shuffle', False), io.get('seed', 42),
                        io.get('max_items', 0))
    yield log(f'scan input_dir={io["input_dir"]} recursive={io.get("recursive", True)}'), previews
    if not files:
        yield log('오류: 입력 폴더에 이미지가 없습니다.'), previews
        return
    yield log(f'found {len(files)} image files (선택 {len(files)}개)'), previews

    steps = build_steps(config['pipeline']['steps'])
    enabled_names = [s.name for s in steps if s.enabled]
    yield log('pipeline: ' + ' -> '.join(enabled_names)), previews

    # Optical reference CDF (once) if needed
    optical_target = None
    for s in steps:
        if s.name == 'histogram_mapping' and s.enabled and \
                s.params.get('mode') == 'unpaired_optical_reference':
            ref = s.params.get('optical_reference_dir')
            yield log(f'optical reference CDF 생성 중: {ref}'), previews
            optical_target = build_optical_reference_cdf(
                ref, int(s.params.get('bins', 1024)))
            if optical_target is None:
                yield log('경고: optical reference를 찾지 못해 sar_only(optical_like_v1)로 대체'), previews

    # manifest
    mf = open(manifest_path, 'w', newline='')
    writer = csv.writer(mf)
    writer.writerow(['input_path', 'output_path', 'status', 'error',
                     'orig_h', 'orig_w', 'out_h', 'out_w', 'steps',
                     'p01', 'p50', 'p99', 'p998', 'zero_ratio', 'elapsed_sec'])

    ok, fail, skip = 0, 0, 0
    save_fmt = io.get('save_format', 'png')
    for i, path in enumerate(files):
        t0 = time.time()
        ctx = {'input_path': path, 'optical_target': optical_target,
               'stats': {}, 'skip': False}
        try:
            raw = _load_gray(path)
            oh, ow = raw.shape[:2]
            img = raw
            for s in steps:
                if not s.enabled:
                    continue
                img, ctx = s.apply(img, ctx)
                if ctx.get('skip'):
                    break
            name = os.path.splitext(os.path.basename(path))[0]
            if ctx.get('skip'):
                skip += 1
                writer.writerow([path, '', 'skipped', ctx.get('skip_reason', ''),
                                 oh, ow, '', '', '|'.join(enabled_names),
                                 '', '', '', '', '', round(time.time() - t0, 3)])
                continue
            arr = img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, -1)
            out_path = os.path.join(img_dir, f'{name}.{save_fmt}')
            Image.fromarray(arr).save(out_path)

            # stats
            g = arr[..., 0].astype(np.float32) / 255.0
            p01, p50, p99, p998 = np.percentile(g, [1, 50, 99, 99.8])
            zero_ratio = float((g <= 1e-4).mean())
            writer.writerow([path, out_path, 'success', '', oh, ow,
                             arr.shape[0], arr.shape[1], '|'.join(enabled_names),
                             round(float(p01), 3), round(float(p50), 3),
                             round(float(p99), 3), round(float(p998), 3),
                             round(zero_ratio, 4), round(time.time() - t0, 3)])
            ok += 1

            # preview (before/after) for first N
            if len(previews) < max_preview:
                before = (np.clip(_safe_gray01(raw), 0, 1) * 255).astype(np.uint8)
                bH = arr.shape[0]
                before = np.asarray(Image.fromarray(before).resize((bH, bH)))
                pair = np.concatenate([np.stack([before] * 3, -1), arr], axis=1)
                pv = os.path.join(prev_dir, f'{name}_ba.png')
                Image.fromarray(pair).save(pv)
                previews.append(pv)
        except Exception:
            fail += 1
            writer.writerow([path, '', 'failed', traceback.format_exc().splitlines()[-1],
                             '', '', '', '', '|'.join(enabled_names), '', '', '', '', '',
                             round(time.time() - t0, 3)])

        if (i + 1) % 5 == 0 or i == 0 or i == len(files) - 1:
            yield log(f'처리 {i+1}/{len(files)}  현재: {os.path.basename(path)}  '
                      f'(성공 {ok} / 건너뜀 {skip} / 실패 {fail})'), previews[-max_preview:]

    mf.close()
    # resolved config
    try:
        with open(os.path.join(out_dir, 'pipeline_config.resolved.json'), 'w') as f:
            json.dump(config, f, indent=2, default=str)
    except Exception:
        pass
    yield log(f'완료: 성공 {ok}, 건너뜀 {skip}, 실패 {fail} -> {img_dir}\n'
              f'manifest: {manifest_path}'), previews[-max_preview:]


def _safe_gray01(a):
    a = np.asarray(a).astype(np.float32)
    if a.ndim == 3:
        a = a.mean(-1)
    mx = float(a.max()) if a.size else 1.0
    return a / mx if mx > 1.0 else a


def preprocess_single(config, path):
    """Run the pipeline on one image; return (before_rgb, after_rgb) uint8."""
    from PIL import Image
    steps = build_steps(config['pipeline']['steps'])
    optical_target = None
    for s in steps:
        if s.name == 'histogram_mapping' and s.enabled and \
                s.params.get('mode') == 'unpaired_optical_reference':
            optical_target = build_optical_reference_cdf(
                s.params.get('optical_reference_dir'), int(s.params.get('bins', 1024)))
    raw = _load_gray(path)
    ctx = {'input_path': path, 'optical_target': optical_target, 'stats': {}, 'skip': False}
    img = raw
    for s in steps:
        if not s.enabled:
            continue
        img, ctx = s.apply(img, ctx)
        if ctx.get('skip'):
            break
    arr = img if getattr(img, 'dtype', None) == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    before = (np.clip(_safe_gray01(raw), 0, 1) * 255).astype(np.uint8)
    before = np.stack([before] * 3, -1)
    return before, arr


# --------------------------------------------------------------------------- #
# CUT layout export
# --------------------------------------------------------------------------- #

def export_cut_layout(sar_dir, out_dir, optical_dir=None, test_ratio=0.1,
                      link_mode='symlink', seed=42):
    """Build datasets/M4-SAR-cut/{trainA,testA[,trainB,testB]} from results."""
    import shutil
    sar_files = scan_images(sar_dir, recursive=True)
    if not sar_files:
        return 'export 실패: 전처리된 SAR 이미지가 없습니다.'
    rnd = random.Random(seed)
    rnd.shuffle(sar_files)
    k = int(len(sar_files) * test_ratio)
    splits = {'testA': sar_files[:k], 'trainA': sar_files[k:]}

    if optical_dir:
        opt = scan_images(optical_dir, recursive=True)
        rnd.shuffle(opt)
        ko = int(len(opt) * test_ratio)
        splits['testB'] = opt[:ko]
        splits['trainB'] = opt[ko:]

    counts = {}
    for sub, files in splits.items():
        d = os.path.join(out_dir, sub)
        os.makedirs(d, exist_ok=True)
        n = 0
        for idx, src in enumerate(files):
            dst = os.path.join(d, f'{idx:06d}_{os.path.basename(src)}')
            try:
                if os.path.exists(dst):
                    pass
                elif link_mode == 'symlink':
                    os.symlink(os.path.abspath(src), dst)
                else:
                    shutil.copy2(src, dst)
                n += 1
            except OSError:
                try:
                    shutil.copy2(src, dst); n += 1
                except Exception:
                    pass
        counts[sub] = n
    return f'CUT layout export 완료 → {out_dir}\n' + \
           '\n'.join(f'  {k}: {v}장' for k, v in counts.items())
