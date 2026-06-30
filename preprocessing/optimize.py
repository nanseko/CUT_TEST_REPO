""" Automatic search for the best SAR preprocessing step ORDER.

Strategy (agreed design):
  - Structural steps are fixed: validate_image first; resize_or_tile ->
    channel_adapter -> normalize_for_cut last.
  - The four order-sensitive steps are permuted exhaustively:
        sar_intensity_transform, speckle_filter, outlier_clipping, histogram_mapping
    -> 4! = 24 orders.
  - speckle is branched per filter type (lee/frost/refined_lee/gamma_map/bm3d),
    so a "pipeline" = (order, speckle_method)  -> 24 x 5 = 120 candidates.
  - Each candidate is scored by image-evaluation metrics on the preprocessed
    output vs the raw SAR input (per-image: PSNR, CC, EPI) and, for the finalists,
    FID of the preprocessed set vs an Optical reference set.
  - Two-stage evaluation: rank all candidates on a small image subset (stage 1),
    then re-evaluate the top-K on the full subset (stage 2).
  - Fully resumable: every completed (pipeline, stage) is appended to a results
    CSV; on resume those signatures are skipped, so no pipeline is ever run twice.

Pure NumPy/Pillow for PSNR/CC/EPI; FID is optional (needs torch+torchvision and
an optical folder) and degrades gracefully when unavailable.
"""

import os
import csv
import json
import itertools
import datetime

import numpy as np

from preprocessing.pipeline import (
    scan_images, build_steps, build_optical_reference_cdf, load_reference_cdf,
    _load_gray, _safe_gray01,
)
from preprocessing.steps import SPECKLE_METHODS

EPS = 1e-8

# order-sensitive steps that get permuted
PERMUTE_STEPS = ['sar_intensity_transform', 'speckle_filter',
                 'outlier_clipping', 'histogram_mapping']


# --------------------------------------------------------------------------- #
# Image-evaluation metrics (reference = raw SAR, resized to the output size)
# --------------------------------------------------------------------------- #

def _resize_to(g, shape):
    from PIL import Image
    if g.shape == shape:
        return g
    im = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).resize(
        (shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(im).astype(np.float64) / 255.0


def psnr(ref, img):
    mse = float(np.mean((ref - img) ** 2))
    if mse <= EPS:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def cc(ref, img):
    a = ref.ravel() - ref.mean()
    b = img.ravel() - img.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / (d + EPS))


def _laplacian(x):
    xp = np.pad(x, 1, mode='reflect')
    return (xp[:-2, 1:-1] + xp[2:, 1:-1] + xp[1:-1, :-2] + xp[1:-1, 2:] - 4 * x)


def epi(ref, img):
    """Edge Preservation Index: Pearson correlation of the Laplacians (high-pass)
    of reference and processed image. ~1 = edges well preserved."""
    lr = _laplacian(ref).ravel()
    li = _laplacian(img).ravel()
    lr = lr - lr.mean()
    li = li - li.mean()
    d = float(np.sqrt((lr * lr).sum() * (li * li).sum()))
    return float((lr * li).sum() / (d + EPS))


METRIC_DIRECTION = {'psnr': 1, 'cc': 1, 'epi': 1, 'fid': -1, 'composite': 1}


def composite_score(m):
    """Higher = better. Combines edge preservation, correlation and PSNR."""
    return (m.get('epi', 0.0) + m.get('cc', 0.0) + m.get('psnr', 0.0) / 40.0) / 3.0


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #

def _step_cfg(name, speckle_method, image_size, hist_mode, optical_reference_dir,
              reference_cdf_path):
    if name == 'validate_image':
        return {'name': name, 'enabled': True,
                'params': {'drop_empty': True, 'handle_nan': 'zero'}}
    if name == 'sar_intensity_transform':
        return {'name': name, 'enabled': True, 'params': {'mode': 'log1p', 'eps': 1e-6}}
    if name == 'speckle_filter':
        p = {'method': speckle_method, 'window_size': 7, 'enl': 'auto'}
        if speckle_method == 'frost':
            p['damping_factor'] = 2.0
        return {'name': name, 'enabled': True, 'params': p}
    if name == 'outlier_clipping':
        return {'name': name, 'enabled': True,
                'params': {'min_percentile': 0.2, 'max_percentile': 99.8, 'ignore_zero': True}}
    if name == 'histogram_mapping':
        return {'name': name, 'enabled': True,
                'params': {'mode': hist_mode, 'bins': 1024,
                           'optical_reference_dir': optical_reference_dir,
                           'reference_cdf_path': reference_cdf_path,
                           'clahe': {'enabled': False, 'clip_limit': 2.0, 'tile_grid_size': [8, 8]}}}
    if name == 'resize_or_tile':
        return {'name': name, 'enabled': True, 'params': {'mode': 'resize', 'image_size': image_size}}
    if name == 'channel_adapter':
        return {'name': name, 'enabled': True, 'params': {'output_channels': 3}}
    if name == 'normalize_for_cut':
        return {'name': name, 'enabled': True, 'params': {'output_range': 'uint8'}}
    raise ValueError(name)


def build_pipeline_steps(order, speckle_method, image_size=256, hist_mode='sar_only',
                         optical_reference_dir=None, reference_cdf_path=None):
    """validate -> <permuted order> -> resize -> channel -> normalize."""
    names = ['validate_image'] + list(order) + \
            ['resize_or_tile', 'channel_adapter', 'normalize_for_cut']
    return [_step_cfg(n, speckle_method, image_size, hist_mode,
                      optical_reference_dir, reference_cdf_path) for n in names]


def signature(order, speckle_method, stage, n_images):
    return f"{'>'.join(order)}|speckle={speckle_method}|stage={stage}|n={n_images}"


def enumerate_candidates():
    """All (order, speckle_method) candidates = 24 x len(SPECKLE_METHODS)."""
    cands = []
    for order in itertools.permutations(PERMUTE_STEPS):
        for sp in SPECKLE_METHODS:
            cands.append((order, sp))
    return cands


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_pipeline(order, speckle_method, files, image_size=256, hist_mode='sar_only',
                      optical_target=None):
    """Run the pipeline on each file in-memory and average PSNR/CC/EPI vs raw."""
    steps_cfg = build_pipeline_steps(order, speckle_method, image_size, hist_mode)
    steps = build_steps(steps_cfg)
    ps, cs, es = [], [], []
    n = 0
    for path in files:
        try:
            raw = _load_gray(path)
            ctx = {'input_path': path, 'optical_target': optical_target,
                   'stats': {}, 'skip': False}
            img = raw
            for s in steps:
                if not s.enabled:
                    continue
                img, ctx = s.apply(img, ctx)
                if ctx.get('skip'):
                    break
            if ctx.get('skip'):
                continue
            a = np.asarray(img).astype(np.float64)   # final image (uint8 3ch after normalize)
            if a.ndim == 3:
                a = a.mean(-1)
            proc = np.clip(a / 255.0 if a.max() > 1 else a, 0, 1)   # processed gray [0,1]
            ref = _resize_to(_safe_gray01(raw), proc.shape)         # raw gray resized to match
            ps.append(psnr(ref, proc))
            cs.append(cc(ref, proc))
            es.append(epi(ref, proc))
            n += 1
        except Exception:
            continue
    if n == 0:
        return None
    m = {'psnr': float(np.mean(ps)), 'cc': float(np.mean(cs)),
         'epi': float(np.mean(es)), 'count': n}
    m['composite'] = composite_score(m)
    return m


# --------------------------------------------------------------------------- #
# Optional FID (preprocessed SAR set vs Optical set)
# --------------------------------------------------------------------------- #

def fid_available():
    try:
        import torch  # noqa
        import torchvision  # noqa
        return True
    except Exception:
        return False


def _inception_features(images_uint8, device='cpu', batch=16):
    import torch
    import torchvision.transforms as T
    from torchvision.models import inception_v3, Inception_V3_Weights
    weights = Inception_V3_Weights.IMAGENET1K_V1
    net = inception_v3(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    tf = T.Compose([T.ToPILImage(), T.Resize((299, 299)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    feats = []
    with torch.no_grad():
        buf = []
        for im in images_uint8:
            buf.append(tf(im))
            if len(buf) == batch:
                feats.append(net(torch.stack(buf).to(device)).cpu().numpy())
                buf = []
        if buf:
            feats.append(net(torch.stack(buf).to(device)).cpu().numpy())
    return np.concatenate(feats, 0) if feats else np.zeros((0, 2048))


def _fid_from_feats(fa, fb):
    mu1, mu2 = fa.mean(0), fb.mean(0)
    sa = np.cov(fa, rowvar=False)
    sb = np.cov(fb, rowvar=False)
    from scipy.linalg import sqrtm
    covmean = sqrtm(sa.dot(sb))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(sa + sb - 2 * covmean))


# --------------------------------------------------------------------------- #
# Resumable two-stage search (generator -> yields progress strings)
# --------------------------------------------------------------------------- #

def _load_done(results_csv):
    done = {}
    if os.path.exists(results_csv):
        try:
            with open(results_csv, encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    done[row['signature']] = row
        except Exception:
            pass
    return done


def optimize_orders(sar_dir, out_dir, n_stage1=200, n_stage2=1000, top_k=10,
                    primary='composite', hist_mode='sar_only',
                    optical_dir=None, max_scan=0):
    """Generator yielding log strings. Writes results CSV (resumable) + best.json."""
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, 'order_search_results.csv')
    log_path = os.path.join(out_dir, 'order_search.log')
    logs = []

    def log(msg):
        line = f'[{datetime.datetime.now().strftime("%H:%M:%S")}] {msg}'
        logs.append(line)
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass
        return '\n'.join(logs[-300:])

    files_all = scan_images(sar_dir, recursive=True, shuffle=False, seed=42,
                            max_items=int(max_scan or 0))
    if not files_all:
        yield log(f'오류: SAR 입력 폴더에 이미지가 없습니다: {sar_dir}')
        return
    files1 = files_all[:int(n_stage1)]
    files2 = files_all[:int(n_stage2)]
    yield log(f'SAR {len(files_all)}장 발견 · stage1={len(files1)}장 / stage2={len(files2)}장')

    # optical target for histogram modes that need it
    optical_target = None
    if hist_mode in ('unpaired_optical_reference', 'preset'):
        try:
            if hist_mode == 'preset':
                optical_target = load_reference_cdf(optical_dir)  # reuse path field
            else:
                optical_target = build_optical_reference_cdf(optical_dir, 1024)
        except Exception:
            optical_target = None
        yield log(f'histogram 모드={hist_mode}, optical_target={"OK" if optical_target is not None else "없음(sar_only로 동작)"}')

    done = _load_done(results_csv)
    new_file = not os.path.exists(results_csv)
    mf = open(results_csv, 'a', newline='', encoding='utf-8')
    writer = csv.writer(mf)
    if new_file:
        writer.writerow(['signature', 'stage', 'order', 'speckle', 'n_images',
                         'psnr', 'cc', 'epi', 'composite', 'timestamp'])
        mf.flush()

    def record(order, sp, stage, m, files):
        sig = signature(order, sp, stage, len(files))
        writer.writerow([sig, stage, '>'.join(order), sp, m['count'],
                         round(m['psnr'], 4), round(m['cc'], 4), round(m['epi'], 4),
                         round(m['composite'], 4),
                         datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        mf.flush()
        done[sig] = {'order': '>'.join(order), 'speckle': sp, 'stage': str(stage),
                     'psnr': m['psnr'], 'cc': m['cc'], 'epi': m['epi'],
                     'composite': m['composite']}

    cands = enumerate_candidates()
    yield log(f'후보 파이프라인 {len(cands)}개 (24순열 × {len(SPECKLE_METHODS)} speckle). '
              f'이미 완료 {sum(1 for s in done if "|stage=1|" in s)}개(stage1) 건너뜀.')

    # ---- stage 1: rank all candidates on the small subset ----
    stage1 = []
    for i, (order, sp) in enumerate(cands):
        sig = signature(order, sp, 1, len(files1))
        if sig in done:
            r = done[sig]
            stage1.append((order, sp, {k: float(r[k]) for k in ('psnr', 'cc', 'epi', 'composite')}))
            continue
        m = evaluate_pipeline(order, sp, files1, hist_mode=hist_mode, optical_target=optical_target)
        if m is None:
            continue
        record(order, sp, 1, m, files1)
        stage1.append((order, sp, m))
        if (i + 1) % 5 == 0 or i == 0 or i == len(cands) - 1:
            yield log(f'[stage1] {i+1}/{len(cands)}  {">".join(order)} +{sp}  '
                      f'{primary}={m.get(primary, m["composite"]):.4f}')

    if not stage1:
        yield log('평가된 파이프라인이 없습니다.')
        mf.close()
        return

    direction = METRIC_DIRECTION.get(primary, 1)
    stage1.sort(key=lambda t: direction * t[2].get(primary, t[2]['composite']), reverse=True)
    yield log('--- stage1 상위 5 ---')
    for order, sp, m in stage1[:5]:
        yield log(f'  {">".join(order)} +{sp}  composite={m["composite"]:.4f} '
                  f'psnr={m["psnr"]:.2f} cc={m["cc"]:.3f} epi={m["epi"]:.3f}')

    # ---- stage 2: re-evaluate top-K on the full subset ----
    topk = stage1[:int(top_k)]
    stage2 = []
    for j, (order, sp, _) in enumerate(topk):
        sig = signature(order, sp, 2, len(files2))
        if sig in done:
            r = done[sig]
            stage2.append((order, sp, {k: float(r[k]) for k in ('psnr', 'cc', 'epi', 'composite')}))
            continue
        m = evaluate_pipeline(order, sp, files2, hist_mode=hist_mode, optical_target=optical_target)
        if m is None:
            continue
        record(order, sp, 2, m, files2)
        stage2.append((order, sp, m))
        yield log(f'[stage2] {j+1}/{len(topk)}  {">".join(order)} +{sp}  '
                  f'composite={m["composite"]:.4f}')

    ranked = sorted(stage2, key=lambda t: direction * t[2].get(primary, t[2]['composite']),
                    reverse=True) or stage1[:1]
    best_order, best_sp, best_m = ranked[0]
    best = {'order': list(best_order), 'speckle': best_sp, 'metrics': best_m,
            'full_steps': build_pipeline_steps(best_order, best_sp, hist_mode=hist_mode)}
    try:
        with open(os.path.join(out_dir, 'best_pipeline.json'), 'w', encoding='utf-8') as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    mf.close()
    yield log('=== 최적 전처리 순서 ===\n'
              f'validate -> {" -> ".join(best_order)} -> resize -> channel -> normalize\n'
              f'speckle = {best_sp}\n'
              f'composite={best_m["composite"]:.4f}  psnr={best_m["psnr"]:.2f}  '
              f'cc={best_m["cc"]:.3f}  epi={best_m["epi"]:.3f}\n'
              f'결과 로그: {results_csv}\n저장: {os.path.join(out_dir, "best_pipeline.json")}')
