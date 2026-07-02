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
import traceback

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


METRIC_DIRECTION = {'psnr': 1, 'cc': 1, 'epi': 1, 'enl': 1,
                    'speckle_index': -1, 'fid': -1, 'composite': 1}


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
    ps, cs, es, es_si, es_enl = [], [], [], [], []
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
            mu, sd = float(proc.mean()), float(proc.std())
            es_si.append(sd / (mu + EPS))                            # speckle index
            es_enl.append((mu * mu) / (sd * sd + EPS))               # ENL
            n += 1
        except Exception:
            continue
    if n == 0:
        return None
    m = {'psnr': float(np.mean(ps)), 'cc': float(np.mean(cs)),
         'epi': float(np.mean(es)), 'speckle_index': float(np.mean(es_si)),
         'enl': float(np.mean(es_enl)), 'count': n}
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


INCEPTION_FILENAME = 'inception_v3_google-0cc3c7bd.pth'


def _default_weights_paths():
    """Local locations searched for the InceptionV3 weights (offline support)."""
    paths = []
    env = os.environ.get('INCEPTION_WEIGHTS')
    if env:
        paths.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    for base in ('.', './weights', './models', here, repo,
                 os.path.join(repo, 'weights'), os.path.join(repo, 'models')):
        paths.append(os.path.join(base, INCEPTION_FILENAME))
    return paths


def resolve_inception_weights(explicit=None):
    """Return a local weights path if found (explicit -> env -> common dirs), else None."""
    for p in ([explicit] if explicit else []) + _default_weights_paths():
        if p and os.path.exists(p):
            return p
    return None


def _load_inception(device='cpu', weights_path=None):
    import torch
    import torchvision.transforms as T
    from torchvision.models import inception_v3, Inception_V3_Weights
    wp = resolve_inception_weights(weights_path)
    if wp:
        # offline: build without downloading, then load local weights
        try:
            net = inception_v3(weights=None, init_weights=False)
        except TypeError:
            net = inception_v3(weights=None)
        sd = torch.load(wp, map_location=device)
        try:
            net.load_state_dict(sd)
        except Exception:
            net.load_state_dict(sd, strict=False)
    else:
        # online: torchvision uses its cache or downloads once
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()      # 2048-d pool features
    net.eval().to(device)
    tf = T.Compose([T.ToPILImage(), T.Resize((299, 299)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return net, tf


def _features_from_arrays(array_iter, net, tf, device='cpu', batch=16):
    import torch
    feats, buf = [], []
    with torch.no_grad():
        for im in array_iter:
            if im is None:
                continue
            if im.ndim == 2:
                im = np.stack([im] * 3, -1)
            buf.append(tf(im.astype(np.uint8)))
            if len(buf) == batch:
                feats.append(net(torch.stack(buf).to(device)).cpu().numpy())
                buf = []
        if buf:
            feats.append(net(torch.stack(buf).to(device)).cpu().numpy())
    return np.concatenate(feats, 0) if feats else np.zeros((0, 2048))


def _sqrtm_trace(s1, s2):
    """trace(sqrtm(s1 @ s2)) for symmetric PSD s1, s2 using only numpy (eigh)."""
    w, v = np.linalg.eigh(s1)
    w = np.clip(w, 0, None)
    s1_half = (v * np.sqrt(w)) @ v.T
    m = s1_half @ s2 @ s1_half
    ev = np.clip(np.linalg.eigvalsh(m), 0, None)
    return float(np.sqrt(ev).sum())


def fid_from_feats(fa, fb):
    """Fréchet Inception Distance between two feature sets (numpy-only)."""
    if fa.shape[0] < 2 or fb.shape[0] < 2:
        return float('nan')
    mu1, mu2 = fa.mean(0), fb.mean(0)
    sa = np.cov(fa, rowvar=False)
    sb = np.cov(fb, rowvar=False)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(sa) + np.trace(sb)
                 - 2.0 * _sqrtm_trace(sa, sb))


def _pipeline_output_arrays(order, speckle_method, files, hist_mode, optical_target):
    """Yield the final preprocessed uint8 3ch image for each file (for FID)."""
    steps = build_steps(build_pipeline_steps(order, speckle_method, hist_mode=hist_mode))
    for path in files:
        try:
            raw = _load_gray(path)
            ctx = {'input_path': path, 'optical_target': optical_target, 'stats': {}, 'skip': False}
            img = raw
            for s in steps:
                if not s.enabled:
                    continue
                img, ctx = s.apply(img, ctx)
                if ctx.get('skip'):
                    break
            if ctx.get('skip'):
                continue
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, -1)
            yield arr
        except Exception:
            continue


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
                    optical_dir=None, max_scan=0,
                    eo_dir=None, compute_fid=False, fid_max=500, device=None,
                    inception_weights=None):
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

    # ---- optional FID setup (EO reference features computed once) ----
    fid_net = fid_tf = fid_dev = eo_feats = None
    fid_on = False
    if compute_fid or primary == 'fid':
        if not fid_available():
            yield log('경고: FID 비활성화 — torch/torchvision이 없습니다. (랭킹은 composite로 진행)')
            if primary == 'fid':
                primary = 'composite'
        elif not eo_dir or not os.path.isdir(eo_dir):
            yield log(f'경고: FID 비활성화 — EO 폴더가 없습니다: {eo_dir}')
            if primary == 'fid':
                primary = 'composite'
        else:
            try:
                import torch
                fid_dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
                wp = resolve_inception_weights(inception_weights)
                yield log(f'FID용 InceptionV3 로드 중 (device={fid_dev}, '
                          f'가중치={"로컬:" + wp if wp else "torch 캐시/다운로드"}) ...')
                fid_net, fid_tf = _load_inception(fid_dev, inception_weights)
                eo_files = scan_images(eo_dir, recursive=True, shuffle=True, seed=42,
                                       max_items=int(fid_max or 0))
                yield log(f'EO 세트 {len(eo_files)}장으로 기준 특징 추출 중 ...')
                eo_feats = _features_from_arrays(
                    (np.asarray(__import__('PIL.Image', fromlist=['Image']).Image.open(p).convert('RGB'))
                     for p in eo_files), fid_net, fid_tf, fid_dev)
                fid_on = eo_feats.shape[0] >= 2
                yield log(f'FID 활성화: EO 특징 {eo_feats.shape[0]}개 (stage2 상위 {top_k}개에 대해 계산)')
            except Exception:
                fid_on = False
                yield log('경고: FID 초기화 실패 -> composite로 진행\n' + traceback.format_exc().splitlines()[-1])
                if primary == 'fid':
                    primary = 'composite'

    done = _load_done(results_csv)
    new_file = not os.path.exists(results_csv)
    mf = open(results_csv, 'a', newline='', encoding='utf-8')
    writer = csv.writer(mf)
    cols = ('psnr', 'cc', 'epi', 'enl', 'speckle_index', 'composite', 'fid')
    if new_file:
        writer.writerow(['signature', 'stage', 'order', 'speckle', 'n_images']
                        + list(cols) + ['timestamp'])
        mf.flush()

    def record(order, sp, stage, m, files):
        sig = signature(order, sp, stage, len(files))
        row = [sig, stage, '>'.join(order), sp, m['count']]
        row += [('' if m.get(c) is None else round(float(m[c]), 4)) for c in cols]
        row += [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        writer.writerow(row)
        mf.flush()
        d = {'order': '>'.join(order), 'speckle': sp, 'stage': str(stage)}
        for c in cols:
            d[c] = m.get(c)
        done[sig] = d

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def metrics_from(r):
        m = {c: _f(r.get(c)) for c in cols}
        if m.get('composite') is None:
            m['composite'] = composite_score({k: (m.get(k) or 0.0) for k in ('epi', 'cc', 'psnr')})
        return m

    def sort_key(metric_name):
        d = METRIC_DIRECTION.get(metric_name, 1)
        worst = -1e18  # candidates missing the metric rank last

        def key(t):
            v = t[2].get(metric_name)
            if v is None:
                return worst
            return d * v
        return key

    cands = enumerate_candidates()
    yield log(f'후보 파이프라인 {len(cands)}개 (24순열 × {len(SPECKLE_METHODS)} speckle). '
              f'이미 완료 {sum(1 for s in done if "|stage=1|" in s)}개(stage1) 건너뜀.')

    stage1_primary = primary if primary != 'fid' else 'composite'  # FID only at stage2

    # ---- stage 1: rank all candidates on the small subset ----
    stage1 = []
    for i, (order, sp) in enumerate(cands):
        sig = signature(order, sp, 1, len(files1))
        if sig in done:
            stage1.append((order, sp, metrics_from(done[sig])))
            continue
        m = evaluate_pipeline(order, sp, files1, hist_mode=hist_mode, optical_target=optical_target)
        if m is None:
            continue
        record(order, sp, 1, m, files1)
        stage1.append((order, sp, m))
        if (i + 1) % 5 == 0 or i == 0 or i == len(cands) - 1:
            yield log(f'[stage1] {i+1}/{len(cands)}  {">".join(order)} +{sp}  '
                      f'{stage1_primary}={m.get(stage1_primary, m["composite"]):.4f}')

    if not stage1:
        yield log('평가된 파이프라인이 없습니다.')
        mf.close()
        return

    stage1.sort(key=sort_key(stage1_primary), reverse=True)
    yield log(f'--- stage1 상위 5 ({stage1_primary} 기준) ---')
    for order, sp, m in stage1[:5]:
        yield log(f'  {">".join(order)} +{sp}  composite={m["composite"]:.4f} '
                  f'epi={m["epi"]:.3f} enl={m["enl"]:.2f} SI={m["speckle_index"]:.3f}')

    # ---- stage 2: re-evaluate top-K on the full subset (+ FID vs EO if enabled) ----
    topk = stage1[:int(top_k)]
    stage2 = []
    for j, (order, sp, _) in enumerate(topk):
        sig = signature(order, sp, 2, len(files2))
        if sig in done and done[sig].get('fid') not in (None, '') or (sig in done and not fid_on):
            stage2.append((order, sp, metrics_from(done[sig])))
            continue
        m = evaluate_pipeline(order, sp, files2, hist_mode=hist_mode, optical_target=optical_target)
        if m is None:
            continue
        if fid_on:
            try:
                feats = _features_from_arrays(
                    _pipeline_output_arrays(order, sp, files2[:int(fid_max)], hist_mode, optical_target),
                    fid_net, fid_tf, fid_dev)
                m['fid'] = fid_from_feats(feats, eo_feats)
            except Exception:
                m['fid'] = None
        record(order, sp, 2, m, files2)
        stage2.append((order, sp, m))
        fidtxt = f" fid={m['fid']:.2f}" if m.get('fid') is not None else ''
        yield log(f'[stage2] {j+1}/{len(topk)}  {">".join(order)} +{sp}  '
                  f'composite={m["composite"]:.4f}{fidtxt}')

    ranked = sorted(stage2, key=sort_key(primary), reverse=True) or stage1[:1]
    best_order, best_sp, best_m = ranked[0]
    best = {'order': list(best_order), 'speckle': best_sp, 'metrics': best_m,
            'full_steps': build_pipeline_steps(best_order, best_sp, hist_mode=hist_mode)}
    try:
        with open(os.path.join(out_dir, 'best_pipeline.json'), 'w', encoding='utf-8') as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    mf.close()

    def fmt(v, p='.4f'):
        return ('%' + p) % v if isinstance(v, (int, float)) else 'n/a'

    yield log('=== 최적 전처리 순서 ===\n'
              f'validate -> {" -> ".join(best_order)} -> resize -> channel -> normalize\n'
              f'speckle = {best_sp}  ·  랭킹 기준 = {primary}\n'
              f'composite={fmt(best_m.get("composite"))}  epi={fmt(best_m.get("epi"),".3f")}  '
              f'enl={fmt(best_m.get("enl"),".2f")}  SI={fmt(best_m.get("speckle_index"),".3f")}  '
              f'fid={fmt(best_m.get("fid"),".2f")}\n'
              f'결과 로그: {results_csv}\n저장: {os.path.join(out_dir, "best_pipeline.json")}')
