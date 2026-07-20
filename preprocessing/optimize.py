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
from preprocessing.img_metrics import (
    resize_to as _resize_to, psnr, cc, epi, composite_score, EPS,
)
from preprocessing import fid_utils as _fu

# re-exported for backward compatibility (gui.py / preprocessing/__init__.py
# import these names directly from preprocessing.optimize)
INCEPTION_FILENAME = _fu.INCEPTION_FILENAME
fid_available = _fu.fid_available
resolve_inception_weights = _fu.resolve_inception_weights
fid_from_feats = _fu.fid_from_feats
_read_rgb_arrays = _fu.read_rgb_arrays
_features_from_arrays = _fu.features_from_arrays


def _load_inception(device='cpu', weights_path=None):
    return _fu.load_inception(device, weights_path)


# order-sensitive steps that get permuted
PERMUTE_STEPS = ['sar_intensity_transform', 'speckle_filter',
                 'outlier_clipping', 'histogram_mapping']


METRIC_DIRECTION = {'psnr': 1, 'cc': 1, 'epi': 1, 'enl': 1,
                    'speckle_index': -1, 'fid': -1, 'composite': 1}


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
# Optional FID (preprocessed SAR set vs Optical set) — FID/KID mechanics live in
# preprocessing/fid_utils.py (shared with evaluation/); this just supplies the
# per-candidate preprocessed-output image stream.
# --------------------------------------------------------------------------- #

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
                    _read_rgb_arrays(eo_files), fid_net, fid_tf, fid_dev)
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


# =========================================================================== #
# STAGE 2: per-step PARAMETER optimization (coordinate descent)
#
# Runs AFTER the order search: takes a fixed order + speckle method (normally
# read from stage-1's best_pipeline.json) and tunes each order-sensitive step's
# numeric parameters one step at a time. Coordinate descent = for each step,
# grid-sweep its params while holding every other step fixed at the current
# best, keep the winner, move to the next step. Candidate count grows as the
# SUM of per-step grids (cheap), not their product.
#
# Reuses the stage-1 infra wholesale: build_steps / the same in-memory
# evaluate loop / the same PSNR/CC/EPI/ENL/SI(+optional FID) metrics / the same
# resumable append-only CSV. Structural steps (resize/channel/normalize/
# validate) are auto-excluded because their PARAM_SPACE is empty.
# =========================================================================== #

from preprocessing.steps import STEP_REGISTRY


def _set_dotted(d, dotted_key, value):
    """Set d['a']['b'] = value for dotted_key 'a.b' (creates nested dicts)."""
    keys = dotted_key.split('.')
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):        # existing scalar where we need a dict
            cur = {}
    cur[keys[-1]] = value


def tunable_steps_in_order(order):
    """The order-sensitive steps (in pipeline order) that expose a non-empty
    PARAM_SPACE -- i.e. the ones the parameter optimizer will tune. Structural
    steps never appear here (empty PARAM_SPACE)."""
    out = []
    for name in order:
        space = dict(getattr(STEP_REGISTRY.get(name), 'PARAM_SPACE', {}) or {})
        if space:
            out.append((name, space))
    return out


def _relevant_param_space(step_name, space, speckle_method):
    """Prune candidate values that don't apply to the current fixed config, so
    the sweep doesn't waste evaluations on no-op combinations."""
    space = {k: list(v) for k, v in space.items()}
    if step_name == 'speckle_filter':
        # damping_factor only affects frost; for other methods it's a dead knob
        if speckle_method != 'frost':
            space.pop('damping_factor', None)
    if step_name == 'histogram_mapping':
        # clahe needs cv2; if it's unavailable, 'enabled=True' is a silent no-op
        # (the step logs a warning and returns unchanged) -> don't sweep it
        try:
            import cv2  # noqa
        except Exception:
            space.pop('clahe.enabled', None)
            space.pop('clahe.clip_limit', None)
    return space


def build_param_pipeline_steps(order, speckle_method, param_overrides,
                               image_size=256, hist_mode='sar_only',
                               optical_reference_dir=None, reference_cdf_path=None):
    """Same as build_pipeline_steps but applies param_overrides =
    {step_name: {dotted_param: value}} on top of the default step configs."""
    steps = build_pipeline_steps(order, speckle_method, image_size, hist_mode,
                                 optical_reference_dir, reference_cdf_path)
    for cfg in steps:
        ov = (param_overrides or {}).get(cfg['name'])
        if ov:
            for dotted, val in ov.items():
                _set_dotted(cfg['params'], dotted, val)
    return steps


def evaluate_param_pipeline(order, speckle_method, param_overrides, files,
                            image_size=256, hist_mode='sar_only', optical_target=None):
    """Like evaluate_pipeline but with per-step parameter overrides applied.
    Returns the same metrics dict (psnr/cc/epi/enl/speckle_index/composite/count)."""
    steps_cfg = build_param_pipeline_steps(order, speckle_method, param_overrides,
                                           image_size, hist_mode)
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
            a = np.asarray(img).astype(np.float64)
            if a.ndim == 3:
                a = a.mean(-1)
            proc = np.clip(a / 255.0 if a.max() > 1 else a, 0, 1)
            ref = _resize_to(_safe_gray01(raw), proc.shape)
            ps.append(psnr(ref, proc))
            cs.append(cc(ref, proc))
            es.append(epi(ref, proc))
            mu, sd = float(proc.mean()), float(proc.std())
            es_si.append(sd / (mu + EPS))
            es_enl.append((mu * mu) / (sd * sd + EPS))
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


def _param_signature(order, speckle_method, param_overrides, n_images):
    payload = json.dumps(param_overrides, sort_keys=True, default=str)
    return f"params|{'>'.join(order)}|speckle={speckle_method}|{payload}|n={n_images}"


def load_best_pipeline(path):
    """Read a stage-1 best_pipeline.json -> (order, speckle) or (None, None)."""
    try:
        with open(path, encoding='utf-8') as f:
            best = json.load(f)
        order = list(best.get('order') or [])
        speckle = best.get('speckle')
        if order and speckle:
            return order, speckle
    except Exception:
        pass
    return None, None


def optimize_params(sar_dir, out_dir, order=None, speckle_method=None,
                    best_pipeline_path=None, n_images=300, primary='composite',
                    hist_mode='sar_only', optical_dir=None, max_scan=0,
                    passes=1):
    """Generator yielding log strings. Coordinate-descent parameter search over
    the tunable steps of a FIXED order. Writes param_search_results.csv
    (resumable) and best_params_pipeline.json under out_dir.

    order/speckle_method: if omitted, read from best_pipeline_path (defaults to
    <out_dir>/best_pipeline.json produced by the order search) -- this is the
    "automatic hand-off" from stage 1 to stage 2.
    passes: how many full coordinate-descent sweeps over all steps (a 2nd pass
    can improve results when steps interact, at ~2x cost).
    """
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, 'param_search_results.csv')
    log_path = os.path.join(out_dir, 'param_search.log')
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

    # --- resolve the fixed order/speckle (from args or best_pipeline.json) ---
    if not order or not speckle_method:
        bp = best_pipeline_path or os.path.join(out_dir, 'best_pipeline.json')
        o2, s2 = load_best_pipeline(bp)
        order = order or o2
        speckle_method = speckle_method or s2
        if order and speckle_method:
            yield log(f'순서 자동 연결: {bp} 에서 order/speckle 로드')
    if not order or not speckle_method:
        yield log('오류: 고정할 순서(order)와 speckle 방법이 필요합니다. '
                  '먼저 "순서 자동 최적화"를 실행하거나 순서를 직접 지정하세요.')
        return
    yield log(f'고정 순서: validate -> {" -> ".join(order)} -> resize -> channel -> normalize  '
              f'· speckle={speckle_method}')

    files = scan_images(sar_dir, recursive=True, shuffle=False, seed=42,
                        max_items=int(max_scan or 0))
    if not files:
        yield log(f'오류: SAR 입력 폴더에 이미지가 없습니다: {sar_dir}')
        return
    files = files[:int(n_images)]
    yield log(f'SAR {len(files)}장으로 파라미터 탐색 (랭킹 기준={primary})')

    optical_target = None
    if hist_mode in ('unpaired_optical_reference', 'preset'):
        try:
            optical_target = (load_reference_cdf(optical_dir) if hist_mode == 'preset'
                              else build_optical_reference_cdf(optical_dir, 1024))
        except Exception:
            optical_target = None

    tunables = tunable_steps_in_order(order)
    if not tunables:
        yield log('이 순서에는 조절 가능한 파라미터를 가진 스텝이 없습니다.')
        return
    yield log('조절 대상 스텝: ' + ', '.join(f'{n}({",".join(_relevant_param_space(n, sp, speckle_method))})'
                                            for n, sp in tunables))

    # --- resumable CSV ---
    done = _load_done(results_csv)
    new_file = not os.path.exists(results_csv)
    mf = open(results_csv, 'a', newline='', encoding='utf-8')
    writer = csv.writer(mf)
    cols = ('psnr', 'cc', 'epi', 'enl', 'speckle_index', 'composite')
    if new_file:
        writer.writerow(['signature', 'step', 'param', 'value', 'overrides', 'n_images']
                        + list(cols) + ['timestamp'])
        mf.flush()

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def metric_of(m):
        if m is None:
            return None
        return m.get(primary)

    direction = METRIC_DIRECTION.get(primary, 1)

    def better(a, b):
        """Is metric a strictly better than b (None = worst)?"""
        if a is None:
            return False
        if b is None:
            return True
        return (a > b) if direction > 0 else (a < b)

    def _round4(m):
        # Round to the SAME precision the CSV stores, and use these rounded
        # values everywhere (fresh run AND resumed run). Otherwise a fresh run
        # decides the coordinate-descent path on full-precision metrics while a
        # resumed run reads back 4-decimal values from the CSV -> the strict
        # better() comparison can diverge -> a different descent path -> new
        # (uncached) trials get evaluated -> resume isn't a no-op. Rounding
        # consistently makes the descent path deterministic across runs.
        return {c: (None if (m or {}).get(c) is None else round(float(m[c]), 4)) for c in cols}

    def evaluate(overrides):
        """Evaluate an override set, using the resumable cache when possible."""
        sig = _param_signature(order, speckle_method, overrides, len(files))
        if sig in done:
            row = done[sig]
            return {c: _f(row.get(c)) for c in cols}
        m = evaluate_param_pipeline(order, speckle_method, overrides, files,
                                    hist_mode=hist_mode, optical_target=optical_target)
        d = _round4(m)
        row = [sig, '', '', '', json.dumps(overrides, sort_keys=True, default=str), len(files)]
        row += [('' if d.get(c) is None else d[c]) for c in cols]
        row += [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        writer.writerow(row)
        mf.flush()
        done[sig] = dict(d)
        return d

    # --- baseline: default params for the fixed order ---
    current = {}                       # step_name -> {dotted_param: value}
    base_m = evaluate(current)
    best_metric = metric_of(base_m)
    yield log(f'기준(기본 파라미터) {primary}={best_metric}')

    total_evals = 1
    for pass_i in range(int(max(1, passes))):
        yield log(f'=== 좌표하강 pass {pass_i+1}/{int(max(1,passes))} ===')
        improved_this_pass = False
        for step_name, full_space in tunables:
            space = _relevant_param_space(step_name, full_space, speckle_method)
            for param, values in space.items():
                best_val, best_here = None, best_metric
                cur_val = current.get(step_name, {}).get(param, '(default)')
                for val in values:
                    trial = {k: dict(v) for k, v in current.items()}
                    trial.setdefault(step_name, {})[param] = val
                    m = evaluate(trial)
                    total_evals += 1
                    mv = metric_of(m)
                    if better(mv, best_here):
                        best_here, best_val = mv, val
                if best_val is not None and better(best_here, best_metric):
                    current.setdefault(step_name, {})[param] = best_val
                    yield log(f'  {step_name}.{param}: {cur_val} -> {best_val}  '
                              f'({primary} {best_metric} -> {best_here})')
                    best_metric = best_here
                    improved_this_pass = True
                else:
                    yield log(f'  {step_name}.{param}: 개선 없음 (유지)')
        if not improved_this_pass:
            yield log('이번 pass에서 개선이 없어 조기 종료합니다.')
            break

    # --- save the tuned pipeline ---
    full_steps = build_param_pipeline_steps(order, speckle_method, current, hist_mode=hist_mode)
    best = {'order': list(order), 'speckle': speckle_method,
            'param_overrides': current, 'primary': primary,
            'metric': best_metric, 'n_images': len(files), 'total_evals': total_evals,
            'full_steps': full_steps}
    try:
        with open(os.path.join(out_dir, 'best_params_pipeline.json'), 'w', encoding='utf-8') as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    mf.close()

    ov_txt = json.dumps(current, ensure_ascii=False) if current else '(기본값이 최적)'
    yield log('=== 최적 파라미터 ===\n'
              f'{ov_txt}\n'
              f'{primary}={best_metric}  · 평가 횟수={total_evals}\n'
              f'결과 로그: {results_csv}\n'
              f'저장: {os.path.join(out_dir, "best_params_pipeline.json")}')
