""" Evaluate CUT model outputs: domain-gap (FID/KID vs real EO), structure
preservation (real_A vs fake_B), identity-path fidelity (real_B vs idt_B), and
no-reference output quality — logged per "experiment" so backbone/attention/
lambda sweeps are comparable over time.

See docs/EVALUATION.md for the metric choices and rationale.
"""

import os
import csv
import json
import datetime
import traceback

import numpy as np

from preprocessing.pipeline import scan_images
from preprocessing.img_metrics import to_luminance, psnr, cc, epi, ssim
from preprocessing import fid_utils as _fu
from preprocessing.metrics import compute_dataset_metrics


EVAL_CSV_COLUMNS = [
    'timestamp', 'experiment', 'checkpoint_epoch', 'n_fake', 'n_eo',
    'fid', 'kid', 'struct_epi', 'struct_cc', 'struct_psnr', 'n_struct_pairs',
    'idt_psnr', 'idt_ssim', 'n_idt_pairs',
    'quality_mean', 'quality_std', 'quality_speckle_index', 'quality_enl',
    'quality_avg_gradient', 'quality_entropy',
    'notes',
]


# --------------------------------------------------------------------------- #
# Pairing helper (match by filename stem; robust to extension differences,
# e.g. real_B saved as .png by test.py vs a source dataset that used .jpg)
# --------------------------------------------------------------------------- #

def _pair_by_stem(dir_a, dir_b, max_items=0):
    def stems(d):
        out = {}
        for p in scan_images(d, recursive=False, shuffle=False, seed=42, max_items=0):
            out[os.path.splitext(os.path.basename(p))[0]] = p
        return out
    a, b = stems(dir_a), stems(dir_b)
    common = sorted(set(a) & set(b))
    if max_items:
        common = common[:int(max_items)]
    return [(a[k], b[k]) for k in common]


# --------------------------------------------------------------------------- #
# Metric groups
# --------------------------------------------------------------------------- #

def compute_domain_metrics(fake_dir, eo_dir, max_items=500, device=None,
                           inception_weights=None, want_kid=True, log=None):
    """FID / KID of the generated set (fake_B) against a real EO reference set.
    Returns None if torch/torchvision or the EO folder are unavailable."""
    if not _fu.fid_available():
        if log:
            log('경고: FID/KID 비활성화 — torch/torchvision이 없습니다.')
        return None
    if not eo_dir or not os.path.isdir(eo_dir):
        if log:
            log(f'경고: FID/KID 비활성화 — EO 참조 폴더가 없습니다: {eo_dir}')
        return None
    fake_files = scan_images(fake_dir, recursive=False, shuffle=False, seed=42)
    eo_files = scan_images(eo_dir, recursive=True, shuffle=True, seed=42,
                           max_items=int(max_items or 0))
    if not fake_files:
        if log:
            log(f'경고: fake_B 폴더에 이미지가 없습니다: {fake_dir}')
        return None
    try:
        import torch
        dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        wp = _fu.resolve_inception_weights(inception_weights)
        if log:
            log(f'FID/KID용 InceptionV3 로드 (device={dev}, '
                f'가중치={"로컬:" + wp if wp else "torch 캐시/다운로드"})')
        net, tf = _fu.load_inception(dev, inception_weights)
        fa = _fu.features_from_arrays(
            _fu.read_rgb_arrays(fake_files[:int(max_items or 0) or None]), net, tf, dev)
        fb = _fu.features_from_arrays(_fu.read_rgb_arrays(eo_files), net, tf, dev)
        out = {'fid': _fu.fid_from_feats(fa, fb), 'n_fake': int(fa.shape[0]), 'n_eo': int(fb.shape[0])}
        if want_kid:
            out['kid'] = _fu.kid_from_feats(fa, fb)
        return out
    except Exception:
        if log:
            log('경고: FID/KID 계산 실패\n' + traceback.format_exc().splitlines()[-1])
        return None


def compute_structure_metrics(real_a_dir, fake_dir, max_items=0):
    """EPI / CC / PSNR of real_A vs fake_B, paired by filename (hallucination
    guard: are edges/structure from the SAR input preserved in the output?)."""
    pairs = _pair_by_stem(real_a_dir, fake_dir, max_items)
    if not pairs:
        return None
    from PIL import Image
    es, cs, ps = [], [], []
    for pa, pb in pairs:
        try:
            a = to_luminance(np.asarray(Image.open(pa).convert('RGB')))
            b = to_luminance(np.asarray(Image.open(pb).convert('RGB')))
            if a.shape != b.shape:
                from preprocessing.img_metrics import resize_to
                a = resize_to(a, b.shape)
            es.append(epi(a, b))
            cs.append(cc(a, b))
            ps.append(psnr(a, b))
        except Exception:
            continue
    if not es:
        return None
    return {'epi': float(np.mean(es)), 'cc': float(np.mean(cs)),
           'psnr': float(np.mean(ps)), 'n_pairs': len(es)}


def compute_identity_metrics(real_b_dir, idt_b_dir, max_items=0):
    """PSNR / SSIM of real_B vs idt_B = G(real_B) — a genuinely paired sanity
    check of how well G preserves EO content (useful for comparing backbones)."""
    pairs = _pair_by_stem(real_b_dir, idt_b_dir, max_items)
    if not pairs:
        return None
    from PIL import Image
    ps, ss = [], []
    for pa, pb in pairs:
        try:
            a = to_luminance(np.asarray(Image.open(pa).convert('RGB')))
            b = to_luminance(np.asarray(Image.open(pb).convert('RGB')))
            if a.shape != b.shape:
                from preprocessing.img_metrics import resize_to
                a = resize_to(a, b.shape)
            ps.append(psnr(a, b))
            ss.append(ssim(a, b))
        except Exception:
            continue
    if not ps:
        return None
    return {'psnr': float(np.mean(ps)), 'ssim': float(np.mean(ss)), 'n_pairs': len(ps)}


def compute_quality_metrics(fake_dir, max_items=0):
    """No-reference quality of fake_B (sharpness/contrast/entropy); reuses the
    preprocessing quality metrics (they are generic per-image statistics, not
    SAR-specific, so they read equally on generated colour images)."""
    res = compute_dataset_metrics(fake_dir, max_items=max_items)
    return res['avg'] if res else None


# --------------------------------------------------------------------------- #
# Orchestration + logging (generator -> yields log strings)
# --------------------------------------------------------------------------- #

def _eval_log_dir(results_dir, name):
    return os.path.join(str(results_dir), str(name), 'eval_logs')


def load_eval_log(results_dir, name):
    """Read back all logged evaluation rows (for a GUI comparison table)."""
    path = os.path.join(_eval_log_dir(results_dir, name), 'eval_results.csv')
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _append_row(results_dir, name, row):
    log_dir = _eval_log_dir(results_dir, name)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, 'eval_results.csv')
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(EVAL_CSV_COLUMNS)
        w.writerow([row.get(c, '') for c in EVAL_CSV_COLUMNS])
    try:
        with open(os.path.join(log_dir, f'eval_{row["timestamp"].replace(":", "").replace(" ", "_")}.json'),
                  'w', encoding='utf-8') as f:
            json.dump(row, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return path


def run_evaluation(results_dir, name, epoch='latest', experiment='', notes='',
                   eo_dir=None, checkpoints_dir=None, cfg=None,
                   compute_identity=False, real_b_dir=None, device=None,
                   inception_weights=None, fid_max=500, quality_max=0, struct_max=0,
                   fake_dir=None, real_a_dir=None):
    """One-button CUT output evaluation. By default reads
    <results_dir>/<name>/test_<epoch>/images/{fake_B,real_A[,real_B]} (as
    produced by test.py / the Web-UI inference tab); pass `fake_dir`/
    `real_a_dir` to analyse any other folder pair instead (e.g. a different
    results_dir, or trainA/trainB). Computes FID/KID vs eo_dir, structure
    metrics vs real_A, optionally generates+evaluates the identity path
    (real_B -> idt_B via the checkpoint), and no-reference quality of fake_B.
    Appends one row to <results_dir>/<name>/eval_logs/eval_results.csv
    (resumable comparison log) and yields progress strings.
    """
    test_dir = os.path.join(str(results_dir), str(name), f'test_{epoch}', 'images')
    fake_dir = fake_dir or os.path.join(test_dir, 'fake_B')
    real_a_dir = real_a_dir or os.path.join(test_dir, 'real_A')
    real_b_dir_out = os.path.join(test_dir, 'real_B')
    logs = []

    def log(msg):
        logs.append(f'[{datetime.datetime.now().strftime("%H:%M:%S")}] {msg}')
        return '\n'.join(logs[-300:])

    if not os.path.isdir(fake_dir):
        yield log(f'오류: fake_B 폴더가 없습니다: {fake_dir}\n'
                  f'먼저 "7. 추론/테스트" 로 test.py 를 실행해 결과를 생성하거나, '
                  f'"fake_B 폴더 직접 지정"에 분석할 폴더를 입력하세요.')
        return
    yield log(f'평가 대상: {fake_dir}')

    row = {'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          'experiment': experiment or name, 'checkpoint_epoch': epoch, 'notes': notes}

    # 1) domain gap: FID/KID vs EO
    dm = compute_domain_metrics(fake_dir, eo_dir, max_items=fid_max, device=device,
                                inception_weights=inception_weights, log=log)
    if dm:
        row.update({'fid': round(dm['fid'], 4) if dm.get('fid') is not None else '',
                    'kid': round(dm['kid'], 5) if dm.get('kid') is not None else '',
                    'n_fake': dm['n_fake'], 'n_eo': dm['n_eo']})
        yield log(f'FID={row["fid"]}  KID={row["kid"]}  (fake={dm["n_fake"]}, eo={dm["n_eo"]})')
    else:
        yield log('FID/KID 건너뜀 (EO 폴더 또는 torch 없음)')

    # 2) structure preservation: real_A vs fake_B
    sm = compute_structure_metrics(real_a_dir, fake_dir, max_items=struct_max)
    if sm:
        row.update({'struct_epi': round(sm['epi'], 4), 'struct_cc': round(sm['cc'], 4),
                    'struct_psnr': round(sm['psnr'], 2), 'n_struct_pairs': sm['n_pairs']})
        yield log(f'구조 보존(real_A vs fake_B): EPI={sm["epi"]:.4f} CC={sm["cc"]:.4f} '
                  f'PSNR={sm["psnr"]:.2f} ({sm["n_pairs"]}쌍)')
    else:
        yield log(f'구조 보존 지표 건너뜀 (real_A 폴더 없음: {real_a_dir})')

    # 3) identity-path fidelity: real_B vs idt_B = G(real_B)
    if compute_identity:
        rb_dir = real_b_dir or (real_b_dir_out if os.path.isdir(real_b_dir_out) else None)
        if not rb_dir or not os.path.isdir(rb_dir):
            yield log('identity 평가 건너뜀: real_B 폴더를 찾을 수 없습니다.')
        elif not checkpoints_dir or not cfg:
            yield log('identity 평가 건너뜀: checkpoints_dir/cfg 가 필요합니다.')
        else:
            try:
                from evaluation.generate import (
                    build_generator_from_cfg, load_generator_checkpoint, generate_from_folder,
                )
                import torch
                dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
                yield log(f'G(real_B) = idt_B 생성 중 (device={dev}) ...')
                net = build_generator_from_cfg(cfg)
                net, ckpt_path = load_generator_checkpoint(net, checkpoints_dir, name, epoch, dev)
                idt_dir = os.path.join(_eval_log_dir(results_dir, name), f'idt_B_{epoch}')
                saved = generate_from_folder(net, rb_dir, idt_dir,
                                             crop_size=int(cfg.get('crop_size', 256) or 256),
                                             device=dev, log=lambda m: None)
                yield log(f'idt_B {len(saved)}장 생성 완료 -> {idt_dir}')
                im = compute_identity_metrics(rb_dir, idt_dir)
                if im:
                    row.update({'idt_psnr': round(im['psnr'], 2), 'idt_ssim': round(im['ssim'], 4),
                               'n_idt_pairs': im['n_pairs']})
                    yield log(f'Identity 충실도(real_B vs idt_B): PSNR={im["psnr"]:.2f} '
                              f'SSIM={im["ssim"]:.4f} ({im["n_pairs"]}쌍)')
                else:
                    yield log('identity 평가 실패: 짝지어진 이미지가 없습니다.')
            except Exception:
                yield log('경고: identity 평가 중 예외\n' + traceback.format_exc().splitlines()[-1])

    # 4) no-reference output quality
    qm = compute_quality_metrics(fake_dir, max_items=quality_max)
    if qm:
        row.update({'quality_mean': round(qm['mean'], 4), 'quality_std': round(qm['std'], 4),
                    'quality_speckle_index': round(qm['speckle_index'], 4),
                    'quality_enl': round(qm['enl'], 2),
                    'quality_avg_gradient': round(qm['avg_gradient'], 4),
                    'quality_entropy': round(qm['entropy'], 4)})
        yield log(f'출력 품질: mean={qm["mean"]:.3f} std={qm["std"]:.3f} '
                  f'sharpness={qm["avg_gradient"]:.3f} entropy={qm["entropy"]:.3f}')

    path = _append_row(results_dir, name, row)
    yield log(f'=== 평가 완료 (실험명: {row["experiment"]}) ===\n'
              f'FID={row.get("fid","n/a")} KID={row.get("kid","n/a")} '
              f'EPI={row.get("struct_epi","n/a")} idt_SSIM={row.get("idt_ssim","n/a")}\n'
              f'로그 저장: {path}')
