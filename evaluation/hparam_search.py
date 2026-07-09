""" Automatic hyperparameter search for CUT training (Successive Halving).

Why not "self-optimize during a single run": GAN losses are adversarial
equilibrium values, NOT quality indicators — G_GAN/NCE going down does not mean
better images, so there is no reliable online signal to adapt lambdas against.
The reliable route is BETWEEN runs: train many SHORT candidate configurations,
rank them on held-out image-quality metrics (FID vs an EO set as primary, EPI
as the structure/hallucination guardrail), then give the survivors more budget.

Design (mirrors the resumable preprocessing order-search):
  - Search space: attention (type / insertion positions / reduction) and the
    structure/hallucination loss weights (lambda_grad / lambda_lap /
    lambda_coherence / lambda_color, reflector_boost, reflector_weighted,
    saliency_patch_sampling). Random-sampled without replacement from the
    grid, canonicalised so equivalent configs (e.g. attention positions when
    attention_type='none') deduplicate.
  - Stage 1: every trial trains briefly (stage1_epochs on stage1_images
    images), runs test.py, and is scored (FID + EPI/CC/PSNR).
  - Stage 2 (successive halving): the top_k trials CONTINUE training
    (--continue_train resume from their stage-1 checkpoint) for stage2_epochs
    more, optionally on the full dataset, then are re-scored; the winner is
    written to best_hparams.json.
  - Fully resumable: each completed (trial, stage) is appended to
    hparam_results.csv and skipped on re-run, so no trial ever trains twice.

The train/test commands are built by injected callables (the GUI passes its own
build_train_cmd/build_test_cmd), so this module stays free of gradio imports
and always constructs commands exactly the way the GUI does.
"""

import os
import csv
import json
import hashlib
import datetime
import traceback
import subprocess

import numpy as np

from preprocessing.pipeline import scan_images
from preprocessing import fid_utils as _fu
from evaluation.evaluate import compute_structure_metrics


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #

DEFAULT_SPACE = {
    # attention: what / where / how strong
    'attention_type': ['none', 'coord', 'cbam', 'eca', 'self', 'cbam_coord'],
    'attention_positions': ['enc', 'enc+res', 'enc+res+dec'],
    'attention_reduction': [8, 16],
    # structure / hallucination losses -- every lambda gets the SAME 3-point
    # grid (off / moderate / full weight) so sweeps are directly comparable
    # across loss terms, not biased by an arbitrarily different max per term.
    'lambda_grad': [0.0, 0.5, 1.0],
    'lambda_lap': [0.0, 0.5, 1.0],
    'lambda_coherence': [0.0, 0.5, 1.0],
    'lambda_color': [0.0, 0.5, 1.0],
    'reflector_boost': [3.0, 5.0],
    'reflector_weighted': [False, True],
    'saliency_patch_sampling': [False, True],
    'grad_no_blur': [False, True],
}

_POSITIONS = {
    'enc': (True, False, False),
    'enc+res': (True, True, False),
    'enc+res+dec': (True, True, True),
}


def canonicalize(overrides):
    """Collapse equivalent configurations to one canonical form so they
    deduplicate (e.g. attention positions/reduction are meaningless when
    attention_type='none'; reflector_weighted has no effect when both
    lambda_grad and lambda_lap are 0; reflector_boost is unused when nothing
    consumes the saliency map)."""
    ov = dict(overrides)
    pos = ov.pop('attention_positions', None)
    if ov.get('attention_type', 'none') == 'none':
        ov['attention_encoder'] = False
        ov['attention_resblocks'] = False
        ov['attention_decoder'] = False
        ov['attention_reduction'] = 16
    else:
        e, r, d = _POSITIONS.get(pos or 'enc+res', (True, True, False))
        ov['attention_encoder'], ov['attention_resblocks'], ov['attention_decoder'] = e, r, d
    if float(ov.get('lambda_grad', 0.0)) == 0.0 and float(ov.get('lambda_lap', 0.0)) == 0.0:
        ov['reflector_weighted'] = False
        ov['grad_no_blur'] = False   # nothing to blur/not-blur when both are off
    if not ov.get('reflector_weighted') and not ov.get('saliency_patch_sampling') \
            and float(ov.get('lambda_coherence', 0.0)) == 0.0:
        ov['reflector_boost'] = 3.0
    return ov


def trial_sig(overrides):
    return hashlib.md5(json.dumps(overrides, sort_keys=True, default=str)
                       .encode('utf-8')).hexdigest()[:10]


def sample_trials(space=None, n_trials=12, seed=42):
    """Random-sample n_trials distinct canonical configurations from the grid."""
    space = space or DEFAULT_SPACE
    rng = np.random.default_rng(seed)
    keys = sorted(space.keys())
    seen, out = set(), []
    for _ in range(int(n_trials) * 50):
        if len(out) >= int(n_trials):
            break
        raw = {k: space[k][int(rng.integers(len(space[k])))] for k in keys}
        ov = canonicalize(raw)
        sig = trial_sig(ov)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(ov)
    return out


def _fmt_overrides(ov):
    parts = []
    at = ov.get('attention_type', 'none')
    if at == 'none':
        parts.append('attn=none')
    else:
        pos = ''.join(s for s, on in zip('ERD', (ov.get('attention_encoder'),
                                                 ov.get('attention_resblocks'),
                                                 ov.get('attention_decoder'))) if on)
        parts.append(f'attn={at}/{pos}/r{ov.get("attention_reduction", 16)}')
    for k, tag in (('lambda_grad', 'grad'), ('lambda_lap', 'lap'),
                   ('lambda_coherence', 'coh'), ('lambda_color', 'col')):
        v = float(ov.get(k, 0.0))
        if v > 0:
            parts.append(f'{tag}={v:g}')
    if ov.get('reflector_weighted'):
        parts.append('rw')
    if ov.get('saliency_patch_sampling'):
        parts.append('sps')
    if ov.get('grad_no_blur'):
        parts.append('noblur')
    if ov.get('reflector_weighted') or ov.get('saliency_patch_sampling') \
            or float(ov.get('lambda_coherence', 0.0)) > 0:
        parts.append(f'b{ov.get("reflector_boost", 3.0):g}')
    return ' '.join(parts)


# --------------------------------------------------------------------------- #
# Resumable search
# --------------------------------------------------------------------------- #

CSV_COLUMNS = ['signature', 'stage', 'status', 'name', 'overrides',
               'epochs', 'fid', 'kid', 'epi', 'cc', 'psnr', 'timestamp']

METRIC_DIRECTION = {'fid': -1, 'kid': -1, 'epi': 1, 'cc': 1, 'psnr': 1}


def _load_done(path):
    done = {}
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    done[(row['signature'], row['stage'])] = row
        except Exception:
            pass
    return done


def _stream_subprocess(cmd, cwd, log, tag, holder):
    """Run cmd, yielding a log line per completed epoch; final rc in holder."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=dict(os.environ, PYTHONUNBUFFERED='1'))
    tail = []
    for raw in iter(proc.stdout.readline, ''):
        line = raw.rstrip()
        if not line:
            continue
        tail.append(line)
        del tail[:-30]
        if 'End of epoch' in line:
            yield log(f'{tag} {line.strip()}')
    proc.wait()
    holder['rc'] = proc.returncode
    holder['tail'] = tail


def hparam_search(base_cfg, build_train_cmd, build_test_cmd, out_dir,
                  space=None, n_trials=12, stage1_epochs=15, stage2_epochs=45,
                  top_k=5, stage1_images=300, stage2_images=0, num_test=100,
                  primary='fid', eo_dir=None, inception_weights=None,
                  fid_max=300, repo_root=None, seed=42):
    """Generator yielding log strings. Writes hparam_results.csv (resumable)
    and best_hparams.json under out_dir.

    base_cfg: a full GUI-style config dict (dataroot, checkpoints_dir,
    results_dir, netG, crop_size, ...); each trial overrides only the searched
    keys plus name/epoch bookkeeping. build_train_cmd(cfg) / build_test_cmd(cfg,
    num_test, epoch) construct the actual CLI commands (inject gui.py's).
    """
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, 'hparam_results.csv')
    log_path = os.path.join(out_dir, 'hparam_search.log')
    repo_root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

    trials = sample_trials(space, n_trials, seed)
    yield log(f'하이퍼파라미터 탐색 시작: 후보 {len(trials)}개 · '
              f'stage1 {stage1_epochs}ep×{stage1_images or "전체"}장 → 상위 {top_k}개 '
              f'stage2 +{stage2_epochs}ep×{stage2_images or "전체"}장 · 랭킹={primary}')

    # ---- FID setup (Inception + EO features loaded ONCE for all trials) ----
    fid_net = fid_tf = fid_dev = eo_feats = None
    fid_on = False
    if primary == 'fid' or (eo_dir and os.path.isdir(str(eo_dir))):
        if not _fu.fid_available():
            yield log('경고: torch/torchvision 없음 → FID 비활성화, EPI로 랭킹')
            primary = 'epi'
        elif not eo_dir or not os.path.isdir(str(eo_dir)):
            yield log(f'경고: EO 폴더 없음({eo_dir}) → FID 비활성화, EPI로 랭킹')
            primary = 'epi'
        else:
            try:
                import torch
                fid_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                wp = _fu.resolve_inception_weights(inception_weights)
                yield log(f'FID용 InceptionV3 로드 (device={fid_dev}, '
                          f'가중치={"로컬:" + wp if wp else "torch 캐시/다운로드"})')
                fid_net, fid_tf = _fu.load_inception(fid_dev, inception_weights)
                eo_files = scan_images(eo_dir, recursive=True, shuffle=True,
                                       seed=42, max_items=int(fid_max or 0))
                eo_feats = _fu.features_from_arrays(
                    _fu.read_rgb_arrays(eo_files), fid_net, fid_tf, fid_dev)
                fid_on = eo_feats.shape[0] >= 2
                yield log(f'EO 기준 특징 {eo_feats.shape[0]}개 준비 완료 (전체 trial에서 재사용)')
            except Exception:
                yield log('경고: FID 초기화 실패 → EPI로 랭킹\n'
                          + traceback.format_exc().splitlines()[-1])
                primary = 'epi'
    if primary == 'fid' and not fid_on:
        primary = 'epi'

    done = _load_done(results_csv)
    new_file = not os.path.exists(results_csv)
    mf = open(results_csv, 'a', newline='', encoding='utf-8')
    writer = csv.writer(mf)
    if new_file:
        writer.writerow(CSV_COLUMNS)
        mf.flush()

    def record(sig, stage, status, name, ov, epochs, m):
        row = [sig, stage, status, name, json.dumps(ov, sort_keys=True, default=str), epochs]
        for c in ('fid', 'kid', 'epi', 'cc', 'psnr'):
            v = (m or {}).get(c)
            row.append('' if v is None else round(float(v), 4))
        row.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        try:
            writer.writerow(row)
            mf.flush()
        except Exception:
            pass
        done[(sig, str(stage))] = dict(zip(CSV_COLUMNS, [str(x) for x in row]))

    def metrics_from_row(row):
        out = {}
        for c in ('fid', 'kid', 'epi', 'cc', 'psnr'):
            try:
                out[c] = float(row.get(c))
            except (TypeError, ValueError):
                out[c] = None
        return out

    def score_trial(name):
        """Evaluate a trained trial: structure metrics + FID of fake_B."""
        test_dir = os.path.join(str(base_cfg['results_dir']), name, 'test_latest', 'images')
        fake_dir = os.path.join(test_dir, 'fake_B')
        real_a_dir = os.path.join(test_dir, 'real_A')
        if not os.path.isdir(fake_dir):
            return None
        m = {}
        sm = compute_structure_metrics(real_a_dir, fake_dir)
        if sm:
            m.update({'epi': sm['epi'], 'cc': sm['cc'], 'psnr': sm['psnr']})
        if fid_on:
            try:
                fake_files = scan_images(fake_dir, recursive=False, shuffle=False,
                                         seed=42, max_items=int(fid_max or 0))
                fa = _fu.features_from_arrays(
                    _fu.read_rgb_arrays(fake_files), fid_net, fid_tf, fid_dev)
                m['fid'] = _fu.fid_from_feats(fa, eo_feats)
                m['kid'] = _fu.kid_from_feats(fa, eo_feats)
            except Exception:
                pass
        return m or None

    def sort_key(t):
        v = t[2].get(primary) if t[2] else None
        if v is None:
            return -1e18
        return METRIC_DIRECTION.get(primary, 1) * v

    def run_stage(sig, ov, stage, epochs_total, images, continue_train):
        """Train (or resume) + test one trial; returns metrics dict or None."""
        name = f'hps_{sig}'
        cfg = dict(base_cfg)
        cfg.update(ov)
        cfg.update(name=name, n_epochs=int(epochs_total), n_epochs_decay=0,
                   save_epoch_freq=int(epochs_total),
                   max_dataset_size=int(images or 0),
                   continue_train=bool(continue_train))
        holder = {}
        yield from _stream_subprocess(build_train_cmd(cfg), repo_root, log,
                                      f'[{name} s{stage}]', holder)
        if holder.get('rc', 1) != 0:
            record(sig, stage, 'failed', name, ov, epochs_total, None)
            yield log(f'[{name} s{stage}] 학습 실패 → 기록 후 건너뜀\n'
                      + '\n'.join(holder.get('tail', [])[-5:]))
            return
        holder2 = {}
        yield from _stream_subprocess(build_test_cmd(cfg, int(num_test), 'latest'),
                                      repo_root, log, f'[{name} s{stage} test]', holder2)
        m = score_trial(name) if holder2.get('rc', 1) == 0 else None
        if m is None:
            record(sig, stage, 'failed', name, ov, epochs_total, None)
            yield log(f'[{name} s{stage}] 평가 실패 → 기록 후 건너뜀\n'
                      + '\n'.join(holder2.get('tail', [])[-5:]))
        else:
            record(sig, stage, 'ok', name, ov, epochs_total, m)

    # ---- stage 1 ----
    stage1_results = []
    for i, ov in enumerate(trials):
        sig = trial_sig(ov)
        prev = done.get((sig, '1'))
        if prev is not None:
            if prev.get('status') == 'ok':
                stage1_results.append((sig, ov, metrics_from_row(prev)))
            yield log(f'[stage1 {i+1}/{len(trials)}] 이미 완료됨({prev.get("status")}) 건너뜀: {_fmt_overrides(ov)}')
            continue
        yield log(f'[stage1 {i+1}/{len(trials)}] 시작: {_fmt_overrides(ov)}')
        yield from run_stage(sig, ov, 1, stage1_epochs, stage1_images, False)
        row = done.get((sig, '1'))
        if row is not None and row.get('status') == 'ok':
            m = metrics_from_row(row)
            stage1_results.append((sig, ov, m))
            yield log(f'[stage1 {i+1}/{len(trials)}] 완료: {primary}='
                      f'{m.get(primary) if m else "n/a"}  {_fmt_overrides(ov)}')

    if not stage1_results:
        yield log('stage1에서 성공한 trial이 없습니다.')
        mf.close()
        return

    stage1_results.sort(key=sort_key, reverse=True)
    yield log(f'--- stage1 상위 {min(top_k, len(stage1_results))} ({primary} 기준) ---')
    for sig, ov, m in stage1_results[:top_k]:
        yield log(f'  {primary}={m.get(primary)}  epi={m.get("epi")}  {_fmt_overrides(ov)}')

    # ---- stage 2 (successive halving: winners get more budget) ----
    total_epochs = int(stage1_epochs) + int(stage2_epochs)
    stage2_results = []
    for j, (sig, ov, _) in enumerate(stage1_results[:top_k]):
        prev = done.get((sig, '2'))
        if prev is not None:
            if prev.get('status') == 'ok':
                stage2_results.append((sig, ov, metrics_from_row(prev)))
            yield log(f'[stage2 {j+1}/{top_k}] 이미 완료됨 건너뜀: {_fmt_overrides(ov)}')
            continue
        yield log(f'[stage2 {j+1}/{top_k}] 이어학습 +{stage2_epochs}ep: {_fmt_overrides(ov)}')
        yield from run_stage(sig, ov, 2, total_epochs, stage2_images, True)
        row = done.get((sig, '2'))
        if row is not None and row.get('status') == 'ok':
            m = metrics_from_row(row)
            stage2_results.append((sig, ov, m))
            yield log(f'[stage2 {j+1}/{top_k}] 완료: {primary}={m.get(primary)}')

    ranked = sorted(stage2_results, key=sort_key, reverse=True) or stage1_results[:1]
    best_sig, best_ov, best_m = ranked[0]
    best = {'signature': best_sig, 'overrides': best_ov, 'metrics': best_m,
            'primary': primary, 'trial_name': f'hps_{best_sig}',
            'checkpoints': os.path.join(str(base_cfg['checkpoints_dir']), f'hps_{best_sig}'),
            'stage1_epochs': int(stage1_epochs), 'stage2_epochs': int(stage2_epochs)}
    try:
        with open(os.path.join(out_dir, 'best_hparams.json'), 'w', encoding='utf-8') as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    mf.close()

    def fmt(v):
        return f'{v:.4f}' if isinstance(v, (int, float)) else 'n/a'

    yield log('=== 최적 하이퍼파라미터 ===\n'
              f'{_fmt_overrides(best_ov)}\n'
              f'{primary}={fmt(best_m.get(primary))}  fid={fmt(best_m.get("fid"))}  '
              f'epi={fmt(best_m.get("epi"))}\n'
              f'세부 설정: {json.dumps(best_ov, sort_keys=True, default=str, ensure_ascii=False)}\n'
              f'결과 CSV: {results_csv}\n저장: {os.path.join(out_dir, "best_hparams.json")}\n'
              f'→ 탭 4/5에 이 설정을 적용(🧬 버튼)한 뒤 전체 epoch으로 본 학습을 실행하세요. '
              f'(이어학습 체크 시 {best["checkpoints"]} 에서 계속할 수도 있습니다)')


def load_best(out_dir):
    """Read best_hparams.json (for the GUI apply-best button). None if absent."""
    path = os.path.join(out_dir, 'best_hparams.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None
