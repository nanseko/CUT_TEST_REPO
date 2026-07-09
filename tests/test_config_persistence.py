""" Smoke test for GUI config persistence (gui.py):

1. gui_config.json path is an ABSOLUTE path anchored to gui.py's own location,
   not the process's current working directory (the bug behind "설정이 서버
   재시작마다 초기화됨": launching from a different cwd used to resolve
   './gui_config.json' to a different file every time).
2. Auto-save round-trip (do_save -> load_config), the handler every widget's
   .change() now fires.
3. Per-checkpoint hyperparameter snapshot: saved at the start of training into
   checkpoints_dir/<name>/gui_train_config.json, restorable into every Tab 1-5
   widget later (even after other checkpoints were trained in between), with
   an architecture-mismatch guard so netG/attention/hrnet drift is caught
   before a --continue_train weight-load failure.

Requires gradio + torch (full GUI import + a couple of tiny real train.py
runs). Run from the repo root:  python tests/test_config_persistence.py
"""

import os
import re
import sys
import json
import subprocess
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gui


def test_config_path_is_absolute_and_cwd_independent():
    assert os.path.isabs(gui.DEFAULT_CONFIG_PATH), gui.DEFAULT_CONFIG_PATH
    assert gui.DEFAULT_CONFIG_PATH == os.path.join(gui.REPO_ROOT, 'gui_config.json')

    # launched (imported) from a completely different working directory ->
    # must resolve to the SAME absolute path, not a CWD-relative one
    other_dir = tempfile.mkdtemp()
    r = subprocess.run(
        [sys.executable, '-c',
         f"import sys; sys.path.insert(0, {ROOT!r}); import gui; print(gui.DEFAULT_CONFIG_PATH)"],
        cwd=other_dir, capture_output=True, text=True)
    printed = r.stdout.strip()
    assert printed == gui.DEFAULT_CONFIG_PATH, (printed, gui.DEFAULT_CONFIG_PATH, r.stderr)
    print('config path absolute + CWD-independent: OK')


def test_pp_config_path_is_absolute_too():
    # the preprocessing tab (Tab 2) has its OWN config file with the exact
    # same relative-path bug that used to affect gui_config.json
    assert os.path.isabs(gui.PP_CONFIG_PATH), gui.PP_CONFIG_PATH
    assert gui.PP_CONFIG_PATH == os.path.join(gui.REPO_ROOT, 'preproc_config.json')
    print('PP_CONFIG_PATH (Tab 2) absolute: OK')


def test_per_tab_folder_fields_are_config_keys():
    # Tab 8 (평가)/Tab 9 (후처리) data-folder fields must be real CONFIG_KEYS
    # widgets so the same auto-save mechanism persists them across restarts
    # (previously these were plain local Textboxes, outside CONFIG_KEYS, and
    # were lost on every restart regardless of the path-persistence fix).
    for k in ('eval_eo_dir', 'eval_real_b_dir', 'eval_fake_dir', 'eval_real_a_dir',
             'rectify_input_dir'):
        assert k in gui.CONFIG_KEYS, f'{k} missing from CONFIG_KEYS'
        assert k in gui.DEFAULTS, f'{k} missing from DEFAULTS'
    print('per-tab folder fields are persisted CONFIG_KEYS: OK')


def test_build_ui_and_config_keys_consistency():
    demo = gui.build_ui()
    assert demo is not None
    src = open(os.path.join(ROOT, 'gui.py'), encoding='utf-8').read()
    comp_keys = set(re.findall(r"comp\['([^']+)'\]\s*=", src))
    assert set(gui.CONFIG_KEYS) == comp_keys, (
        set(gui.CONFIG_KEYS) ^ comp_keys)
    print(f'build_ui + CONFIG_KEYS<->comp consistency ({len(gui.CONFIG_KEYS)} keys): OK')


def test_autosave_roundtrip():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'cfg.json')
    cfg = dict(gui.DEFAULTS)
    cfg.update(dataroot='/my/sar/data', name='myexp', lambda_grad=1.0)
    vals = [cfg[k] for k in gui.CONFIG_KEYS]
    msg = gui.do_save(path, *vals)
    assert os.path.exists(path) and '저장됨' in msg
    reloaded = gui.load_config(path)
    assert reloaded['dataroot'] == '/my/sar/data'
    assert reloaded['name'] == 'myexp'
    assert reloaded['lambda_grad'] == 1.0
    print('autosave round-trip (do_save -> load_config): OK')


def test_checkpoint_config_snapshot_roundtrip():
    tmp = tempfile.mkdtemp()
    ck_dir = os.path.join(tmp, 'checkpoints')
    cfg = dict(gui.DEFAULTS)
    cfg.update(name='expA', checkpoints_dir=ck_dir, netG='hrnet', attention_type='coord',
              attention_encoder=True, hrnet_branches=4)
    path = gui.save_checkpoint_config(cfg)
    assert path and os.path.exists(path)

    loaded = gui.load_checkpoint_config(ck_dir, 'expA')
    assert loaded['netG'] == 'hrnet' and loaded['attention_type'] == 'coord'
    assert '_saved_at' in loaded and '_build' in loaded
    assert gui.load_checkpoint_config(ck_dir, 'no_such_exp') is None
    print('checkpoint config snapshot save/load round-trip: OK')

    # legacy checkpoint (weights only, no json from before this feature existed)
    os.makedirs(os.path.join(ck_dir, 'expB'), exist_ok=True)
    open(os.path.join(ck_dir, 'expB', 'latest_net_G.pth'), 'w').close()
    names = gui.list_checkpoint_experiments(ck_dir)
    assert 'expA' in names
    assert any('expB' in n and '설정 로그 없음' in n for n in names)
    print('list_checkpoint_experiments (with-config + legacy): OK')

    upd = gui.refresh_checkpoint_dropdown(ck_dir)
    assert upd['choices'] and 'expA' in upd['choices']
    print('refresh_checkpoint_dropdown: OK')


def test_arch_mismatch_warnings():
    old = dict(netG='hrnet', normG='instance', attention_type='coord', attention_reduction=16,
              attention_encoder=True, attention_resblocks=False, attention_decoder=False,
              no_antialias=False, no_antialias_up=False,
              hrnet_branches=3, hrnet_modules=3, hrnet_blocks=2)
    same = dict(old)
    changed = dict(old, netG='resnet_9blocks', attention_reduction=8)

    assert gui.arch_mismatch_warnings(old, same) == []
    assert gui.arch_mismatch_warnings(None, changed) == []   # nothing to compare against

    warns = gui.arch_mismatch_warnings(old, changed)
    assert any(w.startswith('netG:') for w in warns)
    assert any(w.startswith('attention_reduction:') for w in warns)
    assert not any(w.startswith('normG:') for w in warns)   # unchanged key must not appear
    print(f'arch_mismatch_warnings: OK ({len(warns)} mismatches detected correctly)')


def test_cfg_apply_checkpoint():
    tmp = tempfile.mkdtemp()
    ck_dir = os.path.join(tmp, 'checkpoints')
    cfg = dict(gui.DEFAULTS)
    cfg.update(name='expA', checkpoints_dir=ck_dir, netG='hrnet', continue_train=False)
    gui.save_checkpoint_config(cfg)

    out = gui.cfg_apply_checkpoint(ck_dir, 'expA')
    msg, updates = out[0], out[1:]
    assert '불러왔습니다' in msg
    assert len(updates) == len(gui.CONFIG_KEYS)
    idx_netG = gui.CONFIG_KEYS.index('netG')
    idx_continue = gui.CONFIG_KEYS.index('continue_train')
    assert updates[idx_netG].get('value') == 'hrnet'
    # continue_train is an action flag, not a durable setting -> must NOT be
    # forced back to the (usually False) value captured in the snapshot
    assert 'value' not in updates[idx_continue]
    print('cfg_apply_checkpoint: restores architecture, preserves continue_train: OK')

    # legacy checkpoint (no snapshot) -> graceful message, no crash, no-op updates
    os.makedirs(os.path.join(ck_dir, 'expB'), exist_ok=True)
    open(os.path.join(ck_dir, 'expB', 'latest_net_G.pth'), 'w').close()
    out2 = gui.cfg_apply_checkpoint(ck_dir, 'expB  (설정 로그 없음)')
    assert '설정 로그가 없습니다' in out2[0]
    assert all('value' not in u for u in out2[1:])
    print('cfg_apply_checkpoint on legacy (no-json) checkpoint: graceful: OK')

    # no selection at all
    out3 = gui.cfg_apply_checkpoint(ck_dir, None)
    assert '먼저 선택' in out3[0]
    print('cfg_apply_checkpoint with no selection: graceful: OK')


def _make_dataset(root, n=3, size=64):
    for sub in ('trainA', 'trainB', 'testA', 'testB'):
        d = os.path.join(root, sub)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            Image.fromarray((np.random.rand(size, size, 3) * 255).astype('uint8')) \
                .save(os.path.join(d, f'{i}.png'))


def test_start_training_snapshot_and_mismatch_guard():
    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, 'data')
    _make_dataset(root)
    ck_dir = os.path.join(tmp, 'ck')
    cfg_path = os.path.join(tmp, 'cfg.json')
    base = dict(gui.DEFAULTS)
    base.update(dataroot=root, checkpoints_dir=ck_dir, gpu_ids='-1',
               batch_size=1, load_size=64, crop_size=64, num_threads=0,
               n_epochs=1, n_epochs_decay=0, save_epoch_freq=1)

    # fresh run -> snapshot written, no architecture check performed
    cfg1 = dict(base, name='expX', netG='resnet_9blocks')
    for _ in gui.start_training(cfg_path, 20, 20, *(cfg1[k] for k in gui.CONFIG_KEYS)):
        pass
    snap = os.path.join(ck_dir, 'expX', 'gui_train_config.json')
    assert os.path.exists(snap)
    log1 = gui.STATE.snapshot()['logs']
    assert '설정 스냅샷 저장' in log1
    assert '아키텍처' not in log1
    print('fresh start: snapshot written, no arch-check: OK')

    # continue_train=True with a DIFFERENT netG -> loud mismatch warning
    cfg2 = dict(base, name='expX', netG='hrnet', continue_train=True)
    for _ in gui.start_training(cfg_path, 20, 20, *(cfg2[k] for k in gui.CONFIG_KEYS)):
        pass
    log2 = gui.STATE.snapshot()['logs']
    assert '아키텍처 설정이 이 체크포인트를 학습할 때와 다릅니다' in log2
    assert "netG: 체크포인트='resnet_9blocks' vs 현재 설정='hrnet'" in log2
    print('continue_train with mismatched arch: warned: OK')

    # separate, untouched experiment: continue_train=True with MATCHING arch -> confirms match
    cfg3 = dict(base, name='expMatch', netG='resnet_9blocks')
    for _ in gui.start_training(cfg_path, 20, 20, *(cfg3[k] for k in gui.CONFIG_KEYS)):
        pass
    cfg4 = dict(cfg3, continue_train=True)
    for _ in gui.start_training(cfg_path, 20, 20, *(cfg4[k] for k in gui.CONFIG_KEYS)):
        pass
    log4 = gui.STATE.snapshot()['logs']
    assert '아키텍처 설정이 이전 체크포인트와 일치합니다' in log4
    print('continue_train with matching arch: confirmed: OK')


def main():
    test_config_path_is_absolute_and_cwd_independent()
    test_pp_config_path_is_absolute_too()
    test_per_tab_folder_fields_are_config_keys()
    test_build_ui_and_config_keys_consistency()
    test_autosave_roundtrip()
    test_checkpoint_config_snapshot_roundtrip()
    test_arch_mismatch_warnings()
    test_cfg_apply_checkpoint()
    test_start_training_snapshot_and_mismatch_guard()
    print('\nAll config-persistence smoke tests passed.')


if __name__ == '__main__':
    main()
