""" Smoke test for the CUT output evaluation module (evaluation/). Requires torch.

Builds a real generator checkpoint (ResNet and HRNet, with attention) without a
full train.py/test.py run, then exercises the full evaluation path: identity-
path generation (G(real_B)), structure/identity metrics, FID/KID math, and the
resumable CSV logging used by the GUI comparison table.

Run from the repo root:  python tests/test_evaluation.py
"""

import os
import sys
import csv
import shutil
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from preprocessing.img_metrics import psnr, cc, epi, ssim
from preprocessing.fid_utils import fid_from_feats, kid_from_feats
from evaluation.generate import (
    build_generator_from_cfg, load_generator_checkpoint, generate_from_folder,
)
from evaluation.evaluate import (
    compute_structure_metrics, compute_identity_metrics, run_evaluation,
    load_eval_log, EVAL_CSV_COLUMNS,
)


def _make_images(folder, n=4, size=48, seed=0):
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        arr = (rng.random((size, size, 3)) * 255).astype('uint8')
        Image.fromarray(arr).save(os.path.join(folder, f'{i:03d}.png'))


def test_img_metrics():
    rng = np.random.default_rng(0)
    x = rng.random((48, 48))
    assert psnr(x, x) >= 99.0
    assert abs(cc(x, x) - 1.0) < 1e-6
    assert abs(epi(x, x) - 1.0) < 1e-6
    assert abs(ssim(x, x) - 1.0) < 1e-6
    y = np.clip(x + rng.normal(0, 0.3, x.shape), 0, 1)
    assert psnr(x, y) < psnr(x, x)
    assert ssim(x, y) < 1.0
    print('img_metrics: OK')


def test_fid_kid_math():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(60, 32))
    b = rng.normal(loc=3.0, size=(60, 32))
    assert fid_from_feats(a, a) < 1e-6
    assert fid_from_feats(a, b) > 1.0
    kid_ab = kid_from_feats(a, b, subset_size=30, num_subsets=5)
    kid_aa = kid_from_feats(a, a, subset_size=30, num_subsets=5)
    assert kid_ab > kid_aa
    print('fid/kid math: OK')


def _checkpoint_for(cfg, tmp, tag):
    """Build a generator matching cfg, save it as '<epoch>_net_G.pth' under a
    fake checkpoints_dir/name, mimicking what train.py would have produced."""
    ck_dir = os.path.join(tmp, 'checkpoints')
    name = f'exp_{tag}'
    os.makedirs(os.path.join(ck_dir, name), exist_ok=True)
    net = build_generator_from_cfg(cfg)
    torch.save(net.state_dict(), os.path.join(ck_dir, name, 'latest_net_G.pth'))
    return ck_dir, name


def test_generate_and_identity_metrics(netG, **attn_flags):
    tmp = tempfile.mkdtemp()
    try:
        cfg = dict(netG=netG, normG='instance', crop_size=48,
                   attention_type='none', attention_reduction=16,
                   attention_encoder=False, attention_resblocks=False, attention_decoder=False,
                   hrnet_branches=3, hrnet_modules=2, hrnet_blocks=1)
        cfg.update(attn_flags)
        ck_dir, name = _checkpoint_for(cfg, tmp, netG)

        real_b_dir = os.path.join(tmp, 'testB')
        _make_images(real_b_dir, n=4, size=48, seed=2)

        net = build_generator_from_cfg(cfg)
        net, path = load_generator_checkpoint(net, ck_dir, name, 'latest', 'cpu')
        assert os.path.exists(path)

        idt_dir = os.path.join(tmp, 'idt_B')
        saved = generate_from_folder(net, real_b_dir, idt_dir, crop_size=48, device='cpu')
        assert len(saved) == 4
        for p in saved:
            assert os.path.exists(p)

        im = compute_identity_metrics(real_b_dir, idt_dir)
        assert im is not None and im['n_pairs'] == 4
        assert np.isfinite(im['psnr']) and np.isfinite(im['ssim'])
        print(f'generate+identity[{netG}, {attn_flags}]: OK '
              f'psnr={im["psnr"]:.2f} ssim={im["ssim"]:.4f}')
        return tmp, cfg, ck_dir, name
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def test_structure_metrics():
    tmp = tempfile.mkdtemp()
    try:
        real_a = os.path.join(tmp, 'real_A')
        fake_b = os.path.join(tmp, 'fake_B')
        _make_images(real_a, n=3, size=32, seed=3)
        _make_images(fake_b, n=3, size=32, seed=3)   # identical -> near-perfect scores
        sm = compute_structure_metrics(real_a, fake_b)
        assert sm is not None and sm['n_pairs'] == 3
        assert sm['epi'] > 0.99 and sm['cc'] > 0.99
        print(f'structure_metrics (identical images): OK epi={sm["epi"]:.4f} cc={sm["cc"]:.4f}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_evaluation_logs_csv():
    tmp, cfg, ck_dir, name = test_generate_and_identity_metrics('resnet_9blocks')
    try:
        results_dir = os.path.join(tmp, 'results')
        test_dir = os.path.join(results_dir, name, 'test_latest', 'images')
        _make_images(os.path.join(test_dir, 'fake_B'), n=3, size=32, seed=5)
        _make_images(os.path.join(test_dir, 'real_A'), n=3, size=32, seed=5)
        real_b_dir = os.path.join(tmp, 'testB')

        last = None
        for line in run_evaluation(
                results_dir=results_dir, name=name, epoch='latest',
                experiment='unit_test', notes='pytest',
                eo_dir=None, checkpoints_dir=ck_dir, cfg=cfg,
                compute_identity=True, real_b_dir=real_b_dir, device='cpu'):
            last = line
        assert '평가 완료' in last

        rows = load_eval_log(results_dir, name)
        assert len(rows) == 1
        row = rows[0]
        assert row['experiment'] == 'unit_test'
        assert set(EVAL_CSV_COLUMNS) <= set(row.keys())
        assert row['n_struct_pairs'] == '3'
        assert row['n_idt_pairs'] == '4'
        assert row['fid'] == ''   # no EO dir given -> skipped, not crashed
        print('run_evaluation CSV logging: OK ->', row['idt_psnr'], row['idt_ssim'])

        # resumable append: a second call adds a second row, doesn't clobber the first
        for line in run_evaluation(
                results_dir=results_dir, name=name, epoch='latest',
                experiment='unit_test_2', eo_dir=None, checkpoints_dir=ck_dir,
                cfg=cfg, compute_identity=False, device='cpu'):
            pass
        rows2 = load_eval_log(results_dir, name)
        assert len(rows2) == 2
        assert rows2[0]['experiment'] == 'unit_test'
        assert rows2[1]['experiment'] == 'unit_test_2'
        print('run_evaluation append (2 experiments logged, no clobber): OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_img_metrics()
    test_fid_kid_math()
    tmp1, *_ = test_generate_and_identity_metrics('resnet_9blocks')
    shutil.rmtree(tmp1, ignore_errors=True)
    tmp2, *_ = test_generate_and_identity_metrics(
        'hrnet', attention_type='coord', attention_resblocks=True)
    shutil.rmtree(tmp2, ignore_errors=True)
    test_structure_metrics()
    test_run_evaluation_logs_csv()
    print('\nAll evaluation smoke tests passed.')


if __name__ == '__main__':
    main()
