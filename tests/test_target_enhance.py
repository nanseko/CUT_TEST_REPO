""" Smoke tests for evaluation/target_enhance.py -- stage A (CFAR + saliency
target detection) and stage B (saliency-guided local enhancement) of
docs/TARGET_ENHANCEMENT_SPEC.md.

Covers: CFAR detection (both 'sigma' and 'ca' methods) correctly flags a
bright compact target while not over-flagging noisy background; saliency
correctly peaks at local-contrast edges; morphology + connected-component +
area filtering removes lone-pixel false alarms; the enhancement primitives
(unsharp_mask/guided_filter/guided_detail_boost/masked_clahe) leave
zero-weight regions unchanged and visibly change full-weight regions;
guided_filter measurably preserves a step edge better than a plain box blur;
enhance_folder mirrors rectify_folder's conventions (fail-fast-once without
opencv when 'clahe' is requested, per-file failure counting, CSV summary);
and the GUI wrapper (gui.cut_enhance_targets) end-to-end, including its
input-folder-override / empty-methods / missing-cv2 error paths.

Run from the repo root:  python tests/test_target_enhance.py
"""

import os
import sys
import shutil
import tempfile
import builtins

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation as EV
from evaluation import target_enhance as te


def _make_synthetic(H=80, W=80, noise=0.05, seed=0):
    """Noisy background + two bright compact 5x5 'target' blobs + one lone
    hot pixel. Blobs are kept <= 5x5 so that guard=4 (used by every CFAR call
    in this file) fully covers each blob's extent -- CFAR's training annulus
    must exclude the whole target or the target leaks into its own
    background estimate (a guard ring smaller than the target under test is
    a test-setup bug, not an algorithm bug -- see cfar_mask's docstring)."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(0.3, noise, (H, W)), 0, 1)
    img[10:15, 10:15] = 0.95
    img[50:55, 40:45] = 0.9
    img[70, 70] = 1.0
    rgb = (np.stack([img, img, img], axis=-1) * 255).astype(np.uint8)
    return rgb


def test_cfar_mask_sigma_flags_targets_not_background():
    rgb = _make_synthetic()
    lum = te.to_luminance01(rgb)
    mask = te.cfar_mask(lum, guard=4, train=6, method='sigma', k_sigma=3.0)
    assert mask[12, 12] and mask[52, 42], 'both target blob centers should be flagged'
    bg_rate = mask[20:40, 20:40].mean()
    assert bg_rate < 0.1, f'background over-flagged: {bg_rate}'
    print(f'cfar_mask(sigma): OK (bg_flag_rate={bg_rate:.4f})')


def test_cfar_mask_ca_flags_targets_not_background():
    # Classical CA-CFAR's threshold factor T = N*(pfa^(-1/N) - 1) is derived
    # for exponentially-distributed clutter POWER (unbounded), so at the
    # module's low default pfa it demands more target/mean contrast than
    # this bounded-[0,1] synthetic image provides at pfa=1e-3. A looser pfa
    # is the correct choice for this test, not a code defect -- exactly why
    # the module recommends 'sigma' over 'ca' for rendered 8-bit imagery.
    rgb = _make_synthetic()
    lum = te.to_luminance01(rgb)
    mask = te.cfar_mask(lum, guard=4, train=6, method='ca', pfa=0.2)
    assert mask[12, 12] and mask[52, 42]
    bg_rate = mask[20:40, 20:40].mean()
    assert bg_rate < 0.15, f'background over-flagged: {bg_rate}'
    print(f'cfar_mask(ca): OK (bg_flag_rate={bg_rate:.4f})')


def test_cfar_mask_rejects_unknown_method():
    try:
        te.cfar_mask(np.zeros((10, 10)), method='bogus')
        assert False, 'should have raised ValueError'
    except ValueError:
        pass
    print('cfar_mask unknown method raises: OK')


def test_brightness_saliency_peaks_at_target_edges():
    rgb = _make_synthetic()
    sal = te.brightness_saliency01(rgb, window=5)
    assert sal.min() >= 0 and sal.max() <= 1.0001
    # The exact blob CENTER can read ~0 here because the 5x5 blob and the
    # 5x5 local-mean window coincide exactly (window = 100% flat blob = zero
    # local contrast at that one pixel); saliency peaks at the blob's EDGE,
    # which is the expected behavior of a local-contrast operator.
    target_peak = sal[10:15, 10:15].max()
    bg_peak = sal[20:40, 20:40].max()
    assert target_peak > bg_peak and target_peak > 0.9
    print(f'brightness_saliency01: OK (target_peak={target_peak:.3f}, bg_peak={bg_peak:.3f})')


def test_morphology_open_removes_isolated_specks():
    m = np.zeros((10, 10), dtype=bool)
    m[3:6, 3:6] = True
    m[0, 0] = True
    opened = te.binary_open(m, iterations=1)
    assert opened[4, 4] and not opened[0, 0]
    closed = te.binary_close(m, iterations=1)
    assert closed[4, 4]
    print('binary_open/close: OK')


def test_connected_components_cv2_and_fallback_agree():
    m = np.zeros((10, 10), dtype=bool)
    m[1:3, 1:3] = True
    m[7:9, 7:9] = True

    labels_cv2, n_cv2 = te._label_connected_components(m)
    assert n_cv2 == 2 and labels_cv2[1, 1] != labels_cv2[7, 7] and labels_cv2[1, 1] != 0

    orig_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == 'cv2':
            raise ImportError('forced')
        return orig_import(name, *a, **k)

    builtins.__import__ = fake_import
    try:
        labels_py, n_py = te._label_connected_components(m)
    finally:
        builtins.__import__ = orig_import
    assert n_py == 2 and labels_py[1, 1] != labels_py[7, 7]
    print(f'_label_connected_components: OK (cv2 n={n_cv2}, fallback n={n_py})')


def test_detect_targets_end_to_end_filters_lone_pixel():
    rgb = _make_synthetic()
    mask, sal, regions = te.detect_targets(
        rgb, guard=4, train=6, method='sigma', k_sigma=3.0, saliency_window=5,
        saliency_floor=0.15, min_area=9, max_area_frac=0.2, morph_iterations=1)
    assert mask.shape == rgb.shape[:2] and sal.shape == rgb.shape[:2]
    assert len(regions) >= 2, f'expected >=2 regions (the two blobs), got {len(regions)}'
    for r in regions:
        assert r['area'] >= 9
        assert not (abs(r['centroid'][0] - 70) < 2 and abs(r['centroid'][1] - 70) < 2), \
            'lone hot pixel must be filtered out by min_area'
    print(f'detect_targets: OK ({len(regions)} regions, areas={[r["area"] for r in regions]})')


def test_unsharp_mask_respects_weight_map():
    rgb = _make_synthetic()
    out_zero = te.unsharp_mask(rgb, np.zeros(rgb.shape[:2]), amount=0.8, radius=3)
    out_full = te.unsharp_mask(rgb, np.ones(rgb.shape[:2]), amount=0.8, radius=3)
    assert np.array_equal(out_zero, rgb), 'zero weight_map must leave image unchanged'
    assert not np.array_equal(out_full, rgb), 'full weight_map must change the image'
    print('unsharp_mask weight_map gating: OK')


def test_guided_filter_preserves_edges_better_than_box_blur():
    x = np.zeros((40, 40))
    x[:, 20:] = 1.0
    gf = te.guided_filter(x, radius=5, eps=1e-2)
    plain_blur = te._box_filter(x, 11)
    band = slice(15, 25)
    err_guided = np.abs(gf[:, band] - x[:, band]).mean()
    err_blur = np.abs(plain_blur[:, band] - x[:, band]).mean()
    assert err_guided < err_blur, f'guided={err_guided} should beat plain blur={err_blur}'
    print(f'guided_filter edge preservation: OK (guided_err={err_guided:.4f} < blur_err={err_blur:.4f})')


def test_guided_detail_boost_respects_weight_map():
    rgb = _make_synthetic()
    out_zero = te.guided_detail_boost(rgb, np.zeros(rgb.shape[:2]), boost=1.8, radius=5)
    out_full = te.guided_detail_boost(rgb, np.ones(rgb.shape[:2]), boost=1.8, radius=5)
    assert np.abs(out_zero.astype(int) - rgb.astype(int)).max() <= 1, \
        'zero weight (boost gain=1) must ~preserve original'
    assert not np.array_equal(out_full, rgb)
    print('guided_detail_boost weight_map gating: OK')


def test_masked_clahe_respects_weight_map():
    rgb = _make_synthetic()
    out_zero = te.masked_clahe(rgb, np.zeros(rgb.shape[:2]), clip_limit=2.0)
    out_full = te.masked_clahe(rgb, np.ones(rgb.shape[:2]), clip_limit=2.0)
    assert np.abs(out_zero.astype(int) - rgb.astype(int)).max() <= 2, \
        'zero weight_map must ~preserve original'
    assert not np.array_equal(out_full, rgb)
    print('masked_clahe weight_map gating: OK')


def test_enhance_targets_pipeline_and_bad_method():
    rgb = _make_synthetic()
    out, info = te.enhance_targets(rgb, methods=('unsharp', 'guided'), return_detection=True)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert len(info['regions']) >= 2
    try:
        te.enhance_targets(rgb, methods=('bogus',))
        assert False, 'should have raised ValueError'
    except ValueError:
        pass
    print('enhance_targets pipeline + bad-method rejection: OK')


def test_enhance_folder_processes_whole_folder():
    from PIL import Image
    tmp = tempfile.mkdtemp()
    try:
        indir = os.path.join(tmp, 'fake_B')
        outdir = os.path.join(tmp, 'enhanced')
        os.makedirs(indir)
        N = 4
        for i in range(N):
            Image.fromarray(_make_synthetic(seed=i)).save(os.path.join(indir, f'{i}.png'))
        csv_path, n_regions, n_ok, n_fail, failures = EV.enhance_folder(
            indir, outdir, methods=('unsharp', 'guided'))
        assert n_ok == N, f'expected all {N} processed, got {n_ok}'
        assert n_fail == 0, failures
        assert n_regions >= 2 * N
        assert os.path.exists(csv_path)
        saved = [f for f in os.listdir(outdir) if f.endswith('.png')]
        assert len(saved) == N
        with open(csv_path) as f:
            lines = f.readlines()
        assert lines[0].strip() == 'image,cx,cy,width,height,area,mean_saliency'
        assert len(lines) == 1 + n_regions
        print(f'enhance_folder whole-folder processing: OK ({n_ok}/{N} files, {n_regions} regions)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_enhance_folder_missing_cv2_fails_loudly_for_clahe():
    tmp = tempfile.mkdtemp()
    try:
        indir = os.path.join(tmp, 'fake_B')
        outdir = os.path.join(tmp, 'enhanced')
        os.makedirs(indir)

        orig_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == 'cv2':
                raise ModuleNotFoundError("No module named 'cv2'")
            return orig_import(name, *a, **k)

        builtins.__import__ = fake_import
        try:
            raised = False
            try:
                EV.enhance_folder(indir, outdir, methods=('clahe',))
            except ImportError as exc:
                raised = True
                assert 'opencv' in str(exc).lower()
        finally:
            builtins.__import__ = orig_import
        assert raised, 'missing cv2 + clahe must raise ImportError immediately'
        assert not os.path.exists(outdir) or not os.listdir(outdir)
        print('enhance_folder missing-cv2 fail-fast (clahe): OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_enhance_folder_partial_failures_counted():
    from PIL import Image
    tmp = tempfile.mkdtemp()
    try:
        indir = os.path.join(tmp, 'fake_B')
        outdir = os.path.join(tmp, 'enhanced')
        os.makedirs(indir)
        Image.fromarray(_make_synthetic()).save(os.path.join(indir, 'good.png'))
        with open(os.path.join(indir, 'zzz_corrupt.png'), 'wb') as f:
            f.write(b'not a real png')
        csv_path, n_regions, n_ok, n_fail, failures = EV.enhance_folder(indir, outdir)
        assert n_ok == 1 and n_fail == 1
        assert failures and failures[0][0] == 'zzz_corrupt.png'
        print(f'enhance_folder partial failure counting: OK (ok={n_ok}, fail={n_fail})')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_enhance_folder_empty_dir():
    tmp = tempfile.mkdtemp()
    try:
        indir = os.path.join(tmp, 'fake_B')
        outdir = os.path.join(tmp, 'enhanced')
        os.makedirs(indir)
        csv_path, n_regions, n_ok, n_fail, failures = EV.enhance_folder(indir, outdir)
        assert (n_ok, n_regions, n_fail) == (0, 0, 0)
        assert os.path.exists(csv_path)
        print('enhance_folder empty dir: OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gui_cut_enhance_targets_folder_override_and_errors():
    from PIL import Image
    import gui

    tmp = tempfile.mkdtemp()
    try:
        custom_dir = os.path.join(tmp, 'anywhere')
        os.makedirs(custom_dir)
        for i in range(3):
            Image.fromarray(_make_synthetic(seed=i)).save(os.path.join(custom_dir, f'{i}.png'))

        cfg = dict(gui.DEFAULTS)
        cfg.update(results_dir=os.path.join(tmp, 'results'), name='unused')
        vals = [cfg[k] for k in gui.CONFIG_KEYS]

        status, gallery = gui.cut_enhance_targets(
            'latest', ['unsharp', 'guided'], 'sigma', 4, 6, 3.0, 5, 0.15, 9, 0.2,
            0.6, 3, 1.5, 5, 2.0, custom_dir, *vals)
        assert '3장 처리' in status or '3' in status, status
        assert len(gallery) == 3
        print('gui.cut_enhance_targets with explicit folder override: OK')

        status_empty, gallery_empty = gui.cut_enhance_targets(
            'latest', [], 'sigma', 4, 6, 3.0, 5, 0.15, 9, 0.2,
            0.6, 3, 1.5, 5, 2.0, custom_dir, *vals)
        assert '방법' in status_empty and len(gallery_empty) == 0
        print('gui.cut_enhance_targets empty-methods validation: OK')

        orig_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == 'cv2':
                raise ModuleNotFoundError("No module named 'cv2'")
            return orig_import(name, *a, **k)

        builtins.__import__ = fake_import
        try:
            status2, gallery2 = gui.cut_enhance_targets(
                'latest', ['clahe'], 'sigma', 4, 6, 3.0, 5, 0.15, 9, 0.2,
                0.6, 3, 1.5, 5, 2.0, custom_dir, *vals)
        finally:
            builtins.__import__ = orig_import
        assert 'opencv' in status2.lower() and len(gallery2) == 0 and '✅' not in status2
        print('gui.cut_enhance_targets surfaces missing-cv2 clearly: OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_cfar_mask_sigma_flags_targets_not_background()
    test_cfar_mask_ca_flags_targets_not_background()
    test_cfar_mask_rejects_unknown_method()
    test_brightness_saliency_peaks_at_target_edges()
    test_morphology_open_removes_isolated_specks()
    test_connected_components_cv2_and_fallback_agree()
    test_detect_targets_end_to_end_filters_lone_pixel()
    test_unsharp_mask_respects_weight_map()
    test_guided_filter_preserves_edges_better_than_box_blur()
    test_guided_detail_boost_respects_weight_map()
    test_masked_clahe_respects_weight_map()
    test_enhance_targets_pipeline_and_bad_method()
    test_enhance_folder_processes_whole_folder()
    test_enhance_folder_missing_cv2_fails_loudly_for_clahe()
    test_enhance_folder_partial_failures_counted()
    test_enhance_folder_empty_dir()
    test_gui_cut_enhance_targets_folder_override_and_errors()
    print('\nAll target_enhance smoke tests passed.')


if __name__ == '__main__':
    main()
