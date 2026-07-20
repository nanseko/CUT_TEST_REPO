""" Smoke test for preprocessing parameter optimization (coordinate descent).

Stage 2 of the preprocessing search: after the ORDER search fixes the step
order (optimize_orders -> best_pipeline.json), optimize_params tunes each
tunable step's numeric parameters one at a time (coordinate descent), holding
the order fixed. Structural steps (resize/channel/normalize/validate) are
excluded automatically via an empty PARAM_SPACE.

Covers: PARAM_SPACE declarations, dotted-key merge, tunable-step filtering,
relevance pruning (frost-only damping, cv2-gated clahe), auto-connect from
best_pipeline.json, an actual coordinate-descent run that improves the metric,
and deterministic resume (a rerun re-reads the CSV and adds zero rows).

Pure NumPy/Pillow -- no torch/gradio needed. Run from the repo root:
    python tests/test_param_optimize.py
"""

import os
import sys
import csv
import json
import time
import shutil
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from preprocessing import optimize as O
from preprocessing.steps import STEP_REGISTRY

ORDER = ['sar_intensity_transform', 'speckle_filter', 'outlier_clipping', 'histogram_mapping']


def _make_sar(folder, n=6, size=80, seed=0):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        base = rng.gamma(1.5, 0.15, (size, size)).clip(0, 1)
        base[rng.random((size, size)) > 0.98] = 1.0   # bright outliers -> clipping matters
        Image.fromarray((base * 255).astype('uint8')).save(os.path.join(folder, f'{i}.png'))


def test_param_space_declarations():
    # tunable steps expose a non-empty PARAM_SPACE; structural steps are empty
    tunable = {'sar_intensity_transform', 'speckle_filter', 'outlier_clipping', 'histogram_mapping'}
    structural = {'validate_image', 'resize_or_tile', 'channel_adapter', 'normalize_for_cut'}
    for name, cls in STEP_REGISTRY.items():
        ps = getattr(cls, 'PARAM_SPACE', {})
        if name in tunable:
            assert ps, f'{name} should be tunable'
        if name in structural:
            assert not ps, f'{name} must NOT be tunable (empty PARAM_SPACE)'
    print('PARAM_SPACE declarations (tunable vs structural): OK')


def test_dotted_merge_and_overrides():
    d = {'clahe': {'enabled': False}}
    O._set_dotted(d, 'clahe.clip_limit', 4.0)
    O._set_dotted(d, 'clahe.enabled', True)
    assert d == {'clahe': {'enabled': True, 'clip_limit': 4.0}}

    steps = O.build_param_pipeline_steps(
        ORDER, 'refined_lee',
        {'outlier_clipping': {'max_percentile': 99.95},
         'histogram_mapping': {'clahe.clip_limit': 4.0}})
    oc = [s for s in steps if s['name'] == 'outlier_clipping'][0]
    hm = [s for s in steps if s['name'] == 'histogram_mapping'][0]
    assert oc['params']['max_percentile'] == 99.95
    assert hm['params']['clahe']['clip_limit'] == 4.0
    # unrelated defaults preserved
    assert hm['params']['clahe']['tile_grid_size'] == [8, 8]
    print('dotted merge + override application: OK')


def test_tunable_filtering_and_pruning():
    # structural steps are dropped even if present in the order
    tun = O.tunable_steps_in_order(['resize_or_tile', 'outlier_clipping', 'normalize_for_cut'])
    assert [n for n, _ in tun] == ['outlier_clipping']

    space = O.tunable_steps_in_order(['speckle_filter'])[0][1]
    # damping_factor only relevant for frost
    assert 'damping_factor' not in O._relevant_param_space('speckle_filter', space, 'lee')
    assert 'damping_factor' in O._relevant_param_space('speckle_filter', space, 'frost')
    print('tunable filtering + frost-only damping pruning: OK')


def test_auto_connect_from_best_pipeline():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, 'search')
        os.makedirs(out)
        json.dump({'order': ORDER, 'speckle': 'refined_lee'},
                  open(os.path.join(out, 'best_pipeline.json'), 'w'))
        order, speckle = O.load_best_pipeline(os.path.join(out, 'best_pipeline.json'))
        assert order == ORDER and speckle == 'refined_lee'

        sar = os.path.join(tmp, 'sar')
        _make_sar(sar)
        last = None
        for line in O.optimize_params(sar, out, n_images=6, primary='composite', passes=1):
            last = line
        assert '순서 자동 연결' in '\n'.join([last])  # last log includes final block; check file instead
        best = json.load(open(os.path.join(out, 'best_params_pipeline.json'), encoding='utf-8'))
        assert best['order'] == ORDER and best['speckle'] == 'refined_lee'
        assert isinstance(best['param_overrides'], dict)
        assert best['metric'] is not None
        print(f'auto-connect + run: OK (metric={best["metric"]:.4f}, evals={best["total_evals"]})')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_coordinate_descent_improves_and_is_monotone():
    tmp = tempfile.mkdtemp()
    try:
        sar = os.path.join(tmp, 'sar')
        _make_sar(sar, seed=3)
        out = os.path.join(tmp, 'search')
        os.makedirs(out)

        # baseline metric (default params, no overrides)
        files = O.scan_images(sar, recursive=True, shuffle=False, seed=42)
        base = O.evaluate_param_pipeline(ORDER, 'refined_lee', {}, files)
        base_composite = base['composite']

        last = None
        for line in O.optimize_params(sar, out, order=ORDER, speckle_method='refined_lee',
                                      n_images=6, primary='composite', passes=1):
            last = line
        best = json.load(open(os.path.join(out, 'best_params_pipeline.json'), encoding='utf-8'))
        # coordinate descent can only KEEP or IMPROVE the primary metric vs baseline
        assert best['metric'] >= base_composite - 1e-9, (best['metric'], base_composite)
        print(f'coordinate descent monotone improvement: OK '
              f'(baseline={base_composite:.4f} -> tuned={best["metric"]:.4f})')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_is_deterministic_noop():
    tmp = tempfile.mkdtemp()
    try:
        sar = os.path.join(tmp, 'sar')
        _make_sar(sar, seed=5)
        out = os.path.join(tmp, 'search')
        os.makedirs(out)
        json.dump({'order': ORDER, 'speckle': 'refined_lee'},
                  open(os.path.join(out, 'best_pipeline.json'), 'w'))
        csv_path = os.path.join(out, 'param_search_results.csv')

        counts, times = [], []
        for _ in range(3):
            t0 = time.time()
            for _line in O.optimize_params(sar, out, n_images=6, primary='composite', passes=1):
                pass
            times.append(time.time() - t0)
            counts.append(len(list(csv.DictReader(open(csv_path, encoding='utf-8')))))
        assert counts[0] == counts[1] == counts[2], f'resume must add no rows: {counts}'
        assert times[1] < 0.5, f'resume must be fast (all cached): {times}'
        print(f'deterministic resume (no-op): OK (rows={counts[0]} stable, '
              f'rerun {times[1]*1000:.0f}ms)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_param_space_declarations()
    test_dotted_merge_and_overrides()
    test_tunable_filtering_and_pruning()
    test_auto_connect_from_best_pipeline()
    test_coordinate_descent_improves_and_is_monotone()
    test_resume_is_deterministic_noop()
    print('\nAll parameter-optimization smoke tests passed.')


if __name__ == '__main__':
    main()
