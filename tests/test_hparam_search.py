""" Smoke test for the hyperparameter auto-search (evaluation/hparam_search.py).

Covers: canonicalisation/dedup of equivalent configs, deterministic sampling,
a real (tiny) end-to-end Successive-Halving run through train.py/test.py, and
resume behaviour (a second identical invocation must retrain nothing and add
no CSV rows).

Requires torch (runs two 1-epoch CPU trainings on a 3-image synthetic set,
~30-60s total). Run from the repo root:  python tests/test_hparam_search.py
"""

import os
import sys
import csv
import time
import shutil
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.hparam_search import (
    canonicalize, sample_trials, trial_sig, hparam_search, load_best,
)


def test_canonicalize_and_sampling():
    # attention 'none' collapses positions/reduction -> equivalents dedupe
    a = canonicalize({'attention_type': 'none', 'attention_positions': 'enc+res',
                      'attention_reduction': 8, 'lambda_grad': 0.0, 'lambda_lap': 0.0,
                      'lambda_coherence': 0.0, 'lambda_color': 0.0,
                      'reflector_boost': 5.0, 'reflector_weighted': True,
                      'saliency_patch_sampling': False})
    b = canonicalize({'attention_type': 'none', 'attention_positions': 'enc',
                      'attention_reduction': 16, 'lambda_grad': 0.0, 'lambda_lap': 0.0,
                      'lambda_coherence': 0.0, 'lambda_color': 0.0,
                      'reflector_boost': 3.0, 'reflector_weighted': False,
                      'saliency_patch_sampling': False})
    assert trial_sig(a) == trial_sig(b)

    # positions decode into the three bool flags
    c = canonicalize({'attention_type': 'coord', 'attention_positions': 'enc+res',
                      'attention_reduction': 8, 'lambda_grad': 1.0, 'lambda_lap': 0.5,
                      'lambda_coherence': 0.5, 'lambda_color': 0.0,
                      'reflector_boost': 5.0, 'reflector_weighted': True,
                      'saliency_patch_sampling': True})
    assert (c['attention_encoder'], c['attention_resblocks'], c['attention_decoder']) \
        == (True, True, False)

    # sampling: n distinct canonical trials, deterministic with seed
    t1 = sample_trials(n_trials=12, seed=42)
    t2 = sample_trials(n_trials=12, seed=42)
    sigs = [trial_sig(t) for t in t1]
    assert len(t1) == 12 and len(set(sigs)) == 12
    assert sigs == [trial_sig(t) for t in t2]
    print('canonicalize/sampling: OK')


def _make_dataset(root, n=3, size=64):
    from PIL import Image
    rng = np.random.default_rng(0)
    for sub in ('trainA', 'trainB', 'testA', 'testB'):
        d = os.path.join(root, sub)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            Image.fromarray((rng.random((size, size, 3)) * 255).astype('uint8')) \
                .save(os.path.join(d, f'{i}.png'))


TINY_SPACE = {
    'attention_type': ['none'],
    'attention_positions': ['enc'],
    'attention_reduction': [16],
    'lambda_grad': [0.0, 1.0],   # -> exactly 2 distinct trials
    'lambda_lap': [0.0],
    'lambda_coherence': [0.0],
    'lambda_color': [0.0],
    'reflector_boost': [3.0],
    'reflector_weighted': [False],
    'saliency_patch_sampling': [False],
}


def test_end_to_end_and_resume():
    import gui   # command builders (mirrors the GUI exactly)

    tmp = tempfile.mkdtemp()
    try:
        data = os.path.join(tmp, 'data')
        _make_dataset(data)
        base = dict(gui.DEFAULTS)
        base.update(dataroot=data, name='unused',
                    checkpoints_dir=os.path.join(tmp, 'ck'),
                    results_dir=os.path.join(tmp, 'res'),
                    gpu_ids='-1', batch_size=1, load_size=64, crop_size=64,
                    num_threads=0)
        out = os.path.join(tmp, 'hps')

        def run():
            last = None
            for line in hparam_search(base, gui.build_train_cmd, gui.build_test_cmd,
                                      out, space=TINY_SPACE, n_trials=4,
                                      stage1_epochs=1, stage2_epochs=1, top_k=1,
                                      stage1_images=3, stage2_images=0, num_test=2,
                                      primary='epi', eo_dir=None):
                last = line
            return last

        last = run()
        assert '최적 하이퍼파라미터' in last
        rows = list(csv.DictReader(open(os.path.join(out, 'hparam_results.csv'),
                                        encoding='utf-8')))
        s1 = [r for r in rows if r['stage'] == '1']
        s2 = [r for r in rows if r['stage'] == '2']
        assert len(s1) == 2 and len(s2) == 1, (len(s1), len(s2))
        assert all(r['status'] == 'ok' for r in rows)
        best = load_best(out)
        assert best is not None and 'overrides' in best and best['metrics']['epi'] is not None
        print(f'end-to-end: OK (stage1={len(s1)}, stage2={len(s2)}, '
              f'best epi={best["metrics"]["epi"]:.4f})')

        # resume: identical invocation adds 0 rows and retrains nothing
        t0 = time.time()
        run()
        dt = time.time() - t0
        rows2 = list(csv.DictReader(open(os.path.join(out, 'hparam_results.csv'),
                                         encoding='utf-8')))
        assert len(rows2) == len(rows), (len(rows2), len(rows))
        assert dt < 10, f'resume must skip all training, took {dt:.1f}s'
        print(f'resume/dedup: OK ({dt:.1f}s, no retraining, no duplicate rows)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_canonicalize_and_sampling()
    test_end_to_end_and_resume()
    print('\nAll hparam-search smoke tests passed.')


if __name__ == '__main__':
    main()
