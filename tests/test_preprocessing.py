""" Smoke test for the SAR preprocessing pipeline (no TensorFlow needed). """

import os
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import preprocessing as PP


def _make_sar(folder, n=5, size=96):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        base = np.zeros((size, size), np.float32)
        base[20:70, 30:60] = 0.6
        img = np.clip(base * rng.gamma(1.0, 1.0, (size, size)) * 120, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(folder, f'{i:05d}.png'))


def main():
    tmp = tempfile.mkdtemp()
    raw = os.path.join(tmp, 'raw')
    out = os.path.join(tmp, 'out')
    _make_sar(raw, n=5)

    cfg = PP.default_config()
    cfg['io'].update(input_dir=raw, output_dir=out, max_items=0)

    # every speckle method must run without error and keep shape
    for m in PP.SPECKLE_METHODS:
        c = PP.default_config()
        c['io'].update(input_dir=raw, output_dir=out)
        for s in c['pipeline']['steps']:
            if s['name'] == 'speckle_filter':
                s['params']['method'] = m
        before, after = PP.preprocess_single(c, os.path.join(raw, '00000.png'))
        assert after.shape == (256, 256, 3), (m, after.shape)
        assert after.dtype == np.uint8

    # full run
    last = None
    for log, prev in PP.run_pipeline(cfg):
        last = (log, prev)
    imgs = os.listdir(os.path.join(out, 'images'))
    assert len(imgs) == 5, imgs
    assert os.path.exists(os.path.join(out, 'manifest.csv'))

    # CUT export
    msg = PP.export_cut_layout(os.path.join(out, 'images'),
                               os.path.join(tmp, 'cut'), test_ratio=0.4)
    assert 'trainA' in msg
    print('Preprocessing smoke tests passed.')


if __name__ == '__main__':
    main()
