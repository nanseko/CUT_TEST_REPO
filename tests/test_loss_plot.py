""" Smoke test for the per-epoch loss-curve module (util/loss_plot.py).

Verifies that loss_log.txt (the format train.py/visualizer already writes) is
parsed into per-epoch means, that the combined discriminator loss D is
synthesized as mean(D_real, D_fake), that a loss enabled mid-run lines up on
the epoch axis, and that the CSV (always) and PNG (if matplotlib present) are
written. No torch / no real training needed.

Run from the repo root:  python tests/test_loss_plot.py
"""

import os
import sys
import csv
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.loss_plot import parse_loss_log, update_loss_plot


def _write_log(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('================ Training Loss ================\n')   # header, must be ignored
        for ln in lines:
            f.write(ln + '\n')


def test_parse_and_synthesize_D():
    tmp = tempfile.mkdtemp()
    log = os.path.join(tmp, 'loss_log.txt')
    _write_log(log, [
        '(epoch: 1, iters: 100, time: 0.12, data: 0.00) G_GAN: 0.60 D_real: 0.50 D_fake: 0.30 G: 4.0 NCE: 3.0',
        '(epoch: 1, iters: 200, time: 0.12, data: 0.00) G_GAN: 0.40 D_real: 0.30 D_fake: 0.10 G: 3.0 NCE: 2.0',
        '(epoch: 2, iters: 100, time: 0.12, data: 0.00) G_GAN: 0.50 D_real: 0.40 D_fake: 0.20 G: 2.0 NCE: 1.0',
    ])
    epochs, series = parse_loss_log(log)
    assert epochs == [1, 2]
    # per-epoch means
    assert abs(series['G'][0] - 3.5) < 1e-9 and abs(series['G'][1] - 2.0) < 1e-9
    assert abs(series['NCE'][0] - 2.5) < 1e-9
    # D synthesized = mean(D_real, D_fake), per epoch
    #   epoch1: D_real mean=0.4, D_fake mean=0.2 -> D=0.3
    assert abs(series['D'][0] - 0.3) < 1e-9, series['D']
    assert abs(series['D'][1] - 0.3) < 1e-9, series['D']
    # header 'iters'/'time'/'data' must NOT be parsed as losses
    assert 'iters' not in series and 'time' not in series and 'data' not in series
    print('parse + D synthesis + header-safety: OK')


def test_midrun_enabled_loss_aligns():
    tmp = tempfile.mkdtemp()
    log = os.path.join(tmp, 'loss_log.txt')
    _write_log(log, [
        '(epoch: 1, iters: 100, time: 0.1, data: 0.0) G: 4.0 NCE: 3.0',
        '(epoch: 2, iters: 100, time: 0.1, data: 0.0) G: 3.0 NCE: 2.0 G_coherence: 0.5',  # enabled at ep2
    ])
    epochs, series = parse_loss_log(log)
    assert epochs == [1, 2]
    # G_coherence absent in epoch 1 -> None, present in epoch 2
    assert series['G_coherence'][0] is None and abs(series['G_coherence'][1] - 0.5) < 1e-9
    print('mid-run enabled loss aligns on epoch axis: OK')


def test_update_writes_csv_and_maybe_png():
    tmp = tempfile.mkdtemp()
    exp = os.path.join(tmp, 'ck', 'expA')
    os.makedirs(exp)
    _write_log(os.path.join(exp, 'loss_log.txt'), [
        f'(epoch: {e}, iters: 100, time: 0.1, data: 0.0) '
        f'G_GAN: {0.5:.2f} D_real: {0.4:.2f} D_fake: {0.2:.2f} G: {5.0-e:.2f} NCE: {4.0-e:.2f}'
        for e in range(1, 6)
    ])
    out = update_loss_plot(os.path.join(tmp, 'ck'), 'expA')
    assert out['ok'] and out['epochs'] == 5
    assert out['csv'] and os.path.exists(out['csv'])
    rows = list(csv.DictReader(open(out['csv'], encoding='utf-8')))
    assert len(rows) == 5
    assert 'D' in rows[0] and 'G' in rows[0] and 'NCE' in rows[0]
    # PNG is optional (matplotlib); if reported, it must exist
    if out['png']:
        assert os.path.exists(out['png'])
        png_note = 'PNG rendered'
    else:
        png_note = 'PNG skipped (matplotlib absent) — CSV still written'
    # empty / missing log is handled gracefully
    empty = update_loss_plot(os.path.join(tmp, 'ck'), 'no_such_exp')
    assert not empty['ok']
    print(f'update_loss_plot writes CSV (+{png_note}); missing-log handled: OK')


def main():
    test_parse_and_synthesize_D()
    test_midrun_enabled_loss_aligns()
    test_update_writes_csv_and_maybe_png()
    print('\nAll loss-plot smoke tests passed.')


if __name__ == '__main__':
    main()
