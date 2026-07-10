""" Smoke test for the shared stall-watchdog primitive (util/subprocess_watchdog.py).

This is the single implementation now reused by:
  - gui.py::training_worker (Tab 6, main training)
  - gui.py::run_inference (Tab 7, inference/test.py)
  - evaluation/hparam_search.py::_stream_subprocess (Tab 10, per-trial train/test)

Before this module existed, only training_worker had hang protection; a stall
during inference or (especially) hyperparameter search would block forever
with no recovery. This test covers the shared primitive directly (both the
blocking `run_watched` and the generator `run_watched_stream` interfaces);
each caller's own integration is covered by its own test file
(test_training_watchdog.py, test_hparam_search.py, and the run_inference
checks below).

Run from the repo root:  python tests/test_subprocess_watchdog.py
"""

import os
import sys
import time
import threading
import tempfile
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.subprocess_watchdog import run_watched, run_watched_stream


FAKE_SCRIPT = textwrap.dedent('''
    import sys, time
    mode = sys.argv[1]
    print("(epoch: 1, iters: 1) G: 1.0", flush=True)
    if mode == 'hang':
        time.sleep(3600)
    elif mode == 'crash':
        sys.exit(1)
    else:
        print("End of epoch 1 / 1", flush=True)
        sys.exit(0)
''')


def _make_script(tmp):
    path = os.path.join(tmp, 'fake.py')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(FAKE_SCRIPT)
    return path


def test_run_watched_succeeds():
    tmp = tempfile.mkdtemp()
    script = _make_script(tmp)
    lines = []
    result = run_watched([sys.executable, '-u', script, 'succeed'], tmp, lines.append,
                         stall_minutes=20)
    assert result['rc'] == 0 and not result['stalled'] and not result['stopped']
    assert lines == ['(epoch: 1, iters: 1) G: 1.0', 'End of epoch 1 / 1']
    print('run_watched succeed: OK')


def test_run_watched_detects_hang():
    tmp = tempfile.mkdtemp()
    script = _make_script(tmp)
    lines = []
    t0 = time.time()
    result = run_watched([sys.executable, '-u', script, 'hang'], tmp, lines.append,
                         stall_minutes=0.03, min_stall_seconds=1)
    dt = time.time() - t0
    assert result['stalled'] and result['rc'] is None and not result['stopped']
    assert dt < 15, f'hang detection took too long: {dt:.1f}s'
    print(f'run_watched hang detection: OK (dt={dt:.1f}s)')


def test_run_watched_reports_crash():
    tmp = tempfile.mkdtemp()
    script = _make_script(tmp)
    lines = []
    result = run_watched([sys.executable, '-u', script, 'crash'], tmp, lines.append,
                         stall_minutes=20)
    assert result['rc'] == 1 and not result['stalled']
    print('run_watched crash reporting: OK')


def test_run_watched_stop_is_immediate_not_stall_wait():
    tmp = tempfile.mkdtemp()
    script = _make_script(tmp)
    stopped = {'v': False}

    def stopper():
        time.sleep(1.0)
        stopped['v'] = True
    threading.Thread(target=stopper, daemon=True).start()

    t0 = time.time()
    result = run_watched([sys.executable, '-u', script, 'hang'], tmp, lambda l: None,
                         stall_minutes=20, is_stopped=lambda: stopped['v'])
    dt = time.time() - t0
    assert result['stopped'] and not result['stalled'] and result['rc'] is None
    assert dt < 10, f'stop must not wait for the (much longer) stall window: {dt:.1f}s'
    print(f'run_watched stop-is-immediate: OK (dt={dt:.1f}s)')


def test_run_watched_stream_generator_interface():
    """The generator variant used by run_inference / hparam_search: yields
    lines as they arrive and populates a caller-supplied holder dict at the
    end (mirrors the pre-existing _stream_subprocess(cmd, cwd, log, tag,
    holder) convention)."""
    tmp = tempfile.mkdtemp()
    script = _make_script(tmp)

    holder = {}
    streamed = list(run_watched_stream([sys.executable, '-u', script, 'succeed'], tmp, holder,
                                       stall_minutes=20))
    assert streamed == ['(epoch: 1, iters: 1) G: 1.0', 'End of epoch 1 / 1']
    assert holder == {'rc': 0, 'stalled': False, 'stopped': False, 'tail': streamed}
    print('run_watched_stream generator interface: OK')

    holder2 = {}
    t0 = time.time()
    streamed2 = list(run_watched_stream([sys.executable, '-u', script, 'hang'], tmp, holder2,
                                        stall_minutes=0.03, min_stall_seconds=1))
    dt = time.time() - t0
    assert holder2['stalled'] and holder2['rc'] is None
    assert dt < 15
    print(f'run_watched_stream hang via generator: OK (dt={dt:.1f}s)')


def test_progress_re_ignores_non_matching_lines_for_stall_purposes():
    """A process that prints UNRELATED chatter (not matching progress_re) but
    never actual progress must still be treated as stalled -- progress_re
    lets callers require a SPECIFIC kind of line (e.g. "iters:") to count,
    not just any output."""
    import re
    tmp = tempfile.mkdtemp()
    script = os.path.join(tmp, 'chatty.py')
    with open(script, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent('''
            import time
            for i in range(50):
                print("just some noise, not real progress", flush=True)
                time.sleep(0.1)
        '''))
    lines = []
    result = run_watched([sys.executable, '-u', script], tmp, lines.append,
                         stall_minutes=0.03, min_stall_seconds=1,
                         progress_re=re.compile(r'iters:'))
    assert result['stalled'], 'chatter that never matches progress_re must still count as a stall'
    assert len(lines) > 0, 'lines should still be captured even though none counted as progress'
    print(f'progress_re gates what counts as progress: OK ({len(lines)} non-progress lines seen)')


def test_run_inference_uses_the_shared_watchdog():
    """Integration check: gui.py's Tab 7 (추론/테스트) previously had NO stall
    protection at all (a plain blocking readline loop) -- verify it now
    detects a hang via the shared primitive, and that normal completion is
    unaffected."""
    import numpy as np
    from PIL import Image
    import gui

    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, 'data')
    for sub in ('trainA', 'trainB', 'testA', 'testB'):
        d = os.path.join(root, sub)
        os.makedirs(d, exist_ok=True)
        for i in range(2):
            Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype('uint8')) \
                .save(os.path.join(d, f'{i}.png'))
    cfg = dict(gui.DEFAULTS)
    cfg.update(dataroot=root, results_dir=os.path.join(tmp, 'results'), name='hangtest')
    vals = [cfg[k] for k in gui.CONFIG_KEYS]

    hang_script = os.path.join(tmp, 'fake_test.py')
    with open(hang_script, 'w', encoding='utf-8') as f:
        f.write("import time\nprint('loading the model', flush=True)\ntime.sleep(3600)\n")

    orig_build_test_cmd = gui.build_test_cmd
    orig_min_stall = gui.MIN_STALL_SECONDS
    gui.build_test_cmd = lambda cfg, num_test, epoch: [sys.executable, '-u', hang_script]
    gui.MIN_STALL_SECONDS = 1
    try:
        t0 = time.time()
        last = None
        for status, gallery in gui.run_inference(5, 'latest', 0.03, *vals):
            last = status
        dt = time.time() - t0
        assert dt < 15, f'run_inference must not hang forever: {dt:.1f}s'
        assert '행(hang) 감지' in last or '강제 종료' in last, last
        print(f'run_inference hang detection (shared watchdog): OK (dt={dt:.1f}s)')
    finally:
        gui.build_test_cmd = orig_build_test_cmd
        gui.MIN_STALL_SECONDS = orig_min_stall

    # normal completion path still works
    ok_script = os.path.join(tmp, 'fake_test_ok.py')
    with open(ok_script, 'w', encoding='utf-8') as f:
        f.write("print('processing (0000)-th image...', flush=True)\n")
    out_dir = os.path.join(tmp, 'results', 'hangtest', 'test_latest', 'images', 'fake_B')
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype('uint8')) \
        .save(os.path.join(out_dir, '0.png'))
    gui.build_test_cmd = lambda cfg, num_test, epoch: [sys.executable, '-u', ok_script]
    try:
        last = None
        for status, gallery in gui.run_inference(5, 'latest', 20, *vals):
            last = (status, gallery)
        assert '완료' in last[0] and len(last[1]) == 1
        print('run_inference normal completion (unaffected by watchdog): OK')
    finally:
        gui.build_test_cmd = orig_build_test_cmd


def main():
    test_run_watched_succeeds()
    test_run_watched_detects_hang()
    test_run_watched_reports_crash()
    test_run_watched_stop_is_immediate_not_stall_wait()
    test_run_watched_stream_generator_interface()
    test_progress_re_ignores_non_matching_lines_for_stall_purposes()
    test_run_inference_uses_the_shared_watchdog()
    print('\nAll subprocess-watchdog smoke tests passed.')


if __name__ == '__main__':
    main()
