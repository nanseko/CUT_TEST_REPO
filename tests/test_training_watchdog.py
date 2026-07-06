""" Smoke test for the training stall-watchdog (gui.py's training_worker).

Long (multi-day) unattended runs can freeze without the process crashing (a
stuck DataLoader worker, a GPU/driver hang, a stalled network-drive read...).
training_worker detects "no forward-progress log line for N minutes" and
auto-restarts with --continue_train from the last checkpoint. This is tested
here WITHOUT a real multi-minute wait or a real GAN training run: gui.py's
MIN_STALL_SECONDS floor is monkeypatched down, and gui.build_train_cmd is
stubbed to launch a tiny fake "trainer" script instead of the real train.py,
so the watchdog logic itself (stall detection, kill, restart with
continue_train, max-restart cap, responsive Stop) is verified directly.

Run from the repo root:  python tests/test_training_watchdog.py
"""

import os
import sys
import time
import textwrap
import tempfile
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gui


FAKE_TRAINER = textwrap.dedent('''
    import sys, time
    mode = sys.argv[1]
    print("(epoch: 1, iters: 1, time: 0.01, data: 0.0) G: 1.0", flush=True)
    if mode == 'hang':
        time.sleep(3600)   # simulate a real hang (killed by the watchdog)
    elif mode == 'crash':
        sys.exit(1)
    else:
        print("End of epoch 1 / 1", flush=True)
        sys.exit(0)
''')


def _make_fake_trainer(tmp):
    path = os.path.join(tmp, 'fake_trainer.py')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(FAKE_TRAINER)
    return path


def test_stall_detect_and_auto_restart():
    """attempt 0 hangs -> watchdog kills+restarts with continue_train=True ->
    attempt 1 (fake 'recovered' run) succeeds -> training completes, restarts>=1."""
    tmp = tempfile.mkdtemp()
    script = _make_fake_trainer(tmp)
    orig_build_cmd = gui.build_train_cmd
    orig_min_stall = gui.MIN_STALL_SECONDS
    calls = []

    def fake_build_cmd(cfg):
        calls.append(dict(cfg))
        mode = 'succeed' if cfg.get('continue_train') else 'hang'
        return [sys.executable, '-u', script, mode]

    gui.build_train_cmd = fake_build_cmd
    gui.MIN_STALL_SECONDS = 1   # allow a short stall window for the test
    try:
        state = gui.TrainingState()
        state.running = True
        state.log_file = None
        cfg = dict(gui.DEFAULTS, n_epochs=1, n_epochs_decay=0,
                  checkpoints_dir=os.path.join(tmp, 'ck'), name='watchdog_test')

        t0 = time.time()
        gui.training_worker(cfg, state, stall_minutes=0.03, max_restarts=5, backoff_seconds=1)
        dt = time.time() - t0

        assert state.message == '완료 (Done)', state.message
        assert state.restarts >= 1, state.restarts
        assert len(calls) == 2, calls
        assert calls[0].get('continue_train') is not True
        assert calls[1].get('continue_train') is True
        assert dt < 30, f'watchdog should recover quickly in this synthetic test, took {dt:.1f}s'
        print(f'stall_detect_and_auto_restart: OK (dt={dt:.1f}s, restarts={state.restarts}, '
              f'attempts={len(calls)})')
    finally:
        gui.build_train_cmd = orig_build_cmd
        gui.MIN_STALL_SECONDS = orig_min_stall


def test_crash_triggers_restart_too():
    """A non-zero exit (not just a stall) must also trigger auto-restart."""
    tmp = tempfile.mkdtemp()
    script = _make_fake_trainer(tmp)
    orig_build_cmd = gui.build_train_cmd
    calls = []

    def fake_build_cmd(cfg):
        calls.append(dict(cfg))
        mode = 'succeed' if cfg.get('continue_train') else 'crash'
        return [sys.executable, '-u', script, mode]

    gui.build_train_cmd = fake_build_cmd
    try:
        state = gui.TrainingState()
        state.running = True
        cfg = dict(gui.DEFAULTS, n_epochs=1, n_epochs_decay=0,
                  checkpoints_dir=os.path.join(tmp, 'ck'), name='crash_test')
        gui.training_worker(cfg, state, stall_minutes=20, max_restarts=5, backoff_seconds=1)
        assert state.message == '완료 (Done)'
        assert state.restarts == 1
        assert len(calls) == 2 and calls[1].get('continue_train') is True
        print('crash_triggers_restart: OK')
    finally:
        gui.build_train_cmd = orig_build_cmd


def test_max_restarts_exceeded():
    """A trainer that always hangs must stop after max_restarts, not loop forever."""
    tmp = tempfile.mkdtemp()
    script = _make_fake_trainer(tmp)
    orig_build_cmd = gui.build_train_cmd
    orig_min_stall = gui.MIN_STALL_SECONDS
    calls = []

    def fake_build_cmd(cfg):
        calls.append(1)
        return [sys.executable, '-u', script, 'hang']   # always hangs

    gui.build_train_cmd = fake_build_cmd
    gui.MIN_STALL_SECONDS = 1
    try:
        state = gui.TrainingState()
        state.running = True
        cfg = dict(gui.DEFAULTS, n_epochs=1, n_epochs_decay=0,
                  checkpoints_dir=os.path.join(tmp, 'ck'), name='alwayshang_test')
        gui.training_worker(cfg, state, stall_minutes=0.03, max_restarts=2, backoff_seconds=1)
        assert '초과' in state.message, state.message
        assert state.restarts == 3, state.restarts   # exceeds max_restarts=2 on the 3rd attempt
        assert len(calls) == 3, calls
        print(f'max_restarts_exceeded: OK (message={state.message!r}, attempts={len(calls)})')
    finally:
        gui.build_train_cmd = orig_build_cmd
        gui.MIN_STALL_SECONDS = orig_min_stall


def test_stop_cancels_during_backoff_wait():
    """Pressing Stop while the watchdog is waiting to restart must cancel promptly,
    not wait out the full backoff or attempt another restart."""
    tmp = tempfile.mkdtemp()
    script = _make_fake_trainer(tmp)
    orig_build_cmd = gui.build_train_cmd
    orig_min_stall = gui.MIN_STALL_SECONDS
    calls = []

    def fake_build_cmd(cfg):
        calls.append(1)
        return [sys.executable, '-u', script, 'crash']   # always crashes -> would restart

    gui.build_train_cmd = fake_build_cmd
    gui.MIN_STALL_SECONDS = 1
    try:
        state = gui.TrainingState()
        state.running = True

        def request_stop_soon():
            time.sleep(1.5)   # let the first attempt crash and enter backoff
            state.stop_requested = True

        threading.Thread(target=request_stop_soon, daemon=True).start()
        cfg = dict(gui.DEFAULTS, n_epochs=1, n_epochs_decay=0,
                  checkpoints_dir=os.path.join(tmp, 'ck'), name='stopcancel_test')
        t0 = time.time()
        gui.training_worker(cfg, state, stall_minutes=20, max_restarts=10, backoff_seconds=30)
        dt = time.time() - t0
        assert state.message == '중단됨 (Stopped)', state.message
        assert dt < 10, f'stop must cancel the 30s backoff wait promptly, took {dt:.1f}s'
        assert len(calls) == 1, calls   # must NOT have restarted after stop was requested
        print(f'stop_cancels_during_backoff: OK (dt={dt:.1f}s, attempts={len(calls)})')
    finally:
        gui.build_train_cmd = orig_build_cmd
        gui.MIN_STALL_SECONDS = orig_min_stall


def main():
    test_stall_detect_and_auto_restart()
    test_crash_triggers_restart_too()
    test_max_restarts_exceeded()
    test_stop_cancels_during_backoff_wait()
    print('\nAll training-watchdog smoke tests passed.')


if __name__ == '__main__':
    main()
