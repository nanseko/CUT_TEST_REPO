""" Stall-detecting subprocess runner, shared by every long-running external
command this project launches (train.py / test.py) — whether from the main
training tab, the inference tab, or a hyperparameter-search trial.

Before this module existed, each caller had its own plain blocking
``for line in iter(proc.stdout.readline, ''): ...`` loop, which has NO way to
notice a hang: a stuck DataLoader worker, a GPU/driver hang, a stalled
network-drive read, etc. would block that loop FOREVER. The main training tab
(gui.py's training_worker) got a hand-rolled fix for this; hyperparameter
search and inference did not, so a single hung trial/run would silently
freeze that whole operation with no recovery. This module makes "run a
subprocess and never block forever" a single, tested, reusable primitive so
the fix lives in one place and applies everywhere.

Mechanism: a background reader thread drains stdout into a queue.Queue, so the
main loop can poll with a timeout (``queue.get(timeout=...)``) instead of
blocking on ``readline()`` directly — plain blocking reads have no portable
timeout, but a thread+queue does, identically on Windows/Linux.

Two interfaces, one implementation:
  - ``run_watched_stream(...)``  — a GENERATOR yielding each stdout line as it
    arrives. Use this when the caller itself needs to stream progress onward
    (e.g. a Gradio generator callback like the inference tab, or
    evaluation/hparam_search.py's per-trial train/test calls).
  - ``run_watched(...)``         — a plain blocking call taking an `on_line`
    callback. Use this when a background-thread worker just needs side
    effects per line (e.g. gui.py's training_worker, which updates shared
    state that a SEPARATE polling loop streams to the UI).
Both return/populate the same result shape: {'rc', 'stalled', 'stopped', 'tail'}.
"""

import os
import time
import queue
import threading
import subprocess

# Floor for the stall window regardless of the requested stall_minutes, so a
# mistakenly tiny value can't cause false-positive kills during normal pauses
# (checkpoint saving, first-batch data_dependent_initialize, ...).
MIN_STALL_SECONDS = 60


def _reader_thread(pipe, q):
    try:
        for raw in iter(pipe.readline, ''):
            q.put(raw)
    except Exception:
        pass
    q.put(None)   # EOF sentinel


def run_watched_stream(cmd, cwd, holder, stall_minutes=20, poll_seconds=5,
                       progress_re=None, is_stopped=None, env=None,
                       min_stall_seconds=None, on_start=None):
    """Run `cmd`, yielding each stdout line as it arrives. The final result
    dict ({'rc', 'stalled', 'stopped', 'tail'} — same shape as run_watched's
    return value) is written into the caller-supplied `holder` dict once the
    generator is fully drained (mirrors the pre-existing
    evaluation/hparam_search.py `_stream_subprocess(cmd, cwd, log, tag,
    holder)` convention, so it's a drop-in replacement there).

    Kills the process if no line matching `progress_re` arrives within
    `stall_minutes` (if `progress_re` is None, ANY non-empty line counts as
    progress). `is_stopped()` is polled every `poll_seconds` so a deliberate
    stop request is honoured immediately. `on_start(proc)`, if given, is
    called right after the process is spawned (e.g. so a caller can stash the
    Popen object for an immediate external `.terminate()`).

    Never raises on a hang or a non-zero exit — callers decide what to do
    (retry, mark-failed-and-move-on, surface an error, ...); it only
    guarantees the generator eventually finishes.
    """
    is_stopped = is_stopped or (lambda: False)
    stall_seconds = max(min_stall_seconds or MIN_STALL_SECONDS, int(float(stall_minutes) * 60))
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env or dict(os.environ, PYTHONUNBUFFERED='1'))
    if on_start:
        on_start(proc)
    q = queue.Queue()
    threading.Thread(target=_reader_thread, args=(proc.stdout, q), daemon=True).start()

    last_progress = time.time()
    tail = []
    eof = False

    while True:
        if is_stopped():
            try:
                proc.terminate()
            except Exception:
                pass
            holder.update(rc=None, stalled=False, stopped=True, tail=tail)
            return
        try:
            raw = q.get(timeout=poll_seconds)
        except queue.Empty:
            pass
        else:
            if raw is None:
                eof = True
                break
            line = raw.rstrip('\n')
            if line and (progress_re is None or progress_re.search(line)):
                last_progress = time.time()
            if line:
                tail.append(line)
                del tail[:-50]
                yield line
        if time.time() - last_progress > stall_seconds:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
            holder.update(rc=None, stalled=True, stopped=False, tail=tail)
            return

    # drain any buffered remaining output, then get the real exit code
    while not eof:
        raw = q.get()
        if raw is None:
            break
        line = raw.rstrip('\n')
        if line:
            tail.append(line)
            del tail[:-50]
            yield line
    holder.update(rc=proc.wait(), stalled=False, stopped=False, tail=tail)


def run_watched(cmd, cwd, on_line, stall_minutes=20, poll_seconds=5,
                progress_re=None, is_stopped=None, env=None,
                min_stall_seconds=None, on_start=None):
    """Blocking variant of run_watched_stream: calls `on_line(line)` for every
    stdout line instead of yielding it, and returns the final result dict
    directly (rather than populating a caller-supplied holder). See
    run_watched_stream for parameter details.
    """
    holder = {}
    for line in run_watched_stream(
            cmd, cwd, holder, stall_minutes=stall_minutes, poll_seconds=poll_seconds,
            progress_re=progress_re, is_stopped=is_stopped, env=env,
            min_stall_seconds=min_stall_seconds, on_start=on_start):
        on_line(line)
    return holder
