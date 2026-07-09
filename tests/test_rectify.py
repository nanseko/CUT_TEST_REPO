""" Smoke test for evaluation/rectify.py's folder-batch fix.

Bug fixed: rectify_folder's per-file processing was wrapped in a bare
`except Exception: continue` with NO reporting, so if opencv wasn't installed
(a real, common case -- it's an optional dependency), EVERY file in the folder
would raise ImportError, be silently swallowed, and the function would return
"0 detected" -- indistinguishable from "ran fine, found nothing". The GUI then
showed a misleading "완료" (success) message with an empty gallery. This test
verifies: (1) a full multi-image folder is genuinely processed end-to-end
(not just one file), (2) a missing-cv2 condition fails LOUDLY and immediately
instead of silently degrading to a fake "success", (3) individual file
failures are counted and surfaced, not swallowed.

Requires opencv (pip install opencv-python / opencv-python-headless).
Run from the repo root:  python tests/test_rectify.py
"""

import os
import sys
import shutil
import tempfile
import builtins

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation as EV
import cv2


def _make_rect_images(folder, n, size=128):
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        img = np.zeros((size, size, 3), np.uint8)
        box = cv2.boxPoints(((size // 2, size // 2), (40, 20), 10 + i * 8)).astype(np.int32)
        cv2.fillPoly(img, [box], (255, 255, 255))
        Image.fromarray(img).save(os.path.join(folder, f'img{i:03d}.png'))


def test_processes_the_whole_folder_not_one_file():
    tmp = tempfile.mkdtemp()
    try:
        indir = os.path.join(tmp, 'fake_B')
        N = 7
        _make_rect_images(indir, N)
        outdir = os.path.join(tmp, 'rectified')
        csv_path, n_regions, n_ok, n_fail, failures = EV.rectify_folder(indir, outdir)
        assert n_ok == N, f'expected all {N} files processed, got {n_ok}'
        assert n_fail == 0, failures
        assert n_regions == N   # one clean rectangle per image
        saved_pngs = [f for f in os.listdir(outdir) if f.endswith('.png')]
        assert len(saved_pngs) == N, saved_pngs
        assert os.path.exists(csv_path)
        print(f'processes_the_whole_folder: OK ({n_ok}/{N} files, {n_regions} rectangles)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_random_noise_folder_all_files_still_processed():
    """No shapes to detect (0 regions is a legitimate outcome), but every file
    must still be genuinely OPENED and PROCESSED, not silently skipped."""
    tmp = tempfile.mkdtemp()
    try:
        indir = os.path.join(tmp, 'fake_B')
        os.makedirs(indir)
        rng = np.random.default_rng(0)
        N = 5
        for i in range(N):
            Image.fromarray((rng.random((100, 100, 3)) * 255).astype('uint8')) \
                .save(os.path.join(indir, f'{i}.png'))
        outdir = os.path.join(tmp, 'rectified')
        csv_path, n_regions, n_ok, n_fail, failures = EV.rectify_folder(indir, outdir)
        assert n_ok == N, f'expected all {N} files processed even with 0 detections, got {n_ok}'
        assert n_fail == 0, failures
        overlay_pngs = [f for f in os.listdir(outdir) if f.endswith('.png')]
        assert len(overlay_pngs) == N, (
            'every processed file must still get an overlay PNG saved, even with 0 regions')
        print(f'random_noise_folder_all_processed: OK ({n_ok}/{N}, {n_regions} regions, '
              f'{len(overlay_pngs)} overlays saved)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_cv2_fails_loudly_not_silently():
    """The historical bug: cv2 missing -> every file's ImportError silently
    swallowed -> misleading '0 detected, success' result. Must now raise
    immediately with a clear message, BEFORE looping over any files."""
    tmp = tempfile.mkdtemp()
    try:
        indir = os.path.join(tmp, 'fake_B')
        _make_rect_images(indir, 4)
        outdir = os.path.join(tmp, 'rectified')

        orig_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == 'cv2':
                raise ModuleNotFoundError("No module named 'cv2'")
            return orig_import(name, *a, **k)

        builtins.__import__ = fake_import
        try:
            raised = False
            try:
                EV.rectify_folder(indir, outdir)
            except ImportError as exc:
                raised = True
                assert 'opencv' in str(exc).lower()
        finally:
            builtins.__import__ = orig_import

        assert raised, 'missing cv2 must raise ImportError immediately, not swallow it'
        # must fail BEFORE processing (no misleading partial output)
        assert not os.path.exists(outdir) or not os.listdir(outdir)
        print('missing_cv2_fails_loudly: OK (raises immediately, no misleading output)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_partial_failures_are_counted_not_swallowed():
    """A corrupt/unreadable file mixed in with good ones must be counted as a
    failure (with its filename+reason reported), not silently dropped -- while
    every other file in the folder still gets processed normally."""
    tmp = tempfile.mkdtemp()
    try:
        indir = os.path.join(tmp, 'fake_B')
        _make_rect_images(indir, 3)
        # a genuinely corrupt "image" (not valid image bytes)
        with open(os.path.join(indir, 'zzz_corrupt.png'), 'wb') as f:
            f.write(b'not a real png')
        outdir = os.path.join(tmp, 'rectified')
        csv_path, n_regions, n_ok, n_fail, failures = EV.rectify_folder(indir, outdir)
        assert n_ok == 3, n_ok
        assert n_fail == 1, n_fail
        assert failures and failures[0][0] == 'zzz_corrupt.png'
        print(f'partial_failures_counted: OK (ok={n_ok}, fail={n_fail}, '
              f'reported: {failures[0]})')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gui_cut_rectify_folder_override_and_clear_cv2_error():
    import gui

    tmp = tempfile.mkdtemp()
    try:
        custom_dir = os.path.join(tmp, 'anywhere')
        _make_rect_images(custom_dir, 4)
        cfg = dict(gui.DEFAULTS)
        cfg.update(results_dir=os.path.join(tmp, 'results'), name='unused')
        vals = [cfg[k] for k in gui.CONFIG_KEYS]

        status, gallery = gui.cut_rectify('latest', 16, 0.5, 0.85, custom_dir, *vals)
        assert '4장 처리' in status or '4' in status, status
        assert len(gallery) == 4
        print('gui.cut_rectify with explicit folder override: OK')

        orig_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == 'cv2':
                raise ModuleNotFoundError("No module named 'cv2'")
            return orig_import(name, *a, **k)

        builtins.__import__ = fake_import
        try:
            status2, gallery2 = gui.cut_rectify('latest', 16, 0.5, 0.85, custom_dir, *vals)
        finally:
            builtins.__import__ = orig_import
        assert 'opencv' in status2.lower()
        assert len(gallery2) == 0
        assert '✅' not in status2   # must NOT look like a success message
        print('gui.cut_rectify surfaces missing-cv2 clearly (no misleading success): OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_processes_the_whole_folder_not_one_file()
    test_random_noise_folder_all_files_still_processed()
    test_missing_cv2_fails_loudly_not_silently()
    test_partial_failures_are_counted_not_swallowed()
    test_gui_cut_rectify_folder_override_and_clear_cv2_error()
    print('\nAll rectify smoke tests passed.')


if __name__ == '__main__':
    main()
