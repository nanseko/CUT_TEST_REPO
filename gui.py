""" Gradio Web-UI for the (attention-augmented) PyTorch CUT model.

Drives the official CUT PyTorch pipeline (train.py / test.py) from a browser and
works both on a normal PC and on Google Colab (a public share link is created
automatically on Colab). It covers the full SAR -> Optical workflow:

  0. Download the M4-SAR dataset and organise it into CUT layout (Colab only).
  1. Point at the dataset root and scan / count images.
  2. SAR preprocessing pipeline (ordered, editable steps) with before/after preview.
  3. Basic training parameters (epochs, lr, batch, ...), saved to gui_config.json.
  4. CUT parameters (netG, netF, NCE, structure/colour losses), saved.
  5. Attention modules (none / CBAM / Coordinate) toggled per position, saved.
  6. Launch training (subprocess `python train.py`) with a live log + epoch/iter/lr.
  7. Inference (subprocess `python test.py`) on a trained checkpoint + result gallery.

The preprocessing pipeline (Tab 2) and dataset utilities (Tab 0) are reused
verbatim from the original SAR-CUT project; the training/inference tabs were
re-wired to the PyTorch CUT codebase in this repository.

Launch:
    python gui.py                 # local, http://127.0.0.1:7860
    python gui.py --share         # force a public share link
On Colab just run `!python gui.py` (the share link is automatic).
"""

import os
import re
import sys
import json
import glob
import time
import signal
import argparse
import datetime
import threading
import traceback

import gradio as gr


# --------------------------------------------------------------------------- #
# Colab detection (only Colab may reach the external network for downloads)
# --------------------------------------------------------------------------- #

def _detect_colab():
    if 'google.colab' in sys.modules:
        return True
    if os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_GPU'):
        return True
    try:
        import importlib.util
        if importlib.util.find_spec('google.colab') is not None:
            return True
    except Exception:
        pass
    return False


IN_COLAB = _detect_colab()
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# Build marker — bump when the GUI changes so you can confirm the running file is
# up to date (printed on launch and shown in the UI header). If the version you
# see in the browser/console does not match the latest, you are running an old
# copy and must replace gui.py / preprocessing/.
BUILD = '2026-07-08.4 (system-wide-subprocess-watchdog)'


# --------------------------------------------------------------------------- #
# Configuration handling
# --------------------------------------------------------------------------- #

# ABSOLUTE path, anchored to gui.py's own location (NOT the process's current
# working directory). A relative path here caused settings (esp. dataroot /
# checkpoints_dir / results_dir) to silently "reset" on server restart: if
# gui.py is launched from a different working directory each time (a desktop
# shortcut, a batch file, Task Scheduler, ...), './gui_config.json' resolves
# to a different file every time, so the previous save is never found.
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, 'gui_config.json')

# kept as a literal (not imported from evaluation.EVAL_CSV_COLUMNS) so the GUI
# module can build its layout even if evaluation/ is temporarily mismatched;
# must stay in sync with evaluation/evaluate.py's EVAL_CSV_COLUMNS.
EVAL_TABLE_HEADERS = [
    'timestamp', 'experiment', 'checkpoint_epoch', 'n_fake', 'n_eo',
    'fid', 'kid', 'struct_epi', 'struct_cc', 'struct_psnr', 'n_struct_pairs',
    'idt_psnr', 'idt_ssim', 'n_idt_pairs',
    'quality_mean', 'quality_std', 'quality_speckle_index', 'quality_enl',
    'quality_avg_gradient', 'quality_entropy',
    'notes',
]

# Stable key order. The Gradio input list is assembled in exactly this order so
# every Save / Start button can collect the whole config consistently.
CONFIG_KEYS = [
    # 1. Dataset / output
    'dataroot', 'name', 'checkpoints_dir', 'results_dir', 'gpu_ids',
    # 3. Basic training params
    'CUT_mode', 'n_epochs', 'n_epochs_decay', 'batch_size', 'lr',
    'beta1', 'beta2', 'save_epoch_freq', 'load_size', 'crop_size', 'num_threads',
    'continue_train', 'max_dataset_size',
    # 4. CUT params
    'netG', 'normG', 'gan_mode', 'netF', 'netF_nc', 'num_patches', 'nce_T',
    'nce_layers', 'lambda_GAN', 'lambda_NCE', 'nce_idt',
    'no_antialias', 'no_antialias_up', 'lambda_grad', 'lambda_lap', 'grad_no_blur',
    'reflector_weighted', 'saliency_patch_sampling', 'reflector_boost', 'lambda_coherence',
    'lambda_color', 'serial_batches',
    # 5. Attention params
    'attention_type', 'attention_reduction',
    'attention_encoder', 'attention_resblocks', 'attention_decoder',
    # HRNet params (only used when netG == hrnet)
    'hrnet_branches', 'hrnet_modules', 'hrnet_blocks',
    # 8/9. Per-tab data-folder overrides (persisted like everything else above;
    # empty string = auto-derive from results_dir/name/test_<epoch> as before)
    'eval_eo_dir', 'eval_real_b_dir', 'eval_fake_dir', 'eval_real_a_dir',
    'rectify_input_dir',
]

DEFAULTS = {
    'dataroot': './datasets/M4-SAR-cut',
    'name': 'sar_cut',
    'checkpoints_dir': './checkpoints',
    'results_dir': './results',
    'gpu_ids': '0',
    'CUT_mode': 'CUT',
    'n_epochs': 200,
    'n_epochs_decay': 200,
    'batch_size': 1,
    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch_freq': 5,
    'load_size': 286,
    'crop_size': 256,
    'num_threads': 4,
    'continue_train': False,
    'max_dataset_size': 0,
    'netG': 'resnet_9blocks',
    'normG': 'instance',
    'gan_mode': 'lsgan',
    'netF': 'mlp_sample',
    'netF_nc': 256,
    'num_patches': 256,
    'nce_T': 0.07,
    'nce_layers': '0,4,8,12,16',
    'lambda_GAN': 1.0,
    'lambda_NCE': 1.0,
    'nce_idt': True,
    'no_antialias': False,
    'no_antialias_up': False,
    'lambda_grad': 0.0,
    'lambda_lap': 0.0,
    'grad_no_blur': False,
    'reflector_weighted': False,
    'saliency_patch_sampling': False,
    'reflector_boost': 3.0,
    'lambda_coherence': 0.0,
    'lambda_color': 0.0,
    'serial_batches': False,
    'attention_type': 'none',
    'attention_reduction': 16,
    'attention_encoder': False,
    'attention_resblocks': False,
    'attention_decoder': False,
    'hrnet_branches': 3,
    'hrnet_modules': 3,
    'hrnet_blocks': 2,
    'eval_eo_dir': './datasets/Optical/trainB',
    'eval_real_b_dir': '',
    'eval_fake_dir': '',
    'eval_real_a_dir': '',
    'rectify_input_dir': '',
}

IMAGE_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
IMAGE_EXTS_FLAT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def load_config(path=DEFAULT_CONFIG_PATH):
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                cfg.update(json.load(f))
        except Exception as exc:
            print(f'[gui] Failed to read config {path}: {exc}')
    return cfg


def save_config(cfg, path=DEFAULT_CONFIG_PATH):
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    return path


def list_images(folder):
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def _cfg_from_values(values):
    return dict(zip(CONFIG_KEYS, values))


def do_save(cfg_path, *values):
    cfg = _cfg_from_values(values)
    path = save_config(cfg, cfg_path or DEFAULT_CONFIG_PATH)
    return f'✅ 저장됨: {path}  ({datetime.datetime.now().strftime("%H:%M:%S")})'


# --------------------------------------------------------------------------- #
# Per-checkpoint hyperparameter snapshot (so switching between experiments and
# resuming each with ITS OWN settings doesn't require remembering/retyping
# them). This is separate from gui_config.json (the GUI's own "last used"
# settings) and from CUT's own <name>/train_opt.txt (a human-readable dump,
# not shaped to round-trip back into these widgets).
# --------------------------------------------------------------------------- #

CHECKPOINT_CONFIG_FILENAME = 'gui_train_config.json'

# keys that determine the network's actual weight SHAPES: if these differ from
# what a checkpoint was originally trained with, loading its state_dict for
# --continue_train will fail (or silently mismatch). See docs/HANDOFF.md §10.
ARCH_CRITICAL_KEYS = [
    'netG', 'normG', 'attention_type', 'attention_reduction',
    'attention_encoder', 'attention_resblocks', 'attention_decoder',
    'no_antialias', 'no_antialias_up',
    'hrnet_branches', 'hrnet_modules', 'hrnet_blocks',
]


def _checkpoint_config_path(checkpoints_dir, name):
    return os.path.join(str(checkpoints_dir), str(name), CHECKPOINT_CONFIG_FILENAME)


def save_checkpoint_config(cfg):
    """Snapshot the full hyperparameter set into checkpoints_dir/<name>/ at the
    moment training starts, so this exact experiment's settings can be
    restored later even after other checkpoints have been trained in between."""
    path = _checkpoint_config_path(cfg['checkpoints_dir'], cfg['name'])
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = dict(cfg)
        payload['_saved_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        payload['_build'] = BUILD
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path
    except Exception:
        return None


def load_checkpoint_config(checkpoints_dir, name):
    """Read back a previously-saved per-checkpoint config. None if absent/unreadable."""
    path = _checkpoint_config_path(checkpoints_dir, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def list_checkpoint_experiments(checkpoints_dir):
    """Names of experiment subfolders under checkpoints_dir: prefers ones with
    a gui_train_config.json, but also surfaces older checkpoints (trained
    before this feature existed, or via bare CLI) that only have *_net_G.pth,
    so the dropdown is never mysteriously empty for an existing project."""
    d = str(checkpoints_dir)
    if not os.path.isdir(d):
        return []
    names = []
    for entry in sorted(os.listdir(d)):
        sub = os.path.join(d, entry)
        if not os.path.isdir(sub):
            continue
        has_config = os.path.exists(os.path.join(sub, CHECKPOINT_CONFIG_FILENAME))
        has_ckpt = any(f.endswith('_net_G.pth') for f in os.listdir(sub))
        if has_config or has_ckpt:
            names.append(entry + ('' if has_config else '  (설정 로그 없음)'))
    return names


def arch_mismatch_warnings(old_cfg, new_cfg):
    """Compare architecture-critical keys; return a list of human-readable
    mismatch descriptions (empty if everything matches or old_cfg is None)."""
    if not old_cfg:
        return []
    out = []
    for k in ARCH_CRITICAL_KEYS:
        ov, nv = old_cfg.get(k), new_cfg.get(k)
        if str(ov) != str(nv):
            out.append(f'{k}: 체크포인트={ov!r} vs 현재 설정={nv!r}')
    return out


def do_scan(dataroot):
    """Scan a CUT dataroot for trainA/trainB/testA/testB image counts."""
    if not dataroot or not os.path.isdir(dataroot):
        return f'경로를 찾을 수 없습니다: {dataroot}'
    msgs = []
    for sub in ('trainA', 'trainB', 'testA', 'testB'):
        d = os.path.join(dataroot, sub)
        files = list_images(d)
        sample = ', '.join(os.path.basename(p) for p in files[:5])
        more = ' ...' if len(files) > 5 else ''
        status = f'{len(files)}개' if files else '없음/경로확인'
        msgs.append(f'• {sub} [{d}] : {status}  {sample}{more}')
    return '\n'.join(msgs)


def attention_all_on():
    return True, True, True


def attention_all_off():
    return False, False, False


# --------------------------------------------------------------------------- #
# Training state shared between the worker thread and the UI
# --------------------------------------------------------------------------- #

class TrainingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.reset()

    def reset(self):
        self.running = False
        self.stop_requested = False
        self.epoch = 0
        self.total_epochs = 0
        self.iters = 0
        self.lr = 0.0
        self.losses = {}
        self.message = '대기 중 (Idle)'
        self.logs = []
        self.log_file = None
        self.proc = None
        self.restarts = 0
        self.loss_png = None

    def log(self, text):
        stamp = datetime.datetime.now().strftime('%H:%M:%S')
        line = f'[{stamp}] {text}' if not text.startswith('[') else text
        with self.lock:
            self.logs.append(line)
            self.logs = self.logs[-500:]
            if self.log_file:
                try:
                    with open(self.log_file, 'a') as f:
                        f.write(line + '\n')
                except Exception:
                    pass
        print(line)

    def snapshot(self):
        with self.lock:
            return {
                'running': self.running,
                'epoch': self.epoch,
                'total_epochs': self.total_epochs,
                'iters': self.iters,
                'lr': self.lr,
                'losses': dict(self.losses),
                'message': self.message,
                'logs': '\n'.join(self.logs[-300:]),
                'loss_png': self.loss_png,
            }


STATE = TrainingState()

# regexes to parse the official CUT console output
_RE_ITER = re.compile(r'epoch:\s*(\d+),\s*iters:\s*(\d+)')
_RE_LR = re.compile(r'learning rate.*?=\s*([0-9.eE+-]+)')
_RE_LOSS = re.compile(r'(\w+):\s*(-?\d+\.\d+)')


# --------------------------------------------------------------------------- #
# Build CLI commands for the PyTorch CUT scripts
# --------------------------------------------------------------------------- #

def _bool(v):
    return bool(v) and str(v).lower() not in ('false', '0', 'none', '')


def _attention_args(cfg):
    args = ['--attention_type', str(cfg['attention_type']),
            '--attention_reduction', str(int(cfg['attention_reduction']))]
    if _bool(cfg['attention_encoder']):
        args.append('--attention_encoder')
    if _bool(cfg['attention_resblocks']):
        args.append('--attention_resblocks')
    if _bool(cfg['attention_decoder']):
        args.append('--attention_decoder')
    if _bool(cfg['no_antialias']):
        args.append('--no_antialias')
    if _bool(cfg['no_antialias_up']):
        args.append('--no_antialias_up')
    return args


def _find_last_epoch(checkpoints_dir, name):
    """Largest N from '<N>_net_G.pth' under checkpoints_dir/name (None if absent)."""
    d = os.path.join(str(checkpoints_dir), str(name))
    if not os.path.isdir(d):
        return None
    epochs = []
    for f in os.listdir(d):
        m = re.match(r'(\d+)_net_G\.pth$', f)
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


def _resume_args(cfg):
    """Build --continue_train args to resume from the last saved epoch.

    Returns (args, message). args is [] when resume was requested but no
    checkpoint exists (-> start fresh).
    """
    if not _bool(cfg.get('continue_train')):
        return [], None
    ckpt = str(cfg['checkpoints_dir'])
    name = str(cfg['name'])
    last = _find_last_epoch(ckpt, name)
    if last is not None:
        return (['--continue_train', '--epoch', str(last), '--epoch_count', str(last + 1)],
                f'이어서 학습: epoch {last} 체크포인트에서 재개 -> epoch {last+1} 부터')
    if os.path.exists(os.path.join(ckpt, name, 'latest_net_G.pth')):
        return (['--continue_train', '--epoch', 'latest', '--epoch_count', '1'],
                '이어서 학습: 번호 체크포인트가 없어 latest 가중치로 재개 (epoch 카운트는 1부터)')
    return [], '이어서 학습 요청됨 — 체크포인트를 찾지 못해 처음부터 학습합니다.'


def recommended_nce_layers(netG, attention_type, attention_encoder, no_antialias, hrnet_branches):
    """Pure-Python mirror of the generators' nce_default so the GUI can show the
    correct PatchNCE tap indices for the current config (editable by the user)."""
    if netG == 'hrnet':
        return ','.join(str(i) for i in range(2 + int(hrnet_branches or 3)))
    n_blocks = {'resnet_9blocks': 9, 'resnet_6blocks': 6, 'resnet_4blocks': 4}.get(netG, 9)
    use_attn = str(attention_type) not in ('none', 'None', '', 'NONE')
    n = [0]
    taps = {}

    def add():
        i = n[0]
        n[0] += 1
        return i

    taps['pixel'] = add()        # ReflectionPad2d(3)
    add(); add(); add()          # conv7, norm, relu
    if use_attn and _bool(attention_encoder):
        add()                    # encoder attention after stem
    for i in range(2):           # two downsampling stages
        taps['enc%d' % i] = add()  # downsample conv (the tap)
        add(); add()             # norm, relu
        if not _bool(no_antialias):
            add()                # Downsample
        if use_attn and _bool(attention_encoder):
            add()                # encoder attention
    for b in range(n_blocks):
        j = add()
        if b == 0:
            taps['res0'] = j
        if b == min(4, n_blocks - 1):
            taps['res4'] = j
    return ','.join(str(taps[k]) for k in ('pixel', 'enc0', 'enc1', 'res0', 'res4'))


def gui_recommend_nce(netG, attention_type, attention_encoder, no_antialias, hrnet_branches):
    return recommended_nce_layers(netG, attention_type, attention_encoder, no_antialias, hrnet_branches)


def build_train_cmd(cfg):
    cmd = [sys.executable, '-u', os.path.join(REPO_ROOT, 'train.py'),
           '--dataroot', str(cfg['dataroot']),
           '--name', str(cfg['name']),
           '--model', 'cut',
           '--CUT_mode', 'CUT' if str(cfg['CUT_mode']).lower() == 'cut' else 'FastCUT',
           '--checkpoints_dir', str(cfg['checkpoints_dir']),
           '--gpu_ids', str(cfg['gpu_ids']),
           '--n_epochs', str(int(cfg['n_epochs'])),
           '--n_epochs_decay', str(int(cfg['n_epochs_decay'])),
           '--batch_size', str(int(cfg['batch_size'])),
           '--lr', str(float(cfg['lr'])),
           '--beta1', str(float(cfg['beta1'])),
           '--beta2', str(float(cfg['beta2'])),
           '--save_epoch_freq', str(int(cfg['save_epoch_freq'])),
           '--load_size', str(int(cfg['load_size'])),
           '--crop_size', str(int(cfg['crop_size'])),
           '--num_threads', str(int(cfg['num_threads'])),
           '--netG', str(cfg['netG']),
           '--normG', str(cfg['normG']),
           '--gan_mode', str(cfg['gan_mode']),
           '--netF', str(cfg['netF']),
           '--netF_nc', str(int(cfg['netF_nc'])),
           '--num_patches', str(int(cfg['num_patches'])),
           '--nce_T', str(float(cfg['nce_T'])),
           '--nce_layers', str(cfg['nce_layers']),
           '--lambda_GAN', str(float(cfg['lambda_GAN'])),
           '--lambda_NCE', str(float(cfg['lambda_NCE'])),
           '--nce_idt', 'True' if _bool(cfg['nce_idt']) else 'False',
           '--lambda_grad', str(float(cfg['lambda_grad'])),
           '--lambda_lap', str(float(cfg.get('lambda_lap', 0.0))),
           '--reflector_boost', str(float(cfg.get('reflector_boost', 3.0))),
           '--lambda_coherence', str(float(cfg.get('lambda_coherence', 0.0))),
           '--lambda_color', str(float(cfg['lambda_color'])),
           '--display_id', '0']    # disable visdom; we stream the console log
    if str(cfg['netG']) == 'hrnet':
        cmd += ['--hrnet_branches', str(int(cfg.get('hrnet_branches', 3))),
                '--hrnet_modules', str(int(cfg.get('hrnet_modules', 3))),
                '--hrnet_blocks', str(int(cfg.get('hrnet_blocks', 2)))]
    if int(cfg.get('max_dataset_size', 0) or 0) > 0:
        # use only the first N files (sorted by name) from trainA/trainB
        cmd += ['--max_dataset_size', str(int(cfg['max_dataset_size']))]
    if _bool(cfg.get('grad_no_blur')):
        cmd.append('--grad_no_blur')
    if _bool(cfg.get('reflector_weighted')):
        cmd.append('--reflector_weighted')
    if _bool(cfg.get('saliency_patch_sampling')):
        cmd.append('--saliency_patch_sampling')
    if _bool(cfg.get('serial_batches')):
        # pair real_A[i] with real_B[i] by sorted order (for aligned SAR/optical
        # sets); default CUT samples real_B randomly (unpaired, by design).
        cmd.append('--serial_batches')
    cmd += _resume_args(cfg)[0]
    cmd += _attention_args(cfg)
    return cmd


def build_test_cmd(cfg, num_test, epoch):
    cmd = [sys.executable, '-u', os.path.join(REPO_ROOT, 'test.py'),
           '--dataroot', str(cfg['dataroot']),
           '--name', str(cfg['name']),
           '--model', 'cut',
           '--CUT_mode', 'CUT' if str(cfg['CUT_mode']).lower() == 'cut' else 'FastCUT',
           '--checkpoints_dir', str(cfg['checkpoints_dir']),
           '--results_dir', str(cfg['results_dir']),
           '--gpu_ids', str(cfg['gpu_ids']),
           '--load_size', str(int(cfg['crop_size'])),
           '--crop_size', str(int(cfg['crop_size'])),
           '--num_threads', '0',
           '--netG', str(cfg['netG']),
           '--normG', str(cfg['normG']),
           '--netF', str(cfg['netF']),
           '--netF_nc', str(int(cfg['netF_nc'])),
           '--nce_layers', str(cfg['nce_layers']),
           '--num_test', str(int(num_test)),
           '--epoch', str(epoch),
           '--phase', 'test']
    if str(cfg['netG']) == 'hrnet':
        cmd += ['--hrnet_branches', str(int(cfg.get('hrnet_branches', 3))),
                '--hrnet_modules', str(int(cfg.get('hrnet_modules', 3))),
                '--hrnet_blocks', str(int(cfg.get('hrnet_blocks', 2)))]
    cmd += _attention_args(cfg)
    return cmd


# --------------------------------------------------------------------------- #
# Training worker (subprocess + live console parsing + stall watchdog)
#
# Long (1-2 day) unattended runs can freeze without the process crashing (a
# stuck DataLoader worker, a GPU/driver hang, a stalled network-drive read,
# ...) — "정지되는 것은 아닌데 멈추는" symptom. This is independent of the
# browser: start_training() spawns this function on a daemon THREAD, so the
# subprocess keeps running regardless of whether anyone's browser tab / the
# network to it is connected — only the gui.py process itself needs to stay
# running (see docs/RESILIENT_TRAINING.md for what that needs at the OS level,
# e.g. disabling sleep). The watchdog below detects "no forward progress for
# N minutes" (not just "process exited") and auto-restarts with
# --continue_train from the last saved checkpoint, so a stall/crash only costs
# the time since the last checkpoint, not the whole run.
# --------------------------------------------------------------------------- #

# floor for the stall window regardless of the requested stall_minutes, so a
# mistakenly tiny value can't cause false-positive restarts during normal
# pauses (checkpoint saving, first-batch data_dependent_initialize, ...).
# Overridden by tests to verify the watchdog without waiting a full minute+.
MIN_STALL_SECONDS = 60


# regex matching any line that counts as "forward progress" for the stall
# watchdog: a print_freq iteration line, or the once-per-epoch summary line.
_PROGRESS_RE = re.compile(r'epoch:\s*\d+,\s*iters:\s*\d+|End of epoch')


def training_worker(cfg, state, stall_minutes=20, max_restarts=20, backoff_seconds=30):
    from util.subprocess_watchdog import run_watched
    attempt = 0
    try:
        with state.lock:
            state.total_epochs = int(cfg['n_epochs']) + int(cfg['n_epochs_decay'])
            state.message = '학습 중 (Training)'

        def on_line(line):
            m = _RE_ITER.search(line)
            if m:
                with state.lock:
                    state.epoch = int(m.group(1))
                    state.iters = int(m.group(2))
                    state.losses = {k: float(v) for k, v in
                                    _RE_LOSS.findall(line.split(')', 1)[-1])}
            ml = _RE_LR.search(line)
            if ml:
                with state.lock:
                    state.lr = float(ml.group(1))
            state.log(line)

        def on_start(proc):
            with state.lock:
                state.proc = proc

        while True:
            if state.stop_requested:
                state.log('사용자 요청으로 학습 중단됨')
                with state.lock:
                    state.message = '중단됨 (Stopped)'
                return

            run_cfg = cfg if attempt == 0 else dict(cfg, continue_train=True)
            cmd = build_train_cmd(run_cfg)
            tag = '' if attempt == 0 else f' [재시작 {attempt}/{max_restarts}]'
            state.log(f'실행 명령{tag}: ' + ' '.join(cmd))

            result = run_watched(cmd, REPO_ROOT, on_line, stall_minutes=stall_minutes,
                                 progress_re=_PROGRESS_RE, is_stopped=lambda: state.stop_requested,
                                 on_start=on_start, min_stall_seconds=MIN_STALL_SECONDS)
            stalled, rc = result['stalled'], result['rc']

            if result['stopped'] or state.stop_requested:
                state.log('사용자 요청으로 학습 중단됨')
                with state.lock:
                    state.message = '중단됨 (Stopped)'
                return

            if stalled:
                state.log(f'⚠️ {stall_minutes}분 동안 진행 없음(행/hang) 감지 → 프로세스를 강제 종료합니다.')

            if not stalled and rc == 0:
                state.log('학습 완료' + (f' (재시작 {attempt}회 후)' if attempt else ''))
                with state.lock:
                    state.message = '완료 (Done)'
                return

            # unexpected stop (stall or non-zero exit) -> auto-restart from checkpoint
            if not stalled:
                state.log(f'⚠️ 학습 프로세스가 코드 {rc} 로 비정상 종료되었습니다.')
            attempt += 1
            with state.lock:
                state.restarts = attempt
            if attempt > max_restarts:
                state.log(f'❌ 최대 재시작 횟수({max_restarts})를 초과했습니다. 자동 재시작을 중단합니다. '
                          f'원인(디스크/네트워크/전원 설정 등)을 확인 후 수동으로 다시 시작하세요.')
                with state.lock:
                    state.message = f'오류 (재시작 {max_restarts}회 초과)'
                return
            state.log(f'🔁 {backoff_seconds}초 후 마지막 체크포인트부터 자동 재시작합니다 '
                      f'(누적 재시작 {attempt}회).')
            with state.lock:
                state.message = f'재시작 대기 중... ({attempt}/{max_restarts})'
            waited = 0.0
            while waited < backoff_seconds:
                if state.stop_requested:
                    state.log('사용자 요청으로 학습 중단됨 (재시작 취소)')
                    with state.lock:
                        state.message = '중단됨 (Stopped)'
                    return
                time.sleep(min(2, backoff_seconds - waited))
                waited += 2
    except Exception:
        state.log('학습 중 예외 발생:\n' + traceback.format_exc())
        with state.lock:
            state.message = '오류 (Error)'
    finally:
        with state.lock:
            state.running = False
            state.proc = None


def _format_status(snap):
    ep = f"{snap['epoch']}/{snap['total_epochs']}"
    it = str(snap['iters'])
    lr = f"{snap['lr']:.7f}" if snap['lr'] else '-'
    loss_str = ', '.join(f'{k}={v:.4f}' for k, v in snap['losses'].items()) or '-'
    png = snap.get('loss_png')
    png = png if (png and os.path.exists(png)) else None   # only show once the first epoch has rendered it
    return ep, it, lr, snap['message'], loss_str, snap['logs'], png


def start_training(cfg_path, stall_minutes, max_restarts, *values):
    cfg = _cfg_from_values(values)
    save_config(cfg, cfg_path or DEFAULT_CONFIG_PATH)

    if STATE.running:
        yield _format_status(STATE.snapshot())
        return

    if not cfg['dataroot'] or not os.path.isdir(cfg['dataroot']):
        STATE.reset()
        STATE.log(f'오류: dataroot 폴더가 없습니다: {cfg["dataroot"]}')
        yield _format_status(STATE.snapshot())
        return

    log_dir = os.path.join(str(cfg['checkpoints_dir']), str(cfg['name']), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    STATE.reset()
    STATE.log_file = os.path.join(
        log_dir, f'gui_train_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
    # train.py refreshes this PNG at the end of every epoch (util/loss_plot.py)
    STATE.loss_png = os.path.join(str(cfg['checkpoints_dir']), str(cfg['name']), 'loss_curve.png')
    STATE.running = True
    STATE.message = '학습 준비 중...'

    # preflight: show exactly which folders will be loaded so a train/test or
    # A/B folder mix-up is obvious (training reads trainA/trainB).
    dataroot = str(cfg['dataroot'])
    counts = {sub: len(list_images(os.path.join(dataroot, sub)))
              for sub in ('trainA', 'trainB', 'testA', 'testB')}
    STATE.log(f'dataroot = {os.path.abspath(dataroot)}')
    STATE.log('폴더 이미지 수 -> ' + ', '.join(f'{k}={v}' for k, v in counts.items()))
    STATE.log(f'학습은 trainA({counts["trainA"]}) -> trainB({counts["trainB"]}) 를 사용합니다 '
              f'(testA/testB 는 추론용).')
    if counts['trainA'] == 0 or counts['trainB'] == 0:
        STATE.log('오류: trainA 또는 trainB 가 비어 있습니다. dataroot 아래 trainA/trainB 폴더를 확인하세요.')
        STATE.running = False
        STATE.message = '오류: 학습 폴더 비어 있음'
        yield _format_status(STATE.snapshot())
        return

    resume_msg = _resume_args(cfg)[1]
    if resume_msg:
        STATE.log(resume_msg)
    STATE.log(f'⚙️ 행(hang) 감시: {float(stall_minutes or 20):.0f}분 동안 진행 없으면 자동 재시작 '
              f'(최대 {int(max_restarts or 20)}회) — PC 절전/최대절전은 꺼두세요.')

    # architecture-mismatch guard: if resuming a checkpoint, warn loudly when
    # the CURRENT tab 4/5 settings differ from what this checkpoint was
    # actually trained with (netG/attention/hrnet mismatches make the saved
    # weights fail -- or silently mismatch -- to load). Not blocking, since a
    # deliberate architecture change is a valid (if unusual) thing to do.
    if _bool(cfg.get('continue_train')):
        prev_cfg = load_checkpoint_config(cfg['checkpoints_dir'], cfg['name'])
        mismatches = arch_mismatch_warnings(prev_cfg, cfg)
        if mismatches:
            STATE.log('⚠️⚠️ 아키텍처 설정이 이 체크포인트를 학습할 때와 다릅니다! '
                      '가중치 로드가 실패하거나 잘못될 수 있습니다:')
            for m in mismatches:
                STATE.log('   - ' + m)
            STATE.log('   → 탭 6의 "📂 체크포인트 설정 불러오기"로 원래 설정을 복원할 수 있습니다.')
        elif prev_cfg is not None:
            STATE.log('✅ 아키텍처 설정이 이전 체크포인트와 일치합니다.')

    # snapshot this run's full hyperparameter set next to the checkpoint, so it
    # can be restored later (탭 6의 "체크포인트 설정 불러오기") even after other
    # experiments have been trained in between.
    ckpt_cfg_path = save_checkpoint_config(cfg)
    if ckpt_cfg_path:
        STATE.log(f'설정 스냅샷 저장: {ckpt_cfg_path}')

    thread = threading.Thread(
        target=training_worker,
        args=(cfg, STATE), kwargs=dict(stall_minutes=float(stall_minutes or 20),
                                       max_restarts=int(max_restarts or 20)),
        daemon=True)
    thread.start()

    while True:
        snap = STATE.snapshot()
        yield _format_status(snap)
        if not snap['running']:
            break
        time.sleep(1.0)
    yield _format_status(STATE.snapshot())


def stop_training():
    if STATE.running:
        STATE.stop_requested = True
        with STATE.lock:
            proc = STATE.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        return '⏹️ 중단 요청됨. 프로세스를 종료합니다.'
    return 'ℹ️ 실행 중인 학습이 없습니다.'


def refresh_checkpoint_dropdown(checkpoints_dir):
    names = list_checkpoint_experiments(checkpoints_dir)
    return gr.update(choices=names, value=(names[0] if names else None))


def cfg_apply_checkpoint(checkpoints_dir, name):
    """Load <checkpoints_dir>/<name>/gui_train_config.json (written when that
    experiment's training was started) and restore it into every Tab 1-5
    widget, so switching between experiments and resuming each one uses ITS
    OWN original settings instead of whatever is currently on screen."""
    if not name:
        return ['실험을 먼저 선택하세요.'] + [gr.update() for _ in CONFIG_KEYS]
    real_name = name.split('  (설정 로그 없음)')[0]
    saved = load_checkpoint_config(checkpoints_dir, real_name)
    if not saved:
        return ([f'"{real_name}" 에는 설정 로그가 없습니다 (이 기능이 추가되기 전에 학습된 '
                f'체크포인트일 수 있습니다). checkpoints_dir/{real_name}/{CHECKPOINT_CONFIG_FILENAME} '
                f'없음 — 수동으로 설정하세요.']
               + [gr.update() for _ in CONFIG_KEYS])
    msg = (f'✅ "{real_name}" 체크포인트의 설정을 불러왔습니다 '
          f'(저장 시각: {saved.get("_saved_at", "?")}, build {saved.get("_build", "?")}). '
          f'이제 탭 6에서 "이어서 학습"을 체크하고 학습을 시작하면 이 설정으로 이어집니다.')
    # 'continue_train' is an action flag (was it a resume WHEN this snapshot was
    # taken), not a durable hyperparameter -- restoring it verbatim would often
    # silently uncheck "이어서 학습" (the snapshot from a first-ever run has it
    # False), contradicting the very reason the user is loading this config.
    # Leave whatever the user currently has checked untouched.
    skip = {'continue_train'}
    updates = [gr.update(value=saved[k]) if (k in saved and k not in skip) else gr.update()
              for k in CONFIG_KEYS]
    return [msg] + updates


# --------------------------------------------------------------------------- #
# Inference (subprocess test.py + result gallery)
# --------------------------------------------------------------------------- #

def run_inference(num_test, epoch, stall_minutes, *cfg_values):
    from util.subprocess_watchdog import run_watched_stream
    cfg = _cfg_from_values(cfg_values)
    gallery = []

    if not cfg['dataroot'] or not os.path.isdir(cfg['dataroot']):
        yield (f'오류: dataroot 폴더가 없습니다: {cfg["dataroot"]}', gallery)
        return
    if not os.path.isdir(os.path.join(cfg['dataroot'], 'testA')):
        yield (f'오류: {cfg["dataroot"]}/testA 가 없습니다. CUT 추론은 testA(+testB) 폴더가 필요합니다.', gallery)
        return

    epoch = str(epoch or 'latest').strip()
    cmd = build_test_cmd(cfg, int(num_test or 50), epoch)
    yield ('실행 명령: ' + ' '.join(cmd) + '\n추론 시작...', gallery)

    try:
        log_lines = []
        holder = {}
        for line in run_watched_stream(cmd, REPO_ROOT, holder, stall_minutes=float(stall_minutes or 20),
                                       min_stall_seconds=MIN_STALL_SECONDS):
            log_lines.append(line)
            if 'processing' in line or 'loading' in line or 'Error' in line or 'Traceback' in line:
                yield ('\n'.join(log_lines[-15:]), gallery)

        if holder.get('stalled'):
            yield ('\n'.join(log_lines[-15:]) +
                  f'\n\n⚠️ {stall_minutes}분 동안 진행 없어 추론 프로세스를 강제 종료했습니다 '
                  f'(DataLoader 정지/GPU 행 등 의심). 다시 시도해 보세요.', gallery)
            return

        out_dir = os.path.join(str(cfg['results_dir']), str(cfg['name']),
                               f'test_{epoch}', 'images', 'fake_B')
        if os.path.isdir(out_dir):
            gallery = list_images(out_dir)
        if gallery:
            yield (f'✅ 완료: {len(gallery)}장 변환 → {out_dir}', gallery[:24])
        else:
            yield ('\n'.join(log_lines[-20:]) +
                   f'\n\n(결과 이미지를 찾지 못했습니다: {out_dir})', gallery)
    except Exception:
        yield ('추론 중 예외 발생:\n' + traceback.format_exc(), gallery)


def cut_rectify(epoch, min_area, max_area_frac, min_rectangularity, input_dir_override,
                *cfg_values):
    """Deterministic post-processing: snap candidate rigid-object regions in
    fake_B (or any folder the user points at) to straight-sided rectangles/
    polygons (classical CV, guarantees exact geometry — unlike the learned
    coherence_loss). Defaults to the same test_<epoch> fake_B output as
    '7. 추론/테스트' / '8. 모델 평가', but `input_dir_override` lets the user
    analyse ANY folder (e.g. trainB, testB, a different results_dir) instead."""
    cfg = _cfg_from_values(cfg_values)
    input_dir = (str(input_dir_override).strip() if input_dir_override else '') or \
        os.path.join(str(cfg['results_dir']), str(cfg['name']), f'test_{epoch}', 'images', 'fake_B')
    if not os.path.isdir(input_dir):
        return (f'오류: 폴더가 없습니다: {input_dir}\n'
                f'(기본값은 "7. 추론/테스트" 결과의 fake_B 폴더입니다 — 먼저 추론을 실행하거나, '
                f'위 "입력 폴더 직접 지정"에 분석할 폴더를 지정하세요.)', [])
    out_dir = os.path.join(str(cfg['results_dir']), str(cfg['name']), f'test_{epoch}', 'rectified')
    try:
        import evaluation as EV
        csv_path, n, n_ok, n_fail, failures = EV.rectify_folder(
            input_dir, out_dir, min_area=float(min_area or 16),
            max_area_frac=float(max_area_frac or 0.25),
            min_rectangularity=float(min_rectangularity or 0.85))
        gallery = list_images(out_dir)
        if n_ok == 0 and n_fail > 0:
            detail = '\n'.join(f'  - {name}: {err}' for name, err in failures)
            return (f'❌ 처리 실패: {input_dir} 의 이미지 {n_fail}개 전부 실패했습니다.\n{detail}', [])
        msg = f'✅ 완료: {input_dir} · {n_ok}장 처리, 사각형 {n}개 검출 → {out_dir}\n좌표/크기: {csv_path}'
        if n_fail:
            detail = '\n'.join(f'  - {name}: {err}' for name, err in failures)
            msg += f'\n⚠️ {n_fail}장 실패:\n{detail}'
        return (msg, gallery[:24])
    except ImportError as exc:
        return f'오류: {exc}', []
    except Exception:
        return '후처리 중 예외:\n' + traceback.format_exc(), []


HPS_APPLY_KEYS = ['attention_type', 'attention_reduction', 'attention_encoder',
                  'attention_resblocks', 'attention_decoder', 'lambda_grad',
                  'lambda_lap', 'lambda_coherence', 'lambda_color',
                  'reflector_weighted', 'saliency_patch_sampling', 'reflector_boost']


def cut_hparam_search(out_dir, n_trials, s1_epochs, s2_epochs, top_k, s1_images,
                      s2_images, num_test, primary, eo_dir, incw, stall_minutes,
                      *cfg_values):
    """One-button Successive-Halving hyperparameter search over attention +
    structure/hallucination loss weights. Short trainings ranked by FID/EPI;
    winners get more budget via continue_train. Resumable (hparam_results.csv).
    Each trial's train.py/test.py runs under the same stall watchdog as the
    main training tab (util/subprocess_watchdog.py) -- a hung trial is killed
    and marked failed instead of blocking the entire multi-hour search."""
    cfg = _cfg_from_values(cfg_values)
    try:
        from evaluation.hparam_search import hparam_search
        for line in hparam_search(
                cfg, build_train_cmd, build_test_cmd,
                out_dir or './hparam_search',
                n_trials=int(n_trials or 12), stage1_epochs=int(s1_epochs or 15),
                stage2_epochs=int(s2_epochs or 45), top_k=int(top_k or 5),
                stage1_images=int(s1_images or 300), stage2_images=int(s2_images or 0),
                num_test=int(num_test or 100), primary=primary,
                eo_dir=(eo_dir or None), inception_weights=(incw or None),
                repo_root=REPO_ROOT, stall_minutes=float(stall_minutes or 20)):
            yield line
    except Exception:
        yield '하이퍼파라미터 탐색 중 예외:\n' + traceback.format_exc()


def hps_apply_best(out_dir):
    """Load best_hparams.json and return updates for the Tab-4/5 widgets."""
    try:
        from evaluation.hparam_search import load_best
        best = load_best(out_dir or './hparam_search')
    except Exception:
        best = None
    if not best:
        return ['best_hparams.json 을 찾을 수 없습니다 — 먼저 탐색을 실행하세요.'] + \
               [gr.update() for _ in HPS_APPLY_KEYS]
    ov = best.get('overrides', {})
    msg = (f'✅ 최적 설정을 탭 4/5에 적용했습니다 ({best.get("primary")}='
           f'{(best.get("metrics") or {}).get(best.get("primary"))}). '
           f'"💾 저장" 후 학습을 시작하세요.\n'
           + json.dumps(ov, sort_keys=True, ensure_ascii=False))
    return [msg] + [gr.update(value=ov[k]) if k in ov else gr.update()
                    for k in HPS_APPLY_KEYS]


def eval_table_rows(results_dir, name):
    """Read the logged evaluation rows for the GUI comparison table."""
    import evaluation as EV
    rows = EV.load_eval_log(results_dir, name)
    return [[r.get(c, '') for c in EV.EVAL_CSV_COLUMNS] for r in rows]


def cut_evaluate(epoch, experiment, notes, eo_dir, compute_identity, real_b_dir,
                 inception_weights, fid_max, quality_max, struct_max,
                 fake_dir_override, real_a_dir_override, *cfg_values):
    """One-button CUT output evaluation (FID/KID vs EO, structure vs real_A,
    optional identity-path vs G(real_B), no-reference quality). By default
    reads outputs already produced by '7. 추론/테스트' (test.py);
    fake_dir_override/real_a_dir_override let the user point at any other
    folder pair instead. Logs a comparison row under
    <results_dir>/<name>/eval_logs/eval_results.csv."""
    import evaluation as EV
    cfg = _cfg_from_values(cfg_values)
    epoch = (epoch or 'latest').strip() or 'latest'
    try:
        last = ''
        for line in EV.run_evaluation(
                results_dir=cfg['results_dir'], name=cfg['name'], epoch=epoch,
                experiment=(experiment or cfg['name']), notes=notes,
                eo_dir=(eo_dir or None), checkpoints_dir=cfg['checkpoints_dir'], cfg=cfg,
                compute_identity=bool(compute_identity), real_b_dir=(real_b_dir or None),
                inception_weights=(inception_weights or None),
                fid_max=int(fid_max or 500), quality_max=int(quality_max or 0),
                struct_max=int(struct_max or 0),
                fake_dir=(str(fake_dir_override).strip() or None) if fake_dir_override else None,
                real_a_dir=(str(real_a_dir_override).strip() or None) if real_a_dir_override else None):
            last = line
            yield last, eval_table_rows(cfg['results_dir'], cfg['name'])
    except Exception:
        yield ('평가 중 예외:\n' + traceback.format_exc(),
              eval_table_rows(cfg['results_dir'], cfg['name']))


# --------------------------------------------------------------------------- #
# M4-SAR dataset download (Colab only)  — reused from the SAR-CUT project
# --------------------------------------------------------------------------- #

M4SAR_REPO = 'wchao0601/m4-sar'
M4SAR_ZIP = 'M4-SAR.zip'


def summarize_extracted(target_dir, max_depth=2):
    if not os.path.isdir(target_dir):
        return '(추출 폴더가 없습니다.)'
    lines = []
    base = target_dir.rstrip(os.sep)
    for root, dirs, files in os.walk(base):
        depth = root[len(base):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        dirs.sort()
        imgs = [f for f in files if f.lower().endswith(IMAGE_EXTS_FLAT)]
        indent = '  ' * depth
        name = os.path.basename(root) or root
        lines.append(f'{indent}{name}/  (이미지 {len(imgs)}개, 전체 {len(files)}개)')
        if len(lines) > 200:
            lines.append('  ...(생략)')
            break
    return '\n'.join(lines)


def download_and_extract(repo_id, filename, target_dir, token, allow_non_colab):
    if not (IN_COLAB or allow_non_colab):
        yield ('⛔ 비활성화됨: Colab 환경이 아닙니다.\n'
               '사내망에서는 외부망 다운로드가 차단됩니다. '
               'Colab에서 실행하거나, 외부망이 가능한 환경이라면 '
               '"외부망 다운로드 강제 허용"을 체크하세요.')
        return

    import zipfile
    try:
        import requests
    except Exception:
        yield '오류: requests 패키지가 필요합니다. `pip install requests` 후 다시 시도하세요.'
        return

    repo_id = (repo_id or M4SAR_REPO).strip()
    filename = (filename or M4SAR_ZIP).strip()
    target_dir = (target_dir or './datasets/M4-SAR').strip()
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, filename)

    url = f'https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}'
    headers = {'Authorization': f'Bearer {token.strip()}'} if token and token.strip() else {}

    yield f'다운로드 시작\n  repo : {repo_id}\n  file : {filename}\n  url  : {url}\n  대상 : {target_dir}'
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
            if r.status_code in (401, 403):
                yield (f'접근 거부(HTTP {r.status_code}). gated/비공개 데이터셋이면 '
                       'HF 토큰을 입력하세요. (huggingface.co/settings/tokens)')
                return
            r.raise_for_status()
            total = int(r.headers.get('Content-Length', 0))
            done = 0
            t0 = time.time()
            last = 0.0
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last > 1.0:
                        last = now
                        spd = done / max(now - t0, 1e-6) / 1e6
                        if total:
                            pct = done / total * 100
                            yield (f'다운로드 중... {done/1e9:.2f} / {total/1e9:.2f} GB '
                                   f'({pct:.1f}%)  {spd:.1f} MB/s')
                        else:
                            yield f'다운로드 중... {done/1e9:.2f} GB  {spd:.1f} MB/s'
    except Exception as exc:
        yield f'다운로드 실패: {exc}'
        return

    yield f'다운로드 완료 ({done/1e9:.2f} GB). 압축 해제 중...'
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            n = len(names)
            for i, member in enumerate(names):
                z.extract(member, target_dir)
                if i % 1000 == 0:
                    yield f'압축 해제 중... {i}/{n}'
    except zipfile.BadZipFile:
        yield f'오류: 잘못된 zip 파일입니다 ({zip_path}). 다시 다운로드하세요.'
        return
    except Exception as exc:
        yield f'압축 해제 실패: {exc}'
        return

    tree = summarize_extracted(target_dir)
    yield ('✅ 완료. 아래 폴더 구조를 참고해 "탭 0 정리" 또는 "탭 1"에서 경로를 지정하세요.\n'
           f'추출 위치: {target_dir}\n\n{tree}')


def organize_m4sar_to_cut(source_root, out_dir, sar_kw, opt_kw, link_mode, test_ratio):
    """Reorganise an extracted dataset into CUT layout (trainA/trainB/testA/testB)."""
    import shutil
    import random

    if not source_root or not os.path.isdir(source_root):
        yield (f'오류: 소스 폴더가 없습니다: {source_root}', gr.update())
        return

    sar_keys = [k.strip().lower() for k in (sar_kw or '').split(',') if k.strip()]
    opt_keys = [k.strip().lower() for k in (opt_kw or '').split(',') if k.strip()]
    if not sar_keys or not opt_keys:
        yield ('SAR / Optical 키워드를 모두 입력하세요.', gr.update())
        return

    out_dir = (out_dir or './datasets/M4-SAR-cut').strip()

    yield ('소스 폴더 스캔 중...', gr.update())
    items = []
    for root, _, files in os.walk(source_root):
        rel = os.path.relpath(root, source_root).lower()
        for f in files:
            if not f.lower().endswith(IMAGE_EXTS_FLAT):
                continue
            hay = rel + '/' + f.lower()
            if any(k in hay for k in sar_keys):
                domain = 'A'
            elif any(k in hay for k in opt_keys):
                domain = 'B'
            else:
                continue
            split = 'test' if ('test' in hay or 'val' in hay or 'valid' in hay) else 'train'
            items.append((os.path.join(root, f), domain, split))

    if not items:
        yield ('분류된 이미지가 없습니다. SAR/Optical 키워드 또는 소스 경로를 확인하세요.', gr.update())
        return

    has_test = any(s == 'test' for _, _, s in items)
    try:
        test_ratio = float(test_ratio)
    except (TypeError, ValueError):
        test_ratio = 0.0
    if not has_test and test_ratio > 0:
        for dom in ('A', 'B'):
            idxs = [i for i, (_, d, _) in enumerate(items) if d == dom]
            random.shuffle(idxs)
            k = int(len(idxs) * test_ratio)
            for i in idxs[:k]:
                p, d, _ = items[i]
                items[i] = (p, d, 'test')

    dests = {key: os.path.join(out_dir, key)
             for key in ('trainA', 'trainB', 'testA', 'testB')}
    for d in dests.values():
        os.makedirs(d, exist_ok=True)

    use_copy = (link_mode == 'copy')
    counters = {}
    counts = {'trainA': 0, 'trainB': 0, 'testA': 0, 'testB': 0}
    total = len(items)
    for n, (src, domain, split) in enumerate(items):
        key = f'{split}{domain}'
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        dst = os.path.join(dests[key], f'{idx:06d}_{os.path.basename(src)}')
        try:
            if os.path.exists(dst) or os.path.islink(dst):
                pass
            elif use_copy:
                shutil.copy2(src, dst)
            else:
                os.symlink(os.path.abspath(src), dst)
            counts[key] += 1
        except OSError:
            try:
                shutil.copy2(src, dst)
                counts[key] += 1
            except Exception:
                pass
        if (n + 1) % 5000 == 0:
            yield (f'정리 중... {n+1}/{total}  {counts}', gr.update())

    summary = (f'✅ CUT 형식 정리 완료 ({"복사" if use_copy else "심볼릭 링크"})\n'
               f'출력 폴더(dataroot): {out_dir}\n'
               f'  trainA (SAR)     : {counts["trainA"]}장\n'
               f'  trainB (Optical) : {counts["trainB"]}장\n'
               f'  testA  (SAR)     : {counts["testA"]}장\n'
               f'  testB  (Optical) : {counts["testB"]}장\n'
               '아래 탭 1의 dataroot 가 자동으로 채워졌습니다.')
    yield (summary, gr.update(value=out_dir))


# --------------------------------------------------------------------------- #
# SAR preprocessing callbacks (reused verbatim from the SAR-CUT project)
# Pipeline is an ORDERED, editable list of steps held in a gr.State.
# --------------------------------------------------------------------------- #

def pp_default_steps():
    return [
        {'name': 'validate_image', 'enabled': True,
         'params': {'drop_empty': True, 'handle_nan': 'zero'}, 'label': 'validate'},
        {'name': 'sar_intensity_transform', 'enabled': True,
         'params': {'mode': 'log1p', 'eps': 1e-6}, 'label': 'intensity: log1p'},
        {'name': 'speckle_filter', 'enabled': True,
         'params': {'method': 'refined_lee', 'window_size': 7, 'enl': 'auto'},
         'label': 'speckle: refined_lee'},
        {'name': 'outlier_clipping', 'enabled': True,
         'params': {'min_percentile': 0.2, 'max_percentile': 99.8, 'ignore_zero': True},
         'label': 'clipping 0.2-99.8'},
        {'name': 'histogram_mapping', 'enabled': True,
         'params': {'mode': 'sar_only', 'bins': 1024, 'optical_reference_dir': None,
                    'clahe': {'enabled': False, 'clip_limit': 2.0, 'tile_grid_size': [8, 8]}},
         'label': 'histogram: sar_only'},
        {'name': 'resize_or_tile', 'enabled': True,
         'params': {'mode': 'resize', 'image_size': 256}, 'label': 'resize 256'},
        {'name': 'channel_adapter', 'enabled': True,
         'params': {'output_channels': 3}, 'label': 'channel 3ch'},
        {'name': 'normalize_for_cut', 'enabled': True,
         'params': {'output_range': 'uint8'}, 'label': 'normalize uint8'},
    ]


def _pp_short(params):
    keys = ('method', 'mode', 'window_size', 'enl', 'damping_factor', 'bm3d_sigma',
            'min_percentile', 'max_percentile', 'bins', 'image_size', 'output_channels')
    return ', '.join(f'{k}={params[k]}' for k in keys if k in params) or '-'


def _pp_rows(steps):
    return [[i + 1, s.get('label', s['name']), _pp_short(s['params'])]
            for i, s in enumerate(steps)]


def _speckle_params(method, window, enl_auto, enl_val, damping, sig_auto, sig_val):
    p = {'method': method}
    if method in ('lee', 'frost', 'refined_lee', 'gamma_map'):
        p['window_size'] = int(window)
        p['enl'] = 'auto' if enl_auto else float(enl_val)
    if method == 'frost':
        p['damping_factor'] = float(damping)
    if method == 'bm3d':
        p['bm3d_sigma'] = 'auto' if sig_auto else float(sig_val)
    return p


def _default_step(category):
    if category == 'speckle':
        return {'name': 'speckle_filter', 'enabled': True,
                'params': {'method': 'lee', 'window_size': 7, 'enl': 'auto'},
                'label': 'speckle: lee'}
    if category == 'intensity':
        return {'name': 'sar_intensity_transform', 'enabled': True,
                'params': {'mode': 'log1p', 'eps': 1e-6}, 'label': 'intensity: log1p'}
    if category == 'clipping':
        return {'name': 'outlier_clipping', 'enabled': True,
                'params': {'min_percentile': 0.2, 'max_percentile': 99.8, 'ignore_zero': True},
                'label': 'clipping 0.2-99.8'}
    if category == 'histogram':
        return {'name': 'histogram_mapping', 'enabled': True,
                'params': {'mode': 'sar_only', 'bins': 1024, 'optical_reference_dir': None,
                           'reference_cdf_path': None,
                           'clahe': {'enabled': False, 'clip_limit': 2.0, 'tile_grid_size': [8, 8]}},
                'label': 'histogram: sar_only'}
    if category == 'resize':
        return {'name': 'resize_or_tile', 'enabled': True,
                'params': {'mode': 'resize', 'image_size': 256}, 'label': 'resize 256'}
    if category == 'channel':
        return {'name': 'channel_adapter', 'enabled': True,
                'params': {'output_channels': 3}, 'label': 'channel 3ch'}
    if category == 'validate':
        return {'name': 'validate_image', 'enabled': True,
                'params': {'drop_empty': True, 'handle_nan': 'zero'}, 'label': 'validate'}
    if category == 'normalize':
        return {'name': 'normalize_for_cut', 'enabled': True,
                'params': {'output_range': 'uint8'}, 'label': 'normalize uint8'}
    raise ValueError(category)


def pp_add_category(steps, category, sel):
    steps = list(steps) + [_default_step(category)]
    return steps, _pp_rows(steps), len(steps) - 1


def pp_move_up(steps, sel):
    steps = list(steps)
    i = int(sel)
    if 0 < i < len(steps):
        steps[i - 1], steps[i] = steps[i], steps[i - 1]
        i -= 1
    return steps, _pp_rows(steps), i


def pp_move_down(steps, sel):
    steps = list(steps)
    i = int(sel)
    if 0 <= i < len(steps) - 1:
        steps[i + 1], steps[i] = steps[i], steps[i + 1]
        i += 1
    return steps, _pp_rows(steps), i


def pp_remove_sel(steps, sel):
    steps = list(steps)
    i = int(sel)
    if 0 <= i < len(steps):
        del steps[i]
    i = max(0, min(i, len(steps) - 1)) if steps else 0
    return steps, _pp_rows(steps), i


def pp_reset_steps():
    steps = pp_default_steps()
    return steps, _pp_rows(steps), 0


def pp_speckle_vis(method):
    win = method in ('lee', 'frost', 'refined_lee', 'gamma_map')
    damp = (method == 'frost')
    bm = (method == 'bm3d')
    return (gr.update(visible=win), gr.update(visible=win), gr.update(visible=win),
            gr.update(visible=damp), gr.update(visible=bm), gr.update(visible=bm))


def pp_on_select(steps, evt: gr.SelectData):
    row = 0
    try:
        row = int(evt.index[0]) if evt and evt.index is not None else 0
    except Exception:
        row = 0
    if not steps or row >= len(steps):
        return [gr.update()] * 27
    s = steps[row]
    name = s['name']
    p = s.get('params', {})

    method = p.get('method', 'lee')
    window = int(p.get('window_size', 7))
    enl = p.get('enl', 'auto')
    enl_auto = (enl == 'auto')
    enl_val = 10.0 if enl_auto else float(enl)
    damp = float(p.get('damping_factor', 2.0))
    sig = p.get('bm3d_sigma', 'auto')
    sig_auto = (sig == 'auto')
    sig_val = 0.1 if sig_auto else float(sig)
    intmode = p.get('mode', 'log1p') if name == 'sar_intensity_transform' else 'log1p'
    cmin = float(p.get('min_percentile', 0.2))
    cmax = float(p.get('max_percentile', 99.8))
    ign = bool(p.get('ignore_zero', True))
    histmode = p.get('mode', 'sar_only') if name == 'histogram_mapping' else 'sar_only'
    bins = int(p.get('bins', 1024))
    optref = p.get('optical_reference_dir') or ''
    refcdf = p.get('reference_cdf_path') or ''
    clahe = bool((p.get('clahe', {}) or {}).get('enabled', False))
    size = int(p.get('image_size', 256))
    ch = int(p.get('output_channels', 3))

    is_spk = (name == 'speckle_filter')
    is_int = (name == 'sar_intensity_transform')
    is_clip = (name == 'outlier_clipping')
    is_hist = (name == 'histogram_mapping')
    is_resize = (name == 'resize_or_tile')
    is_chan = (name == 'channel_adapter')
    win_v = method in ('lee', 'frost', 'refined_lee', 'gamma_map')
    damp_v = (method == 'frost')
    bm_v = (method == 'bm3d')

    title = f'편집 중: #{row + 1}  ·  {s.get("label", name)}'
    if name in ('validate_image', 'normalize_for_cut'):
        title += '  (이 스텝은 조절할 파라미터가 없습니다)'

    return [
        row,
        gr.update(visible=True),
        title,
        gr.update(visible=is_spk),
        gr.update(visible=is_int),
        gr.update(visible=is_clip),
        gr.update(visible=is_hist),
        gr.update(visible=is_resize),
        gr.update(visible=is_chan),
        gr.update(value=method),
        gr.update(value=window, visible=win_v),
        gr.update(value=enl_auto, visible=win_v),
        gr.update(value=enl_val, visible=win_v),
        gr.update(value=damp, visible=damp_v),
        gr.update(value=sig_auto, visible=bm_v),
        gr.update(value=sig_val, visible=bm_v),
        gr.update(value=intmode),
        gr.update(value=cmin),
        gr.update(value=cmax),
        gr.update(value=ign),
        gr.update(value=histmode),
        gr.update(value=bins),
        gr.update(value=optref),
        gr.update(value=clahe),
        gr.update(value=size),
        gr.update(value=ch),
        gr.update(value=refcdf),
    ]


def pp_apply(steps, sel, method, window, enl_auto, enl_val, damp, sig_auto, sig_val,
             intmode, cmin, cmax, ign, histmode, bins, optref, clahe, size, ch, refcdf):
    steps = list(steps)
    i = int(sel)
    if not (0 <= i < len(steps)):
        return steps, _pp_rows(steps)
    name = steps[i]['name']
    if name == 'speckle_filter':
        steps[i]['params'] = _speckle_params(method, window, enl_auto, enl_val,
                                             damp, sig_auto, sig_val)
        steps[i]['label'] = f'speckle: {method}'
    elif name == 'sar_intensity_transform':
        steps[i]['params'] = {'mode': intmode, 'eps': 1e-6}
        steps[i]['label'] = f'intensity: {intmode}'
    elif name == 'outlier_clipping':
        steps[i]['params'] = {'min_percentile': float(cmin), 'max_percentile': float(cmax),
                              'ignore_zero': bool(ign)}
        steps[i]['label'] = f'clipping {cmin}-{cmax}'
    elif name == 'histogram_mapping':
        steps[i]['params'] = {'mode': histmode, 'bins': int(bins),
                              'optical_reference_dir': (optref or None),
                              'reference_cdf_path': (refcdf or None),
                              'clahe': {'enabled': bool(clahe), 'clip_limit': 2.0,
                                        'tile_grid_size': [8, 8]}}
        steps[i]['label'] = f'histogram: {histmode}'
    elif name == 'resize_or_tile':
        steps[i]['params'] = {'mode': 'resize', 'image_size': int(size)}
        steps[i]['label'] = f'resize {int(size)}'
    elif name == 'channel_adapter':
        steps[i]['params'] = {'output_channels': int(ch)}
        steps[i]['label'] = f'channel {int(ch)}ch'
    return steps, _pp_rows(steps)


def _pp_config_from_steps(input_dir, output_dir, max_items, recursive, shuffle, steps, num_workers=1):
    return {
        'io': {'input_dir': input_dir, 'output_dir': output_dir,
               'max_items': int(max_items or 0), 'recursive': bool(recursive),
               'shuffle': bool(shuffle), 'seed': 42, 'save_format': 'png',
               'num_workers': max(1, int(num_workers or 1))},
        'pipeline': {'steps': [{'name': s['name'], 'enabled': s.get('enabled', True),
                                'params': s['params']} for s in steps]},
    }


# see DEFAULT_CONFIG_PATH's comment: must be absolute (anchored to gui.py's own
# location), not CWD-relative, or preprocessing folder settings silently
# "reset" whenever gui.py happens to be launched from a different directory.
PP_CONFIG_PATH = os.path.join(REPO_ROOT, 'preproc_config.json')


def pp_load_settings():
    if os.path.exists(PP_CONFIG_PATH):
        try:
            with open(PP_CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def pp_save_settings(steps, input_dir, output_dir, max_items, recursive, shuffle, num_workers=1):
    data = {'input_dir': input_dir, 'output_dir': output_dir,
            'max_items': int(max_items or 0), 'recursive': bool(recursive),
            'shuffle': bool(shuffle), 'num_workers': max(1, int(num_workers or 1)), 'steps': steps}
    try:
        with open(PP_CONFIG_PATH, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass


def pp_save_btn_fn(steps, input_dir, output_dir, max_items, recursive, shuffle, num_workers):
    pp_save_settings(steps, input_dir, output_dir, max_items, recursive, shuffle, num_workers)
    return f'✅ 전처리 설정 저장됨: {PP_CONFIG_PATH} ({datetime.datetime.now().strftime("%H:%M:%S")})'


def pp_preview(steps, input_dir, output_dir, max_items, recursive, shuffle, num_workers):
    import preprocessing as PP
    pp_save_settings(steps, input_dir, output_dir, max_items, recursive, shuffle, num_workers)
    if not steps:
        return None, None, '파이프라인에 스텝이 없습니다. 스텝을 추가하세요.'
    cfg = _pp_config_from_steps(input_dir, output_dir, max_items, recursive, shuffle, steps, num_workers)
    files = PP.scan_images(input_dir, bool(recursive), False, 42, 1)
    if not files:
        return None, None, '입력 폴더에 이미지가 없습니다.'
    try:
        before, after = PP.preprocess_single(cfg, files[0])
        return before, after, f'미리보기: {os.path.basename(files[0])}'
    except Exception:
        return None, None, '미리보기 오류:\n' + traceback.format_exc()


def pp_run(steps, input_dir, output_dir, max_items, recursive, shuffle, num_workers):
    # Feed the gallery in-memory numpy arrays (not file paths). Returning a local
    # file path to a gr.Gallery can raise in Gradio's output postprocess when the
    # file is outside the allowed paths — that exception fires OUTSIDE this
    # function, killing the stream right after the first preview appears (the
    # "1장만 처리되고 오류" symptom). Numpy arrays are encoded by Gradio itself,
    # so there is no path validation and it works on every platform/version.
    import numpy as np
    from PIL import Image
    import preprocessing as PP
    try:
        pp_save_settings(steps, input_dir, output_dir, max_items, recursive, shuffle, num_workers)
        if not steps:
            yield '파이프라인에 스텝이 없습니다.', []
            return
        cfg = _pp_config_from_steps(input_dir, output_dir, max_items, recursive, shuffle, steps, num_workers)
        cache = {}

        def to_arrays(paths):
            arrs = []
            for p in (paths or []):
                if p not in cache:
                    try:
                        cache[p] = np.asarray(Image.open(p).convert('RGB'))
                    except Exception:
                        cache[p] = None
                if cache[p] is not None:
                    arrs.append(cache[p])
            return arrs

        for log, prev in PP.run_pipeline(cfg):
            yield log, to_arrays(prev)
    except Exception:
        yield ('전처리 중 예외:\n' + traceback.format_exc(), [])


def pp_train_reference(optical_dir, save_path, bins, max_items):
    """Pre-train: build the optical histogram CDF and save it to .npy for preset use."""
    import preprocessing as PP
    if not optical_dir or not os.path.isdir(optical_dir):
        return f'오류: Optical 폴더가 없습니다: {optical_dir}'
    save_path = (save_path or './optical_hist.npy').strip()
    try:
        path, nbins = PP.save_reference_cdf(optical_dir, save_path,
                                            bins=int(bins or 1024), max_items=int(max_items or 0))
        return (f'✅ 사전 히스토그램 저장 완료: {os.path.abspath(path)}  (bins={nbins})\n'
                f'→ histogram_mapping 스텝을 모드 "preset" 으로 두고 '
                f'"사전 히스토그램 .npy 경로" 에 위 경로를 넣으면 SAR만으로 매핑됩니다.')
    except Exception:
        return '사전 히스토그램 학습 실패:\n' + traceback.format_exc()


def pp_optimize(sar_dir, out_dir, n1, n2, topk, primary, hist_mode, optical,
                eo_dir, use_fid, fid_max, inception_weights):
    """Resumable two-stage search for the best preprocessing step order (one button)."""
    import preprocessing as PP
    try:
        for line in PP.optimize_orders(
                sar_dir, out_dir or './datasets/_order_search',
                n_stage1=int(n1 or 200), n_stage2=int(n2 or 1000),
                top_k=int(topk or 10), primary=primary,
                hist_mode=hist_mode, optical_dir=(optical or None),
                eo_dir=(eo_dir or None), compute_fid=bool(use_fid),
                fid_max=int(fid_max or 500),
                inception_weights=(inception_weights or None)):
            yield line
    except Exception:
        yield '순서 최적화 중 예외:\n' + traceback.format_exc()


def pp_metrics(output_dir, max_items):
    """SAR preprocessing quality metrics averaged over <output_dir>/images.

    A per-image CSV + summary log is written under <output_dir>/metrics_logs/.
    """
    import preprocessing as PP
    img_dir = os.path.join(output_dir or '', 'images')
    save_dir = output_dir or '.'
    if not os.path.isdir(img_dir):
        # allow pointing directly at a folder of images too
        img_dir = output_dir or ''
        save_dir = output_dir or '.'
    try:
        res = PP.compute_dataset_metrics(img_dir, max_items=int(max_items or 0), save_dir=save_dir)
        out = PP.format_metrics(res)
        if res and res.get('saved'):
            out += f'\n\n📝 로그 저장: {os.path.abspath(res["saved"])}\n   (metrics_logs 폴더에 CSV/TXT/JSON)'
        return out
    except Exception:
        return '지표 계산 오류:\n' + traceback.format_exc()


def pp_export(output_dir, optical_dir, out_root, test_ratio, link_mode):
    import preprocessing as PP
    sar_dir = os.path.join(output_dir, 'images')
    try:
        return PP.export_cut_layout(sar_dir, out_root or './datasets/M4-SAR-cut',
                                    optical_dir or None, float(test_ratio), link_mode)
    except Exception:
        return 'export 오류:\n' + traceback.format_exc()


# --------------------------------------------------------------------------- #
# Build the Gradio UI
# --------------------------------------------------------------------------- #

def build_ui():
    cfg = load_config()
    comp = {}

    with gr.Blocks(title='CUT + Attention 학습 GUI (PyTorch)', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# CUT + Attention 학습 GUI (PyTorch)\n'
                    'SAR→Optical CUT 모델을 폴더 지정만으로 전처리·학습·추론합니다. '
                    '**모든 탭의 값은 입력할 때마다 자동 저장**되어 서버를 재시작해도 그대로 남습니다 '
                    '(`gui_config.json`). "저장" 버튼은 저장 시각을 명시적으로 확인하고 싶을 때만 누르면 됩니다.\n'
                    '학습/추론은 이 저장소의 `train.py` / `test.py` 를 실행합니다.')

        gr.Markdown(f'<sub>build: <code>{BUILD}</code> · gradio {getattr(gr, "__version__", "?")}</sub>')

        cfg_path = gr.Textbox(value=DEFAULT_CONFIG_PATH, label='설정 파일 경로 (config json)')
        autosave_status = gr.Textbox(label='자동 저장 상태', interactive=False, value='(값을 변경하면 자동 저장됩니다)')

        env_txt = ('🟢 Colab 환경 감지됨 — 데이터셋 다운로드 사용 가능'
                   if IN_COLAB else
                   '🔒 비-Colab 환경 — 외부망 차단 가정으로 데이터셋 다운로드 기본 비활성화')
        gr.Markdown(f'**실행 환경:** {env_txt}')

        # ---- Tab 0 : dataset download + organize ----------------------- #
        with gr.Tab('0. 데이터셋 다운로드 / 정리 (M4-SAR)'):
            gr.Markdown(
                'HuggingFace `wchao0601/m4-sar` 에서 **M4-SAR.zip** 을 받아 압축을 풉니다.\n\n'
                '- **Colab 환경에서만** 기본 활성화됩니다.\n'
                '- 사내망(비-Colab)에서는 외부망 차단을 가정해 비활성화됩니다.')
            ds_override = gr.Checkbox(
                value=False,
                label='외부망 다운로드 강제 허용 (사내망에서는 체크하지 마세요)',
                visible=not IN_COLAB)
            with gr.Row():
                ds_repo = gr.Textbox(M4SAR_REPO, label='HF dataset repo_id')
                ds_file = gr.Textbox(M4SAR_ZIP, label='zip 파일명')
            ds_target = gr.Textbox('./datasets/M4-SAR', label='압축 해제 대상 폴더')
            ds_token = gr.Textbox('', label='HF 토큰 (gated/비공개일 때만)', type='password')
            ds_btn = gr.Button('⬇️ 다운로드 + 압축 해제', variant='primary', interactive=IN_COLAB)
            ds_out = gr.Textbox(label='진행 상황 / 결과', lines=12, interactive=False)

            ds_override.change(
                lambda v: gr.update(interactive=(IN_COLAB or bool(v))),
                inputs=ds_override, outputs=ds_btn)
            ds_btn.click(download_and_extract,
                         inputs=[ds_repo, ds_file, ds_target, ds_token, ds_override],
                         outputs=ds_out)

            gr.Markdown('---\n### CUT 형식으로 정리 (trainA/trainB/testA/testB)\n'
                        '추출된 폴더를 SAR=Source(A), Optical=Target(B)로 자동 분류해 '
                        'CUT 학습 폴더 구조(dataroot)로 만듭니다.')
            with gr.Row():
                org_src = gr.Textbox('./datasets/M4-SAR', label='정리할 소스(추출) 폴더')
                org_out = gr.Textbox('./datasets/M4-SAR-cut', label='CUT dataroot 출력 폴더')
            with gr.Row():
                org_sar_kw = gr.Textbox('sar,vh,vv', label='SAR(Source/A) 키워드')
                org_opt_kw = gr.Textbox('optical,opt,rgb,vis,visible', label='Optical(Target/B) 키워드')
            with gr.Row():
                org_mode = gr.Radio(['symlink', 'copy'], value='symlink',
                                    label='파일 처리 (대용량은 symlink 권장)')
                org_ratio = gr.Number(0.1, label='test 폴더 없을 때 분리 비율 (0=안함)')
            org_btn = gr.Button('🗂️ CUT 형식으로 정리', variant='primary')
            org_out_box = gr.Textbox(label='정리 결과', lines=8, interactive=False)

        # ---- Tab 1 : dataroot + scan ----------------------------------- #
        with gr.Tab('1. 데이터셋 (dataroot)'):
            comp['dataroot'] = gr.Textbox(cfg['dataroot'],
                                          label='dataroot — trainA/trainB/testA/testB 를 포함하는 폴더')
            with gr.Row():
                comp['name'] = gr.Textbox(cfg['name'], label='실험 이름 (name)')
                comp['checkpoints_dir'] = gr.Textbox(cfg['checkpoints_dir'], label='체크포인트 폴더')
            with gr.Row():
                comp['results_dir'] = gr.Textbox(cfg['results_dir'], label='추론 결과 폴더 (results_dir)')
                comp['gpu_ids'] = gr.Textbox(cfg['gpu_ids'], label='gpu_ids (예: 0 / 0,1 / -1=CPU)')
            scan_btn = gr.Button('📂 dataroot 스캔 (trainA/trainB/testA/testB 개수)')
            scan_out = gr.Textbox(label='스캔 결과', lines=5, interactive=False)
            scan_btn.click(do_scan, inputs=[comp['dataroot']], outputs=scan_out)

        # ---- Tab 2 : SAR preprocessing --------------------------------- #
        with gr.Tab('2. SAR 전처리 (학습 전)'):
            import preprocessing as PP
            gr.Markdown(
                'CUT 학습 **전에** SAR 이미지를 전처리합니다. 전처리 스텝을 '
                '**원하는 순서로 추가/이동/삭제**하고, 미리보기로 확인한 뒤 실행하세요. '
                '설계: `docs/README_pipeline.md`')

            _pps = pp_load_settings()
            _pp_steps0 = _pps.get('steps') or pp_default_steps()

            with gr.Accordion('① 폴더 / 데이터', open=True):
                pp_in = gr.Textbox(_pps.get('input_dir', './datasets/M4-SAR/raw_sar'), label='입력 SAR 폴더')
                pp_out = gr.Textbox(_pps.get('output_dir', './datasets/M4-SAR-preprocessed'), label='출력 폴더')
                with gr.Row():
                    pp_max = gr.Number(_pps.get('max_items', 20), label='처리 개수 (0=전체)', precision=0)
                    pp_recursive = gr.Checkbox(_pps.get('recursive', True), label='하위 폴더 포함')
                    pp_shuffle = gr.Checkbox(_pps.get('shuffle', False), label='섞기(shuffle)')
                    pp_workers = gr.Number(_pps.get('num_workers', 1),
                                           label='병렬 처리 수 num_workers (1=순차, CPU 코어수 권장)', precision=0)
                with gr.Row():
                    pp_save_btn = gr.Button('💾 전처리 설정 저장 (폴더/순서 보존)')
                    pp_save_msg = gr.Textbox(label='', interactive=False)

            with gr.Accordion('② 전처리 순서 만들기', open=True):
                gr.Markdown(
                    '1) **추가할 전처리** 종류를 고르고 `➕ 추가` → 맨 아래 #으로 생성됩니다.\n'
                    '2) 표에서 **행(#)을 클릭**하면 선택되고, 아래 **편집 패널**이 열립니다.\n'
                    '3) 선택한 #을 `⬆/⬇` 로 이동, `🗑` 로 삭제합니다.')
                pp_steps = gr.State(_pp_steps0)
                pp_sel = gr.State(0)
                pp_table = gr.Dataframe(
                    headers=['#', '스텝', '파라미터'], datatype=['number', 'str', 'str'],
                    value=_pp_rows(_pp_steps0), interactive=False, wrap=True,
                    label='현재 파이프라인 (위→아래 순서로 실행 · 행 클릭해 선택/편집)')
                with gr.Row():
                    pp_addcat = gr.Dropdown(
                        ['speckle', 'intensity', 'clipping', 'histogram', 'resize',
                         'channel', 'validate', 'normalize'],
                        value='speckle', label='추가할 전처리 (상위 메뉴)')
                    pp_add_btn = gr.Button('➕ 추가', variant='primary')
                with gr.Row():
                    pp_up_btn = gr.Button('⬆ 위로')
                    pp_down_btn = gr.Button('⬇ 아래로')
                    pp_rm_btn = gr.Button('🗑 선택 삭제')
                    pp_reset_btn = gr.Button('↺ 기본 순서로')

            with gr.Group(visible=False) as pp_edit_panel:
                pp_edit_title = gr.Markdown('편집')
                with gr.Group(visible=False) as g_spk:
                    e_method = gr.Dropdown(PP.SPECKLE_METHODS, value='lee', label='speckle 필터 종류')
                    with gr.Row():
                        e_window = gr.Number(7, label='window_size', precision=0)
                        e_enlauto = gr.Checkbox(True, label='ENL auto')
                        e_enlval = gr.Number(10, label='ENL 값')
                    with gr.Row():
                        e_damp = gr.Number(2.0, label='Frost damping_factor', visible=False)
                        e_sigauto = gr.Checkbox(True, label='BM3D sigma auto', visible=False)
                        e_sigval = gr.Number(0.1, label='BM3D sigma 값', visible=False)
                with gr.Group(visible=False) as g_int:
                    e_intmode = gr.Dropdown(PP.INTENSITY_MODES, value='log1p', label='intensity mode')
                with gr.Group(visible=False) as g_clip:
                    with gr.Row():
                        e_cmin = gr.Number(0.2, label='clip min %')
                        e_cmax = gr.Number(99.8, label='clip max %')
                        e_ign = gr.Checkbox(True, label='0값 제외')
                with gr.Group(visible=False) as g_hist:
                    with gr.Row():
                        e_histmode = gr.Dropdown(PP.HISTOGRAM_MODES, value='sar_only',
                                                 label='histogram 모드 (sar_only / unpaired_optical_reference / preset)')
                        e_bins = gr.Number(1024, label='bins', precision=0)
                        e_clahe = gr.Checkbox(False, label='CLAHE')
                    e_optref = gr.Textbox('', label='Optical 참조 폴더 (unpaired_optical_reference 모드)')
                    e_refcdf = gr.Textbox('', label='사전 히스토그램 .npy 경로 (preset 모드: 아래 ④에서 먼저 학습/저장)')
                with gr.Group(visible=False) as g_resize:
                    e_size = gr.Number(256, label='resize image_size', precision=0)
                with gr.Group(visible=False) as g_chan:
                    e_ch = gr.Number(3, label='출력 채널', precision=0)
                pp_apply_btn = gr.Button('✔ 적용', variant='primary')

            edit_widgets = [e_method, e_window, e_enlauto, e_enlval, e_damp, e_sigauto,
                            e_sigval, e_intmode, e_cmin, e_cmax, e_ign, e_histmode,
                            e_bins, e_optref, e_clahe, e_size, e_ch, e_refcdf]
            edit_groups = [g_spk, g_int, g_clip, g_hist, g_resize, g_chan]

            pp_add_btn.click(pp_add_category, inputs=[pp_steps, pp_addcat, pp_sel],
                             outputs=[pp_steps, pp_table, pp_sel])
            pp_up_btn.click(pp_move_up, inputs=[pp_steps, pp_sel],
                            outputs=[pp_steps, pp_table, pp_sel])
            pp_down_btn.click(pp_move_down, inputs=[pp_steps, pp_sel],
                              outputs=[pp_steps, pp_table, pp_sel])
            pp_rm_btn.click(pp_remove_sel, inputs=[pp_steps, pp_sel],
                            outputs=[pp_steps, pp_table, pp_sel])
            pp_reset_btn.click(pp_reset_steps, outputs=[pp_steps, pp_table, pp_sel])
            pp_table.select(pp_on_select, inputs=[pp_steps],
                            outputs=[pp_sel, pp_edit_panel, pp_edit_title] + edit_groups + edit_widgets)
            e_method.change(pp_speckle_vis, inputs=e_method,
                            outputs=[e_window, e_enlauto, e_enlval, e_damp, e_sigauto, e_sigval])
            pp_apply_btn.click(pp_apply, inputs=[pp_steps, pp_sel] + edit_widgets,
                               outputs=[pp_steps, pp_table])

            pp_io_inputs = [pp_steps, pp_in, pp_out, pp_max, pp_recursive, pp_shuffle, pp_workers]
            pp_save_btn.click(pp_save_btn_fn, inputs=pp_io_inputs, outputs=pp_save_msg)
            # auto-save: editing the folder/scan fields immediately persists the
            # full preprocessing config (same rationale as the main auto-save).
            for _pp_widget in (pp_in, pp_out, pp_max, pp_recursive, pp_shuffle, pp_workers):
                _pp_widget.change(pp_save_btn_fn, inputs=pp_io_inputs, outputs=pp_save_msg)

            with gr.Accordion('④ Optical 사전 히스토그램 학습/저장 (preset 모드용)', open=True):
                gr.Markdown(
                    'N장의 **Optical 이미지 폴더**로 히스토그램(CDF)을 미리 학습해 `.npy` 로 저장합니다. '
                    '이후 **SAR 이미지만 있어도** histogram_mapping 모드를 `preset` 으로 두고 이 파일을 참조하면 '
                    'SAR 히스토그램을 Optical 분포에 맞춥니다.')
                with gr.Row():
                    pp_tr_opt = gr.Textbox('./datasets/Optical/trainB', label='Optical 이미지 폴더 (학습용 N장)')
                    pp_tr_save = gr.Textbox('./optical_hist.npy', label='저장 경로 (.npy)')
                with gr.Row():
                    pp_tr_bins = gr.Number(1024, label='bins', precision=0)
                    pp_tr_max = gr.Number(0, label='사용 개수 (0=전체)', precision=0)
                    pp_tr_btn = gr.Button('🧠 사전 히스토그램 학습/저장', variant='primary')
                pp_tr_msg = gr.Textbox(label='결과', lines=4, interactive=False)
                pp_tr_btn.click(pp_train_reference,
                                inputs=[pp_tr_opt, pp_tr_save, pp_tr_bins, pp_tr_max],
                                outputs=pp_tr_msg)

            with gr.Accordion('⑤ 미리보기 (Before / After)', open=True):
                pp_prev_btn = gr.Button('🔍 첫 이미지 미리보기')
                with gr.Row():
                    pp_before = gr.Image(label='Before (원본 SAR)', type='numpy')
                    pp_after = gr.Image(label='After (전처리)', type='numpy')
                pp_prev_msg = gr.Textbox(label='', interactive=False)
                pp_prev_btn.click(pp_preview, inputs=pp_io_inputs,
                                  outputs=[pp_before, pp_after, pp_prev_msg])

            with gr.Accordion('⑥ 실행 / Export', open=True):
                pp_run_btn = gr.Button('▶ 전처리 실행', variant='primary')
                pp_log = gr.Textbox(label='로그', lines=10, interactive=False, max_lines=10)
                pp_gallery = gr.Gallery(label='Before|After 미리보기', columns=3, height='auto')
                pp_run_btn.click(pp_run, inputs=pp_io_inputs, outputs=[pp_log, pp_gallery])

                gr.Markdown('---\n**CUT dataroot 로 export** (전처리 결과 → trainA/testA, optical → trainB/testB)')
                with gr.Row():
                    pp_exp_opt = gr.Textbox('', label='Optical 폴더 (trainB/testB용, 선택)')
                    pp_exp_root = gr.Textbox('./datasets/M4-SAR-cut', label='출력 dataroot')
                with gr.Row():
                    pp_exp_ratio = gr.Number(0.1, label='test 비율')
                    pp_exp_link = gr.Radio(['symlink', 'copy'], value='symlink', label='파일 처리')
                pp_exp_btn = gr.Button('🗂️ CUT dataroot export')
                pp_exp_msg = gr.Textbox(label='export 결과', lines=4, interactive=False)
                pp_exp_btn.click(pp_export,
                                 inputs=[pp_out, pp_exp_opt, pp_exp_root, pp_exp_ratio, pp_exp_link],
                                 outputs=pp_exp_msg)

            with gr.Accordion('⑦ 전처리 성능 지표 (output 평균)', open=True):
                gr.Markdown(
                    '위 **① 출력 폴더**의 `images/` 에 대해 SAR 전처리 품질 지표를 이미지 평균으로 계산합니다.\n'
                    '- **Speckle Index(σ/μ)**: 낮을수록 스페클↓ · **ENL((μ/σ)²)**: 높을수록 스페클 억제↑\n'
                    '- **Avg Gradient(선명도)** / **Entropy(정보량)**: 높을수록 디테일 유지 · **Mean/Std**: 밝기/대비')
                with gr.Row():
                    pp_met_max = gr.Number(0, label='평가 개수 (0=전체)', precision=0)
                    pp_met_btn = gr.Button('📊 성능 지표 계산', variant='primary')
                pp_met_out = gr.Textbox(label='성능 지표 결과', lines=10, interactive=False)
                pp_met_btn.click(pp_metrics, inputs=[pp_out, pp_met_max], outputs=pp_met_out)

            with gr.Accordion('⑧ 전처리 순서 자동 최적화 (버튼 하나 · 재개 가능)', open=False):
                gr.Markdown(
                    '의미있는 4스텝(intensity · speckle · clipping · histogram)을 **전수 순열(24) × speckle(5) = 120 후보**로, '
                    '이미지 평가지표(PSNR·CC·EPI, 원본 SAR 기준)로 **2단계 평가**(stage1 랭킹 → 상위 K개 stage2 재평가)하여 '
                    '최적 순서를 자동 선정합니다.\n'
                    '- validate=맨앞, resize→channel→normalize=맨뒤로 고정됩니다.\n'
                    '- **중단해도** 결과 CSV에 기록된 완료 순서는 다시 실행하지 않습니다(재개 가능).\n'
                    '- 결과: `<결과폴더>/order_search_results.csv` (순서·speckle·지표), `best_pipeline.json` (최적 순서).')
                with gr.Row():
                    opt_sar = gr.Textbox('./datasets/M4-SAR/raw_sar', label='SAR 입력 폴더')
                    opt_out = gr.Textbox('./datasets/_order_search', label='결과/로그 폴더')
                with gr.Row():
                    opt_n1 = gr.Number(200, label='stage1 평가 장수', precision=0)
                    opt_n2 = gr.Number(1000, label='stage2 평가 장수', precision=0)
                    opt_topk = gr.Number(10, label='stage2 상위 K개', precision=0)
                with gr.Row():
                    opt_primary = gr.Dropdown(['fid', 'composite', 'epi', 'enl', 'speckle_index', 'psnr', 'cc'],
                                              value='fid', label='랭킹 기준 지표 (SAR→EO는 fid 권장)')
                    opt_hist = gr.Dropdown(PP.HISTOGRAM_MODES, value='sar_only', label='histogram 모드')
                    opt_optical = gr.Textbox('', label='histogram용 Optical 폴더/.npy (unpaired/preset 시)')
                gr.Markdown('**FID (EO 도메인 근접도, 낮을수록 좋음)** — SAR→EO 성능과 가장 직접적인 참고치입니다. '
                            'stage2 상위 후보에 대해서만 EO 세트 대비 계산합니다. (torch+torchvision 필요, 첫 실행 시 InceptionV3 가중치 다운로드)')
                with gr.Row():
                    opt_eo = gr.Textbox('./datasets/Optical/trainB', label='EO(광학) 참조 폴더 (FID 기준)')
                    opt_use_fid = gr.Checkbox(True, label='FID 계산 (stage2)')
                    opt_fid_max = gr.Number(500, label='FID 평가 장수 (EO/후보 각각)', precision=0)
                opt_incw = gr.Textbox(
                    '', label='InceptionV3 가중치 .pth 경로 (오프라인/사내망용, 비우면 자동탐색·캐시·다운로드)')
                gr.Markdown('오프라인이면 `inception_v3_google-0cc3c7bd.pth` 를 미리 받아 위 경로에 지정하거나, '
                            '`weights/` 폴더에 두거나, 환경변수 `INCEPTION_WEIGHTS` 로 지정하세요. (자세한 방법은 docs)')
                opt_btn = gr.Button('🚀 순서 자동 최적화 실행', variant='primary')
                opt_log = gr.Textbox(label='최적화 진행/결과 로그', lines=18, interactive=False, max_lines=18)
                opt_btn.click(pp_optimize,
                              inputs=[opt_sar, opt_out, opt_n1, opt_n2, opt_topk,
                                      opt_primary, opt_hist, opt_optical,
                                      opt_eo, opt_use_fid, opt_fid_max, opt_incw],
                              outputs=opt_log)

        # ---- Tab 3 : Basic training params ----------------------------- #
        with gr.Tab('3. 기본 학습 파라미터'):
            train_dataroot_mirror = gr.Textbox(
                cfg['dataroot'],
                label='학습 데이터 폴더 (dataroot) — 탭 1과 동일한 값. 여기서 바꿔도 됩니다.')
            gr.Markdown('ℹ️ 탭 1에서 이미 지정했다면 그대로 두세요. 이 필드를 바꾸면 탭 1의 값도 함께 갱신·저장됩니다 '
                       '(반대로 탭 1에서 바꾼 값은 페이지를 새로고침해야 여기 표시에 반영됩니다).')
            train_dataroot_mirror.change(lambda v: gr.update(value=v),
                                         inputs=[train_dataroot_mirror], outputs=[comp['dataroot']])
            with gr.Row():
                comp['CUT_mode'] = gr.Dropdown(['CUT', 'FastCUT'], value=cfg['CUT_mode'], label='CUT_mode')
                comp['n_epochs'] = gr.Number(cfg['n_epochs'], label='n_epochs (고정 lr)', precision=0)
                comp['n_epochs_decay'] = gr.Number(cfg['n_epochs_decay'], label='n_epochs_decay (lr 감쇠)', precision=0)
            with gr.Row():
                comp['batch_size'] = gr.Number(cfg['batch_size'], label='batch_size', precision=0)
                comp['lr'] = gr.Number(cfg['lr'], label='learning rate')
                comp['save_epoch_freq'] = gr.Number(cfg['save_epoch_freq'], label='save_epoch_freq', precision=0)
            with gr.Row():
                comp['beta1'] = gr.Number(cfg['beta1'], label='beta1')
                comp['beta2'] = gr.Number(cfg['beta2'], label='beta2')
                comp['num_threads'] = gr.Number(cfg['num_threads'], label='num_threads', precision=0)
            with gr.Row():
                comp['load_size'] = gr.Number(cfg['load_size'], label='load_size', precision=0)
                comp['crop_size'] = gr.Number(cfg['crop_size'], label='crop_size', precision=0)
                comp['max_dataset_size'] = gr.Number(
                    cfg['max_dataset_size'], precision=0,
                    label='학습 사용 개수 max_dataset_size (0=전체, 예: 2000/5000)')
            comp['continue_train'] = gr.Checkbox(
                bool(cfg['continue_train']),
                label='이어서 학습 (continue_train) — 마지막 저장된 epoch 체크포인트에서 재개')
            gr.Markdown(
                'ℹ️ 체크 시 `checkpoints_dir/name` 의 마지막 `<N>_net_*.pth` 를 불러와 '
                'epoch N+1 부터 이어서 학습합니다(설정은 학습 때와 동일해야 함). '
                '체크포인트는 `save_epoch_freq` epoch마다 저장됩니다.')
            save_basic = gr.Button('💾 기본 파라미터 저장', variant='primary')
            save_basic_out = gr.Textbox(label='', interactive=False)

        # ---- Tab 4 : CUT params ---------------------------------------- #
        with gr.Tab('4. CUT 파라미터'):
            with gr.Row():
                comp['netG'] = gr.Dropdown(['resnet_9blocks', 'resnet_6blocks', 'resnet_4blocks', 'hrnet'],
                                           value=cfg['netG'],
                                           label='netG (hrnet = 고해상도 보존, 강반사체 블러↓)')
                comp['normG'] = gr.Dropdown(['instance', 'batch', 'none'], value=cfg['normG'], label='normG')
                comp['gan_mode'] = gr.Dropdown(['lsgan', 'nonsaturating', 'vanilla'], value=cfg['gan_mode'], label='gan_mode')
            with gr.Row():
                comp['netF'] = gr.Dropdown(['mlp_sample', 'sample', 'reshape'], value=cfg['netF'], label='netF')
                comp['netF_nc'] = gr.Number(cfg['netF_nc'], label='netF_nc', precision=0)
                comp['num_patches'] = gr.Number(cfg['num_patches'], label='num_patches', precision=0)
            with gr.Row():
                comp['nce_T'] = gr.Number(cfg['nce_T'], label='nce_T (temperature)')
                comp['nce_idt'] = gr.Checkbox(bool(cfg['nce_idt']), label='nce_idt')
            with gr.Row():
                comp['nce_layers'] = gr.Textbox(
                    cfg['nce_layers'],
                    label='nce_layers (PatchNCE tap, 쉼표구분 · 직접 지정하면 그대로 사용)')
                nce_reco_btn = gr.Button('🔢 현재 설정 권장값 계산')
            gr.Markdown(
                'ℹ️ PatchNCE tap은 직접 지정할 수 있습니다. netG/attention/no_antialias 설정을 바꾼 뒤 '
                '**권장값 계산**을 누르면 현재 구조에 맞는 인덱스가 채워집니다(이후 자유롭게 편집). '
                '비워두거나 기본값(0,4,8,12,16)이면 학습 시 자동 보정됩니다. '
                'hrnet은 0 ~ (branches+1) 범위만 유효합니다.')
            with gr.Accordion('HRNet 전용 옵션 (netG=hrnet일 때만 적용)', open=False):
                with gr.Row():
                    comp['hrnet_branches'] = gr.Number(cfg['hrnet_branches'], precision=0,
                                                       label='branches (병렬 해상도 스트림 수, 2~4)')
                    comp['hrnet_modules'] = gr.Number(cfg['hrnet_modules'], precision=0,
                                                      label='modules (융합 모듈 수 = 깊이)')
                    comp['hrnet_blocks'] = gr.Number(cfg['hrnet_blocks'], precision=0,
                                                     label='blocks (브랜치당 residual 블록 수 = 폭)')
                gr.Markdown('branches↑/modules↑/blocks↑ → 표현력·디테일↑, 메모리·시간↑. '
                            'branches를 바꾸면 PatchNCE tap 수도 바뀌니 위 **권장값 계산**을 다시 누르세요.')
            with gr.Row():
                comp['lambda_GAN'] = gr.Number(cfg['lambda_GAN'], label='lambda_GAN')
                comp['lambda_NCE'] = gr.Number(cfg['lambda_NCE'], label='lambda_NCE')
            with gr.Row():
                comp['lambda_grad'] = gr.Number(cfg['lambda_grad'], label='lambda_grad (구조/에지 보존)')
                comp['lambda_lap'] = gr.Number(cfg['lambda_lap'], label='lambda_lap (고주파/라플라시안, 블러↓)')
                comp['lambda_color'] = gr.Number(cfg['lambda_color'], label='lambda_color (색 일관성, nce_idt 필요)')
            comp['grad_no_blur'] = gr.Checkbox(
                bool(cfg['grad_no_blur']),
                label='grad_no_blur (구조 손실에서 입력 블러 끔 → 더 날카로운 에지 타깃)')
            gr.Markdown(
                'ℹ️ 강반사체 주변 블러가 심하면: `netG=hrnet` + `lambda_grad`(예 1.0) + `lambda_lap`(예 0.5) + '
                '`grad_no_blur` 체크를 함께 써보세요. 너무 강하면 결과가 SAR처럼 밋밋해질 수 있으니 값으로 조절하세요.')
            gr.Markdown(
                '**소형 물체(요트·탱크·건물 등) 형상 보존** — 균일 평균 손실/균일 패치샘플링은 이미지의 '
                '극히 일부만 차지하는 강반사체(금속 물체) 를 거의 감독하지 못해 구름·블롭으로 뭉개지기 쉽습니다. '
                '아래 두 옵션은 SAR 입력의 국소 밝기 피크(강반사체=물체 후보)에 가중치를 줘서 이 문제를 직접 완화합니다.')
            with gr.Row():
                comp['reflector_weighted'] = gr.Checkbox(
                    bool(cfg['reflector_weighted']),
                    label='reflector_weighted (lambda_grad/lambda_lap을 강반사체 위치에서 더 강하게)')
                comp['saliency_patch_sampling'] = gr.Checkbox(
                    bool(cfg['saliency_patch_sampling']),
                    label='saliency_patch_sampling (PatchNCE 패치 샘플링을 강반사체 쪽으로 편향)')
                comp['reflector_boost'] = gr.Number(
                    cfg['reflector_boost'], label='reflector_boost (가중 강도, 0=끔)')
            comp['lambda_coherence'] = gr.Number(
                cfg['lambda_coherence'],
                label='lambda_coherence (강반사체 위치를 "구름형 블롭" 대신 "직선 에지"로 유도, 0=끔)')
            gr.Markdown(
                'ℹ️ `lambda_coherence` 는 구조텐서 기반으로 강반사체 위치의 출력이 등방성 블롭(구름처럼 뭉개짐) '
                '대신 국소적으로 뚜렷한 직선 에지를 갖도록 유도합니다. **90도 직각을 보장하지는 않습니다** '
                '— 자세한 원리와 한계는 docs/SMALL_OBJECT_PRESERVATION.md 참고. 0.3~1.0 부터 시작해보세요.')
            with gr.Row():
                comp['no_antialias'] = gr.Checkbox(bool(cfg['no_antialias']), label='no_antialias (다운샘플 stride2)')
                comp['no_antialias_up'] = gr.Checkbox(bool(cfg['no_antialias_up']), label='no_antialias_up')
            comp['serial_batches'] = gr.Checkbox(
                bool(cfg['serial_batches']),
                label='serial_batches (정렬된 짝 데이터: real_A[i]↔real_B[i] 같은 순번 사용. '
                      '끄면 CUT 기본=real_B 무작위/비짝)')
            gr.Markdown(
                'ℹ️ CUT는 **비짝(unpaired)** 학습이라 기본적으로 real_A(SAR)와 real_B(optical)는 '
                '서로 다른 이미지를 참조하는 것이 정상입니다. SAR→optical 변환 결과는 **fake_B** 입니다. '
                'SAR/optical 파일명이 1:1로 정렬된 짝 데이터라면 위 `serial_batches` 를 켜서 같은 순번끼리 묶을 수 있습니다.')
            save_cut = gr.Button('💾 CUT 파라미터 저장', variant='primary')
            save_cut_out = gr.Textbox(label='', interactive=False)

        # ---- Tab 5 : Attention ----------------------------------------- #
        with gr.Tab('5. Attention 설정'):
            comp['attention_type'] = gr.Radio(['none', 'cbam', 'coord', 'eca', 'self', 'cbam_coord'],
                                              value=cfg['attention_type'],
                                              label='Attention 종류 (none = 완전 OFF)')
            gr.Markdown(
                '- **cbam**: 채널+공간 attention (범용) · **coord**: 방향성(H/W) 위치 인코딩\n'
                '- **eca**: 경량 채널 attention (파라미터 거의 없음, 안정적)\n'
                '- **self**: non-local self-attention — 이미지 전역 픽셀 관계를 모델링해 '
                '**건물/선박의 전체 형태 일관성** 보존에 유리. 단 메모리 비용이 커서 '
                '**resblocks 위치(저해상도)에만** 삽입 권장(encoder/decoder 체크 시 고해상도라 무거움).\n'
                '- **cbam_coord**: 하이브리드 — CBAM 적용 후 Coordinate Attention을 이어서 적용(직렬)')
            comp['attention_reduction'] = gr.Number(cfg['attention_reduction'],
                                                    label='attention_reduction (bottleneck 축소비)', precision=0)
            gr.Markdown('**적용 위치 On/Off** — 개별 토글하거나 아래 버튼으로 모두 켜고 끌 수 있습니다. '
                        'attention 을 켜면 PatchNCE tap(`nce_layers`)이 자동으로 보정됩니다.')
            with gr.Row():
                comp['attention_encoder'] = gr.Checkbox(bool(cfg['attention_encoder']), label='Encoder')
                comp['attention_resblocks'] = gr.Checkbox(bool(cfg['attention_resblocks']), label='ResBlocks')
                comp['attention_decoder'] = gr.Checkbox(bool(cfg['attention_decoder']), label='Decoder')
            with gr.Row():
                all_on = gr.Button('모두 ON')
                all_off = gr.Button('모두 OFF')
            save_att = gr.Button('💾 Attention 설정 저장', variant='primary')
            save_att_out = gr.Textbox(label='', interactive=False)

            all_on.click(attention_all_on, outputs=[comp['attention_encoder'],
                                                    comp['attention_resblocks'],
                                                    comp['attention_decoder']])
            all_off.click(attention_all_off, outputs=[comp['attention_encoder'],
                                                      comp['attention_resblocks'],
                                                      comp['attention_decoder']])

        # fill nce_layers with the recommended taps for the current architecture
        nce_reco_btn.click(
            gui_recommend_nce,
            inputs=[comp['netG'], comp['attention_type'], comp['attention_encoder'],
                    comp['no_antialias'], comp['hrnet_branches']],
            outputs=comp['nce_layers'])

        # Organize button (Tab 0) auto-fills the dataroot on completion.
        org_btn.click(organize_m4sar_to_cut,
                      inputs=[org_src, org_out, org_sar_kw, org_opt_kw, org_mode, org_ratio],
                      outputs=[org_out_box, comp['dataroot']])

        # ---- Tab 6 : Train & Monitor ----------------------------------- #
        with gr.Tab('6. 학습 실행 / 모니터링'):
            gr.Markdown('현재 설정으로 `train.py` 를 실행합니다. 시작 시 모든 탭의 값이 저장됩니다.')
            gr.Markdown(
                '📂 **체크포인트 설정 불러오기** — 여러 실험(체크포인트)을 오가며 학습할 때, 학습 시작 시점에 '
                '`checkpoints_dir/<이름>/gui_train_config.json` 로 그 실험의 전체 설정이 자동 저장됩니다. '
                '다른 체크포인트로 전환했다가 이 실험을 다시 이어서 학습하려면 아래에서 불러오세요 — '
                '**특히 netG/attention/hrnet 설정이 어긋나면 가중치 로드가 실패**하므로, "이어서 학습" 전에 '
                '항상 먼저 불러오는 걸 권장합니다.')
            with gr.Row():
                ckpt_picker = gr.Dropdown(choices=list_checkpoint_experiments(cfg['checkpoints_dir']),
                                          label='체크포인트(실험) 선택', interactive=True)
                ckpt_refresh_btn = gr.Button('🔄 목록 새로고침')
                ckpt_load_btn = gr.Button('📂 이 설정 불러오기', variant='secondary')
            ckpt_load_msg = gr.Textbox(label='불러오기 결과', interactive=False)
            ckpt_refresh_btn.click(refresh_checkpoint_dropdown, inputs=[comp['checkpoints_dir']],
                                   outputs=[ckpt_picker])
            gr.Markdown(
                'ℹ️ **행(hang) 자동 복구** — 장시간(1~2일) 학습 중 프로세스가 죽지 않은 채 멈추는 경우'
                '(DataLoader 정지, GPU/드라이버 행, 네트워크 드라이브 I/O 정지 등)를 대비해, 아래 시간 동안'
                ' 진행(iters/epoch 로그)이 없으면 **자동으로 프로세스를 종료하고 마지막 체크포인트부터 재시작**'
                '합니다. 웹페이지/브라우저 연결이 끊겨도 학습 자체(백그라운드 프로세스)는 계속 진행됩니다 — '
                '단, **PC가 절전/최대절전 모드로 들어가면 안 됩니다** (자세한 OS 설정은 docs/RESILIENT_TRAINING.md).')
            with gr.Row():
                stall_minutes = gr.Number(20, label='행(hang) 감지 시간 (분)', precision=0)
                max_restarts = gr.Number(20, label='최대 자동 재시작 횟수', precision=0)
            with gr.Row():
                start_btn = gr.Button('▶ 학습 시작', variant='primary')
                stop_btn = gr.Button('⏹ 중단', variant='stop')
            with gr.Row():
                st_epoch = gr.Textbox(label='Epoch', interactive=False)
                st_iters = gr.Textbox(label='Iters (epoch 내)', interactive=False)
                st_lr = gr.Textbox(label='현재 학습률 (lr)', interactive=False)
                st_msg = gr.Textbox(label='상태', interactive=False)
            st_loss = gr.Textbox(label='현재 손실', interactive=False)
            st_log = gr.Textbox(label='로그 (Log)', lines=18, interactive=False, max_lines=18)
            gr.Markdown(
                '📈 **epoch별 손실 그래프** — 매 epoch 끝마다 자동 갱신되어 '
                '`checkpoints_dir/<name>/loss_curve.png` (+ `loss_history.csv`)로 저장됩니다. '
                '위: D / G / NCE 주요 곡선, 아래: 세부 손실. D가 0으로 붕괴하거나 G_GAN이 발산하면 '
                '학습 불균형 신호입니다. 400 epoch 완료 시 최종 그래프가 체크포인트에 남습니다.')
            st_graph = gr.Image(label='손실 곡선 (loss_curve.png)', interactive=False, type='filepath')

            monitor_outputs = [st_epoch, st_iters, st_lr, st_msg, st_loss, st_log, st_graph]
            stop_btn.click(stop_training, outputs=st_msg)

        # ---- Tab 7 : Inference ----------------------------------------- #
        with gr.Tab('7. 추론 / 테스트'):
            gr.Markdown(
                '학습된 체크포인트로 `test.py` 를 실행해 `dataroot/testA` 이미지를 변환합니다.\n\n'
                '- ⚠️ **탭 4/5의 CUT·Attention 설정이 학습 때와 동일**해야 가중치가 올바르게 로드됩니다.\n'
                '- CUT(unaligned)은 testA 와 testB 폴더가 모두 필요합니다.\n'
                '- 진행이 멈추면(DataLoader 정지 등) 아래 시간 후 자동으로 프로세스를 종료합니다 '
                '(재시작은 안 함 — 다시 "▶ 추론 실행"을 눌러 재시도하세요).')
            with gr.Row():
                inf_num = gr.Number(50, label='num_test (변환 장수)', precision=0)
                inf_epoch = gr.Textbox('latest', label='epoch (latest 또는 숫자)')
                inf_stall = gr.Number(20, label='행(hang) 감지 시간 (분)', precision=0)
            inf_btn = gr.Button('▶ 추론 실행', variant='primary')
            inf_status = gr.Textbox(label='진행 상황', lines=6, interactive=False)
            inf_gallery = gr.Gallery(label='변환 결과 (fake_B)', columns=4, height='auto')


        # ---- Tab 8 : Model evaluation (CUT outputs) --------------------- #
        with gr.Tab('8. 모델 평가 (CUT 출력)'):
            gr.Markdown(
                '**"7. 추론/테스트"로 생성된 결과**(`results_dir/name/test_<epoch>/images/{fake_B,real_A,real_B}`)'
                ' 를 평가합니다. 백본(ResNet/HRNet)·attention·lambda 값을 바꿀 때마다 **실험명**을 다르게 주고 '
                '실행하면 아래 비교표에 누적되어 설정 간 비교가 됩니다.\n\n'
                '- **FID / KID** (↓ 낮을수록 좋음): fake_B ↔ EO 참조 세트. SAR→EO 도메인 근접도(핵심 지표).\n'
                '- **EPI / CC / PSNR** (↑): real_A ↔ fake_B. 구조·에지 보존(허상 가드레일).\n'
                '- **idt PSNR / SSIM** (↑): real_B ↔ idt_B=G(real_B) — 짝 데이터라 정확한 비교. 백본 변경 시 유용.\n'
                '- **품질 지표**: fake_B 단독(선명도·대비·정보량), 무참조.')
            with gr.Row():
                eval_epoch = gr.Textbox('latest', label='epoch (테스트에 사용한 값과 동일하게)')
                eval_exp = gr.Textbox('', label='실험명 (비워두면 name 사용, 예: hrnet_coord_lgrad1.0)')
            eval_notes = gr.Textbox('', label='메모 (선택)')
            with gr.Row():
                comp['eval_eo_dir'] = gr.Textbox(cfg['eval_eo_dir'], label='EO(광학) 참조 폴더 (FID/KID 기준)')
                eval_incw = gr.Textbox('', label='InceptionV3 가중치 .pth 경로 (오프라인용, 비우면 자동탐색)')
            with gr.Row():
                eval_id_on = gr.Checkbox(True, label='Identity 평가(idt_B 생성, 탭4/5 설정과 동일한 체크포인트 사용)')
                comp['eval_real_b_dir'] = gr.Textbox(
                    cfg['eval_real_b_dir'], label='real_B 폴더 (비우면 test 결과의 real_B 자동 사용)')
            with gr.Row():
                comp['eval_fake_dir'] = gr.Textbox(
                    cfg['eval_fake_dir'], label='fake_B 폴더 직접 지정 (비우면 results_dir/name/test_<epoch> 자동)')
                comp['eval_real_a_dir'] = gr.Textbox(
                    cfg['eval_real_a_dir'], label='real_A 폴더 직접 지정 (비우면 자동)')
            with gr.Row():
                eval_fid_max = gr.Number(500, label='FID/KID 평가 장수', precision=0)
                eval_struct_max = gr.Number(0, label='구조 평가 장수 (0=전체)', precision=0)
                eval_qual_max = gr.Number(0, label='품질 평가 장수 (0=전체)', precision=0)
            eval_btn = gr.Button('▶ 평가 실행', variant='primary')
            eval_log = gr.Textbox(label='평가 진행/결과 로그', lines=14, interactive=False, max_lines=14)
            gr.Markdown('**실험 비교표** (실행할 때마다 누적, `eval_results.csv`)')
            eval_table = gr.Dataframe(headers=EVAL_TABLE_HEADERS, wrap=True,
                                      label='실험별 평가 결과 비교')
            eval_refresh = gr.Button('🔄 비교표 새로고침')

        # ---- Tab 9 : Deterministic rectification (post-processing) ------ #
        with gr.Tab('9. 형상 후처리 (직사각 스냅)'):
            gr.Markdown(
                '**결정론적 후처리** — `fake_B`에서 강체 물체 후보 영역을 검출해 최소외접회전사각형'
                '(`cv2.minAreaRect`)/단순화 다각형으로 스냅합니다. 학습된 `lambda_coherence`(탭 4)와 달리 '
                '**90도 직각을 기하학적으로 보장**하지만, 사실적인 텍스처가 아니라 벡터 도형처럼 보입니다. '
                '사진 같은 결과가 아니라 **형상 추출/판독**(건물 footprint, 선박 크기·방위)이 목적일 때 사용하세요.\n\n'
                '- 원형/불규칙 블롭(원형도가 낮은 객체)은 자동으로 제외됩니다 (사각형이 아닌 걸 억지로 사각형화하지 않음).\n'
                '- ⚠️ opencv가 설치되어 있어야 동작합니다(`pip install opencv-python`). 미설치 시 명확한 오류 메시지가 표시됩니다.')
            comp['rectify_input_dir'] = gr.Textbox(
                cfg['rectify_input_dir'],
                label='입력 폴더 직접 지정 (비우면 기본값: results_dir/name/test_<epoch>/images/fake_B)')
            with gr.Row():
                rect_epoch = gr.Textbox('latest', label='epoch (입력 폴더를 직접 지정하면 무시됨)')
            with gr.Row():
                rect_min_area = gr.Number(16, label='최소 영역 크기 (px², 노이즈 제거)')
                rect_max_frac = gr.Number(0.25, label='최대 영역 비율 (이미지 대비, 배경 제외)')
                rect_min_rectangularity = gr.Number(
                    0.85, label='최소 사각형도 (0~1, 원은 이론상 최대 0.785)')
            rect_btn = gr.Button('▶ 형상 후처리 실행', variant='primary')
            rect_status = gr.Textbox(label='결과', lines=6, interactive=False)
            rect_gallery = gr.Gallery(label='검출된 사각형 오버레이 (초록=사각형, 빨강=단순화 다각형)',
                                      columns=4, height='auto')

        # ---- Tab 10 : Hyperparameter auto-search (Successive Halving) --- #
        with gr.Tab('10. 하이퍼파라미터 자동 탐색'):
            gr.Markdown(
                '**Attention(type/위치/reduction) + 구조/허상 손실 가중치**(lambda_grad/lap/coherence/color, '
                'reflector_boost, 가중/샘플링 옵션)를 자동으로 탐색합니다.\n\n'
                '**방식 (Successive Halving)** — GAN 학습 손실은 품질 지표가 아니므로, 짧은 학습 N개를 돌려 '
                '**FID(EO 대비)·EPI** 로 랭킹하고, 상위 K개만 **이어학습**으로 예산을 더 줘 재평가합니다.\n'
                '- 탭 1~5의 현재 설정(dataroot/netG/crop 등)이 **기본값**이 되고, 탐색 대상 파라미터만 trial마다 바뀝니다.\n'
                '- **중단해도 재개 가능**: 완료된 trial은 `hparam_results.csv` 에 기록되어 다시 학습하지 않습니다.\n'
                '- 예상 시간: trial당 대략 (stage1 장수 × epoch) 학습 + 추론/평가. RTX 5080 기준 '
                '300장×15ep ≈ 수 분/trial → 12 trial이면 한나절~하룻밤 수준.\n'
                '- ⚙️ **각 trial의 학습/추론도 메인 학습 탭과 동일한 행(hang) watchdog으로 보호**됩니다 — '
                '한 trial이 멈춰도 전체 탐색이 멈추지 않고, 그 trial만 실패 처리 후 다음으로 넘어갑니다.')
            with gr.Row():
                hps_out = gr.Textbox('./hparam_search', label='탐색 결과/로그 폴더')
                hps_primary = gr.Dropdown(['fid', 'epi'], value='fid',
                                          label='랭킹 지표 (SAR→EO는 fid 권장, EO 없으면 자동 epi 전환)')
            with gr.Row():
                hps_n = gr.Number(12, label='후보 trial 수', precision=0)
                hps_topk = gr.Number(5, label='stage2 진출 상위 K', precision=0)
                hps_s1ep = gr.Number(15, label='stage1 epoch', precision=0)
                hps_s2ep = gr.Number(45, label='stage2 추가 epoch', precision=0)
            with gr.Row():
                hps_s1img = gr.Number(300, label='stage1 학습 장수', precision=0)
                hps_s2img = gr.Number(0, label='stage2 학습 장수 (0=전체)', precision=0)
                hps_ntest = gr.Number(100, label='평가용 추론 장수 (num_test)', precision=0)
            with gr.Row():
                hps_eo = gr.Textbox('./datasets/Optical/trainB', label='EO(광학) 참조 폴더 (FID 기준)')
                hps_incw = gr.Textbox('', label='InceptionV3 가중치 .pth (오프라인용, 비우면 자동)')
                hps_stall = gr.Number(20, label='trial별 행(hang) 감지 시간 (분)', precision=0)
            hps_btn = gr.Button('🚀 자동 탐색 실행', variant='primary')
            hps_log = gr.Textbox(label='탐색 진행/결과 로그', lines=18, interactive=False, max_lines=18)
            with gr.Row():
                hps_apply_btn = gr.Button('🧬 최적 설정을 탭 4/5에 적용')
                hps_apply_msg = gr.Textbox(label='적용 결과', lines=3, interactive=False)

            hps_apply_btn.click(hps_apply_best, inputs=[hps_out],
                                outputs=[hps_apply_msg] + [comp[k] for k in HPS_APPLY_KEYS])

        # ------------------------------------------------------------------ #
        # Wiring that needs `ordered_inputs` (all CONFIG_KEYS widgets, incl.
        # ones defined in tabs 8/9 above) — must come after every tab is
        # built, since Python needs every comp[...] entry to exist first.
        # Widget LAYOUT stays in each tab's own `with gr.Tab(...)` block above;
        # only the event WIRING lives here (Gradio doesn't require them to be
        # textually co-located with the widgets they connect).
        # ------------------------------------------------------------------ #
        ordered_inputs = [comp[k] for k in CONFIG_KEYS]

        save_basic.click(do_save, inputs=[cfg_path] + ordered_inputs, outputs=save_basic_out)
        save_cut.click(do_save, inputs=[cfg_path] + ordered_inputs, outputs=save_cut_out)
        save_att.click(do_save, inputs=[cfg_path] + ordered_inputs, outputs=save_att_out)

        # Auto-save: any field on any tab writes the FULL current config to
        # disk immediately (not just the fields on that tab), so values never
        # need a manual "저장" click to survive a server restart. Each widget's
        # .change fires the same do_save with the full current value set.
        for _widget in ordered_inputs:
            _widget.change(do_save, inputs=[cfg_path] + ordered_inputs, outputs=autosave_status)

        ckpt_load_btn.click(cfg_apply_checkpoint, inputs=[comp['checkpoints_dir'], ckpt_picker],
                            outputs=[ckpt_load_msg] + ordered_inputs)
        start_btn.click(start_training,
                        inputs=[cfg_path, stall_minutes, max_restarts] + ordered_inputs,
                        outputs=monitor_outputs)
        inf_btn.click(run_inference,
                      inputs=[inf_num, inf_epoch, inf_stall] + ordered_inputs,
                      outputs=[inf_status, inf_gallery])
        eval_btn.click(cut_evaluate,
                      inputs=[eval_epoch, eval_exp, eval_notes, comp['eval_eo_dir'],
                              eval_id_on, comp['eval_real_b_dir'], eval_incw,
                              eval_fid_max, eval_qual_max, eval_struct_max,
                              comp['eval_fake_dir'], comp['eval_real_a_dir']] + ordered_inputs,
                      outputs=[eval_log, eval_table])
        eval_refresh.click(lambda *v: eval_table_rows(_cfg_from_values(v)['results_dir'],
                                                       _cfg_from_values(v)['name']),
                           inputs=ordered_inputs, outputs=eval_table)
        rect_btn.click(cut_rectify,
                      inputs=[rect_epoch, rect_min_area, rect_max_frac, rect_min_rectangularity,
                              comp['rectify_input_dir']] + ordered_inputs,
                      outputs=[rect_status, rect_gallery])
        hps_btn.click(cut_hparam_search,
                      inputs=[hps_out, hps_n, hps_s1ep, hps_s2ep, hps_topk,
                              hps_s1img, hps_s2img, hps_ntest, hps_primary,
                              hps_eo, hps_incw, hps_stall] + ordered_inputs,
                      outputs=hps_log)

    return demo


def main():
    parser = argparse.ArgumentParser(description='CUT + Attention training GUI (PyTorch)')
    parser.add_argument('--share', action='store_true', help='Force a public share link')
    parser.add_argument('--no-share', action='store_true', help='Disable share link')
    parser.add_argument('--port', type=int, default=7860, help='Server port')
    args = parser.parse_args()

    share = (args.share or IN_COLAB) and not args.no_share

    if IN_COLAB:
        print('\n[gui] Colab 감지됨. 출력의 공개 URL (https://XXXX.gradio.live) 을 클릭하세요. '
              '127.0.0.1 / localhost 는 Colab에서 접속되지 않습니다.\n')

    print(f'[gui] build {BUILD}  (gradio {getattr(gr, "__version__", "?")})')
    demo = build_ui()
    # show_error=True surfaces the real exception text in the UI toast instead of
    # a generic "오류", which is essential for diagnosing environment-specific
    # failures. allowed_paths lets Gradio serve result files from the working
    # tree (defence-in-depth alongside the numpy-array gallery output).
    try:
        demo.queue().launch(share=share, server_port=args.port,
                            show_error=True, allowed_paths=[os.getcwd()])
    except TypeError:
        # older Gradio without allowed_paths / show_error
        demo.queue().launch(share=share, server_port=args.port, show_error=True)


if __name__ == '__main__':
    main()
