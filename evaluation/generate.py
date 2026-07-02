""" Build a generator from a config dict + load a trained checkpoint, then run
it on a folder of images. Used to produce idt_B = G(real_B) for the identity-
path evaluation (test.py does not save idt_B: it is only added to
CUTModel.visual_names when opt.isTrain is True, i.e. during training).

`cfg` is any dict with the same keys used by gui.py's CONFIG_KEYS (netG, normG,
attention_*, hrnet_*, no_antialias, no_antialias_up, checkpoints_dir, name) —
the same config that produced the checkpoint being evaluated, so the rebuilt
generator's architecture matches exactly.
"""

import os
import types

import numpy as np


def build_generator_from_cfg(cfg):
    """Construct (untrained) netG matching the architecture implied by cfg."""
    from models.networks import define_G
    opt = types.SimpleNamespace(
        attention_type=cfg.get('attention_type', 'none'),
        attention_reduction=int(cfg.get('attention_reduction', 16) or 16),
        attention_encoder=bool(cfg.get('attention_encoder', False)),
        attention_resblocks=bool(cfg.get('attention_resblocks', False)),
        attention_decoder=bool(cfg.get('attention_decoder', False)),
        hrnet_branches=int(cfg.get('hrnet_branches', 3) or 3),
        hrnet_modules=int(cfg.get('hrnet_modules', 3) or 3),
        hrnet_blocks=int(cfg.get('hrnet_blocks', 2) or 2),
    )
    net = define_G(
        int(cfg.get('input_nc', 3) or 3), int(cfg.get('output_nc', 3) or 3),
        int(cfg.get('ngf', 64) or 64), cfg.get('netG', 'resnet_9blocks'),
        cfg.get('normG', 'instance'), False, 'xavier', 0.02,
        bool(cfg.get('no_antialias', False)), bool(cfg.get('no_antialias_up', False)),
        [], opt)
    return net


def _strip_module_prefix(state_dict):
    if any(k.startswith('module.') for k in state_dict.keys()):
        return {k[len('module.'):] if k.startswith('module.') else k: v
                for k, v in state_dict.items()}
    return state_dict


def load_generator_checkpoint(net, checkpoints_dir, name, epoch='latest', device='cpu'):
    """Load '<epoch>_net_G.pth' from checkpoints_dir/name into net (in place)."""
    import torch
    path = os.path.join(str(checkpoints_dir), str(name), f'{epoch}_net_G.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f'체크포인트를 찾을 수 없습니다: {path}')
    sd = torch.load(path, map_location=device)
    sd = _strip_module_prefix(sd)
    try:
        net.load_state_dict(sd)
    except Exception as exc:
        raise RuntimeError(
            f'체크포인트 로드 실패 ({path}): 아키텍처(netG/attention/hrnet 옵션)가 '
            f'학습 때와 다를 수 있습니다. ({exc})') from exc
    net.eval().to(device)
    return net, path


def generate_from_folder(net, src_dir, out_dir, crop_size=256, device='cpu',
                         max_items=0, recursive=False, log=None):
    """Run G on every image in src_dir (BICUBIC-resized to crop_size, normalized
    to [-1,1] exactly like CUT's test-time transform) and save to out_dir with
    the SAME basenames, so results can be paired with real_A/real_B/fake_B by
    filename. Returns the list of saved output paths."""
    import torch
    from PIL import Image
    from preprocessing.pipeline import scan_images

    os.makedirs(out_dir, exist_ok=True)
    files = scan_images(src_dir, recursive=recursive, shuffle=False, seed=42,
                        max_items=int(max_items or 0))
    saved = []
    with torch.no_grad():
        for i, p in enumerate(files):
            try:
                im = Image.open(p).convert('RGB').resize(
                    (int(crop_size), int(crop_size)), Image.BICUBIC)
                arr = np.asarray(im).astype(np.float32) / 127.5 - 1.0
                t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
                out = net(t)[0].detach().cpu().numpy()
                out = ((out.transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                # always save as .png (matches test.py's save_images convention),
                # so identity-pair files can be matched to real_B by filename stem
                stem = os.path.splitext(os.path.basename(p))[0]
                outp = os.path.join(out_dir, f'{stem}.png')
                Image.fromarray(out).save(outp)
                saved.append(outp)
            except Exception:
                continue
            if log and ((i + 1) % 50 == 0 or i == len(files) - 1):
                log(f'생성 중 {i+1}/{len(files)}')
    return saved
