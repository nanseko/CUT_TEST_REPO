""" Smoke test for the CUT + attention integration (PyTorch). Requires torch.

Verifies that attention (CBAM / Coordinate) inserted into the official CUT
ResnetGenerator keeps the generator output shape, that PatchNCE-style feature
tapping still works with the attention-aware `nce_default` indices, and that the
optional structure / colour losses run and backprop.

Run from the repo root:  python tests/test_attention_port.py
"""

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from models.attention import CBAM, CoordinateAttention, make_attention
from models.networks import ResnetGenerator, HRNetGenerator, PatchSampleF
from models.losses_extra import (
    gradient_loss, laplacian_loss, color_moment_loss, coherence_loss,
    reflector_saliency_map, reflector_saliency_weights_for_shapes,
)


def test_attention_shapes():
    x = torch.randn(2, 64, 32, 32)
    assert CBAM(64)(x).shape == x.shape
    assert CoordinateAttention(64)(x).shape == x.shape
    assert make_attention('none', 64)(x).shape == x.shape
    print('attention shape: OK')


def test_generator(attention_type, **flags):
    g = ResnetGenerator(3, 3, norm_layer=nn.InstanceNorm2d, n_blocks=9,
                        attention_type=attention_type, **flags)
    x = torch.randn(1, 3, 256, 256)
    # plain forward
    y = g(x)
    assert y.shape == (1, 3, 256, 256), y.shape
    # PatchNCE-style encode_only with the model's attention-aware default taps
    nce = g.nce_default
    feats = g(x, layers=nce, encode_only=True)
    assert isinstance(feats, list) and len(feats) == len(nce)
    assert all(f.dim() == 4 for f in feats)
    print(f"generator[{attention_type}, {flags}] OK | nce_layers={nce} | "
          f"feat shapes={[tuple(f.shape[1:]) for f in feats]}")


def test_hrnet(attention_type, **flags):
    g = HRNetGenerator(3, 3, ngf=32, norm_layer=nn.InstanceNorm2d,
                       attention_type=attention_type, **flags)
    x = torch.randn(1, 3, 256, 256)
    y = g(x)
    assert y.shape == (1, 3, 256, 256), y.shape
    nce = g.nce_default
    feats = g(x, layers=nce, encode_only=True)
    assert isinstance(feats, list) and len(feats) == len(nce)
    assert all(f.dim() == 4 for f in feats)
    # HRNet keeps a full-resolution stream: an early tap must still be 256x256
    assert feats[2].shape[-1] == 256, feats[2].shape
    print(f"hrnet[{attention_type}, {flags}] OK | nce_layers={nce} | "
          f"feat shapes={[tuple(f.shape[1:]) for f in feats]}")


def test_losses():
    a = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    b = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    gl = gradient_loss(a, b)
    ll = laplacian_loss(a, b, blur=False)
    cl = color_moment_loss(a, b)
    assert gl.item() >= 0 and ll.item() >= 0 and cl.item() >= 0
    # gradients flow through all three
    a = a.requires_grad_(True)
    (gradient_loss(a, b, blur=False) + laplacian_loss(a, b) + color_moment_loss(a, b)).backward()
    assert a.grad is not None
    print(f'losses: grad={gl.item():.4f} lap={ll.item():.4f} color={cl.item():.4f} | backward OK')


def test_reflector_saliency():
    x = torch.zeros(1, 3, 64, 64) - 1.0                 # flat dark background
    x[:, :, 30:34, 30:34] = 1.0                          # small bright "reflector" blob

    w = reflector_saliency_map(x, boost=3.0)
    assert w.shape == (1, 1, 64, 64)
    assert w[0, 0, 32, 32].item() > w[0, 0, 5, 5].item() + 0.5, 'blob must get higher weight than background'
    w0 = reflector_saliency_map(x, boost=0.0)
    assert torch.allclose(w0, torch.ones_like(w0)), 'boost=0 must be a no-op (uniform weight=1)'

    gen = x.clone()
    gen[:, :, 30:34, 30:34] = -1.0                       # generated output lost the blob
    gl_u = gradient_loss(x, gen, blur=False, weighted=False)
    gl_w = gradient_loss(x, gen, blur=False, weighted=True, boost=3.0)
    assert gl_w.item() > gl_u.item(), 'weighted loss must penalise the lost blob more than unweighted'
    ll_u = laplacian_loss(x, gen, weighted=False)
    ll_w = laplacian_loss(x, gen, weighted=True, boost=3.0)
    assert ll_w.item() > ll_u.item()

    shapes = [(64, 64), (32, 32), (16, 16)]
    ws = reflector_saliency_weights_for_shapes(x, shapes, boost=3.0)
    for (h, w_), arr in zip(shapes, ws):
        assert arr.shape == (h * w_,) and (arr > 0).all()
    print(f'reflector_saliency: OK | grad_loss unweighted={gl_u.item():.4f} weighted={gl_w.item():.4f} '
          f'| lap_loss unweighted={ll_u.item():.4f} weighted={ll_w.item():.4f}')


def test_coherence_loss():
    """Verify coherence_loss physically ranks crisp edges as better than
    (progressively) blurred/soft blobs, noisy texture, and erasure -- the
    "small object turned into a cloud" failure mode."""
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]
    square = ((np.abs(yy - H // 2) < 12) & (np.abs(xx - W // 2) < 12)).astype(np.float32)

    def gauss_blur(a, sigma):
        # tiny separable-box approximation of a Gaussian blur (avoid a scipy dep in tests)
        k = max(1, int(sigma * 3))
        t = torch.from_numpy(a)[None, None]
        for _ in range(3):
            t = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(t, (k, k, k, k), mode='reflect'),
                kernel_size=2 * k + 1, stride=1)
        return t[0, 0].numpy()

    def mk(img01):
        t = torch.from_numpy(img01.astype(np.float32))[None, None].repeat(1, 3, 1, 1) * 2 - 1
        return t

    src = mk(square)
    rng = np.random.RandomState(0)
    noisy = np.clip(square * (0.5 + 0.5 * rng.randn(H, W)), 0, 1).astype(np.float32)
    flat = np.zeros((H, W), np.float32)
    soft_mild = gauss_blur(square, 2)
    soft_heavy = gauss_blur(square, 4)

    l_crisp = coherence_loss(src, mk(square), boost=3.0).item()
    l_soft_mild = coherence_loss(src, mk(soft_mild), boost=3.0).item()
    l_soft_heavy = coherence_loss(src, mk(soft_heavy), boost=3.0).item()
    l_noisy = coherence_loss(src, mk(noisy), boost=3.0).item()
    l_flat = coherence_loss(src, mk(flat), boost=3.0).item()

    assert l_crisp < l_soft_mild < l_soft_heavy, (l_crisp, l_soft_mild, l_soft_heavy)
    assert l_crisp < l_noisy
    assert l_crisp < l_flat
    print(f'coherence_loss: OK | crisp={l_crisp:.4f} < soft_mild={l_soft_mild:.4f} '
          f'< soft_heavy={l_soft_heavy:.4f}; crisp < noisy={l_noisy:.4f}; crisp < flat={l_flat:.4f}')

    # boost=0 must be a no-op-strength call (still runs, weight uniform)
    a = torch.randn(1, 3, 32, 32).clamp(-1, 1)
    b = torch.randn(1, 3, 32, 32).clamp(-1, 1).requires_grad_(True)
    l0 = coherence_loss(a, b, boost=0.0)
    l0.backward()
    assert b.grad is not None and torch.isfinite(b.grad).all()
    print('coherence_loss boost=0 + backward: OK')


def test_weighted_patch_sampling():
    torch.manual_seed(0)
    np.random.seed(0)
    H, W = 64, 64
    n_total = H * W
    weight = np.ones(n_total, dtype=np.float64)
    hot_idx = np.arange(50, 60)
    weight[hot_idx] = 50.0

    sampler = PatchSampleF(use_mlp=False)
    feat = torch.randn(1, 8, H, W)
    counts = np.zeros(n_total)
    trials = 100
    for _ in range(trials):
        _, ids = sampler([feat], num_patches=64, patch_ids=None, weights=[weight])
        counts[ids[0].numpy()] += 1
    hot_rate = counts[hot_idx].mean() / trials
    cold_rate = np.delete(counts, hot_idx).mean() / trials
    assert hot_rate > cold_rate * 10, 'weighted sampling must strongly favour high-weight positions'

    # backward compatibility: no weights arg -> unchanged uniform behaviour
    _, ids0 = sampler([feat], num_patches=64, patch_ids=None)
    assert ids0[0].shape == (64,)
    print(f'weighted_patch_sampling: OK | hot_rate={hot_rate:.3f} cold_rate={cold_rate:.4f} '
          f'ratio={hot_rate / cold_rate:.1f}x')


def main():
    test_attention_shapes()
    test_generator('none')
    test_generator('coord', attention_encoder=True, attention_resblocks=True)
    test_generator('cbam', attention_encoder=True, attention_resblocks=True, attention_decoder=True)
    test_generator('coord', attention_encoder=True, attention_resblocks=True, no_antialias=True)
    test_hrnet('none')
    test_hrnet('coord', attention_encoder=True, attention_resblocks=True)
    test_losses()
    test_reflector_saliency()
    test_coherence_loss()
    test_weighted_patch_sampling()
    print('\nAll attention-integration smoke tests passed.')


if __name__ == '__main__':
    main()
