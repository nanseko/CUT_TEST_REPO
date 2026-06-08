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

from models.attention import CBAM, CoordinateAttention, make_attention
from models.networks import ResnetGenerator
from models.losses_extra import gradient_loss, color_moment_loss


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


def test_losses():
    a = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    b = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    gl = gradient_loss(a, b)
    cl = color_moment_loss(a, b)
    assert gl.item() >= 0 and cl.item() >= 0
    # gradients flow
    a = a.requires_grad_(True)
    (gradient_loss(a, b) + color_moment_loss(a, b)).backward()
    assert a.grad is not None
    print(f'losses: grad={gl.item():.4f} color={cl.item():.4f} | backward OK')


def main():
    test_attention_shapes()
    test_generator('none')
    test_generator('coord', attention_encoder=True, attention_resblocks=True)
    test_generator('cbam', attention_encoder=True, attention_resblocks=True, attention_decoder=True)
    test_generator('coord', attention_encoder=True, attention_resblocks=True, no_antialias=True)
    test_losses()
    print('\nAll attention-integration smoke tests passed.')


if __name__ == '__main__':
    main()
