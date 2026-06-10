""" PyTorch port of the optional structure / colour losses (modules/losses.py).

Add these to the official CUT's models/cut_model.py (see README_pytorch_port.md):
    - gradient_loss(real_A, fake_B): edge/structure preservation (input vs output)
    - color_moment_loss(idt_B, real_B): per-channel mean/std colour consistency

Both expect images in the [-1, 1] range (CUT's default), NCHW.
"""

import torch
import torch.nn.functional as F

EPS = 1e-5


def _luminance(x):
    """[-1,1] NCHW -> [0,1] single-channel luminance (NCHW with C=1)."""
    x = (x + 1.0) * 0.5
    if x.shape[1] == 3:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b
    return x.mean(dim=1, keepdim=True)


def gradient_loss(source, generated, blur=True):
    """L1 between input/output spatial gradients; source blurred to ignore speckle.

    Set ``blur=False`` for sharper edge targets (keeps strong reflector edges
    instead of softening them), at the cost of letting some SAR speckle through.
    """
    gs = _luminance(source)
    gg = _luminance(generated)
    if blur:
        gs = F.avg_pool2d(gs, kernel_size=3, stride=1, padding=1)
    s_dx = gs[:, :, :, 1:] - gs[:, :, :, :-1]
    s_dy = gs[:, :, 1:, :] - gs[:, :, :-1, :]
    g_dx = gg[:, :, :, 1:] - gg[:, :, :, :-1]
    g_dy = gg[:, :, 1:, :] - gg[:, :, :-1, :]
    return (s_dx - g_dx).abs().mean() + (s_dy - g_dy).abs().mean()


def _laplacian(x):
    """4-neighbour Laplacian (second derivative) of a single-channel NCHW map."""
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    xp = F.pad(x, (1, 1, 1, 1), mode='reflect')
    return F.conv2d(xp, k)


def laplacian_loss(source, generated, blur=False):
    """L1 between input/output Laplacians (high-frequency / fine-edge term).

    The Laplacian is large at sharp points/edges and ~0 on smooth regions, so a
    blurred output (low Laplacian) where the input is sharp is penalised. This
    directly discourages the blur that appears around strong SAR reflectors.
    """
    gs = _luminance(source)
    gg = _luminance(generated)
    if blur:
        gs = F.avg_pool2d(gs, kernel_size=3, stride=1, padding=1)
    return (_laplacian(gs) - _laplacian(gg)).abs().mean()


def color_moment_loss(generated, reference):
    """Match per-channel mean/std (identity path: idt_B vs real_B)."""
    g_mean = generated.mean(dim=(2, 3))
    r_mean = reference.mean(dim=(2, 3))
    g_std = generated.var(dim=(2, 3), unbiased=False).add(EPS).sqrt()
    r_std = reference.var(dim=(2, 3), unbiased=False).add(EPS).sqrt()
    return (g_mean - r_mean).abs().mean() + (g_std - r_std).abs().mean()
