""" PyTorch port of the optional structure / colour losses (modules/losses.py).

Add these to the official CUT's models/cut_model.py (see README_pytorch_port.md):
    - gradient_loss(real_A, fake_B): edge/structure preservation (input vs output)
    - color_moment_loss(idt_B, real_B): per-channel mean/std colour consistency

Both expect images in the [-1, 1] range (CUT's default), NCHW.
"""

import numpy as np
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


def reflector_saliency_map(source, window=5, boost=3.0):
    """Weight map highlighting small, LOCALLY bright compact regions — a proxy
    for strong SAR reflectors (ships/vehicles/building corners: metal ->
    corner-reflector response). These are exactly the small rigid objects that
    unpaired GAN translation tends to hallucinate away, because a uniformly
    averaged loss barely notices a handful of pixels among a whole image.

    Returns a (B,1,H,W) map in [1, 1+boost]: ~1 on flat/background regions, up
    to 1+boost at local brightness peaks. boost=0 recovers a uniform map (=1),
    i.e. no behaviour change.
    """
    lum = _luminance(source)                                          # (B,1,H,W) in [0,1]
    local_mean = F.avg_pool2d(lum, kernel_size=window, stride=1, padding=window // 2)
    contrast = (lum - local_mean).clamp(min=0)                        # local brightness peaks
    b = contrast.shape[0]
    ref = torch.quantile(contrast.reshape(b, -1), 0.995, dim=1).clamp(min=EPS).view(b, 1, 1, 1)
    norm = (contrast / ref).clamp(0, 1)
    return (1.0 + boost * norm).detach()


def reflector_saliency_weights_for_shapes(source, shapes, boost=3.0, window=5):
    """reflector_saliency_map resized to each (H, W) in `shapes` (one per
    PatchNCE tap layer), batch-averaged, flattened row-major (H then W) to
    match PatchSampleF's ``feat.permute(0,2,3,1).flatten(1,2)`` patch order.
    Used to bias PatchNCE patch sampling toward small bright objects instead
    of uniform-random (which barely samples objects covering few pixels).
    Returns a list of 1-D float64 numpy arrays, one per shape.
    """
    base = reflector_saliency_map(source, window=window, boost=boost)  # (B,1,Hs,Ws)
    out = []
    for (h, w) in shapes:
        wm = F.interpolate(base, size=(int(h), int(w)), mode='bilinear', align_corners=False)
        wm = wm.mean(dim=0).squeeze(0)                # batch-average -> (H, W)
        wm = wm.detach().cpu().numpy().reshape(-1).astype(np.float64)
        out.append(np.clip(wm, 1e-6, None))
    return out


def gradient_loss(source, generated, blur=True, weighted=False, boost=3.0):
    """L1 between input/output spatial gradients; source blurred to ignore speckle.

    Set ``blur=False`` for sharper edge targets (keeps strong reflector edges
    instead of softening them), at the cost of letting some SAR speckle through.
    Set ``weighted=True`` to weight the loss by ``reflector_saliency_map`` so
    small bright objects (ships/vehicles/buildings) get proportionally more
    supervision than their pixel count would otherwise give them.
    """
    gs = _luminance(source)
    gg = _luminance(generated)
    if blur:
        gs = F.avg_pool2d(gs, kernel_size=3, stride=1, padding=1)
    s_dx = gs[:, :, :, 1:] - gs[:, :, :, :-1]
    s_dy = gs[:, :, 1:, :] - gs[:, :, :-1, :]
    g_dx = gg[:, :, :, 1:] - gg[:, :, :, :-1]
    g_dy = gg[:, :, 1:, :] - gg[:, :, :-1, :]
    if weighted:
        weight = reflector_saliency_map(source, boost=boost)
        w_dx, w_dy = weight[:, :, :, 1:], weight[:, :, 1:, :]
        return ((s_dx - g_dx).abs() * w_dx).mean() + ((s_dy - g_dy).abs() * w_dy).mean()
    return (s_dx - g_dx).abs().mean() + (s_dy - g_dy).abs().mean()


def _laplacian(x):
    """4-neighbour Laplacian (second derivative) of a single-channel NCHW map."""
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    xp = F.pad(x, (1, 1, 1, 1), mode='reflect')
    return F.conv2d(xp, k)


def laplacian_loss(source, generated, blur=False, weighted=False, boost=3.0):
    """L1 between input/output Laplacians (high-frequency / fine-edge term).

    The Laplacian is large at sharp points/edges and ~0 on smooth regions, so a
    blurred output (low Laplacian) where the input is sharp is penalised. This
    directly discourages the blur that appears around strong SAR reflectors.
    ``weighted``/``boost``: see gradient_loss — emphasise small bright objects.
    """
    gs = _luminance(source)
    gg = _luminance(generated)
    if blur:
        gs = F.avg_pool2d(gs, kernel_size=3, stride=1, padding=1)
    diff = (_laplacian(gs) - _laplacian(gg)).abs()
    if weighted:
        diff = diff * reflector_saliency_map(source, boost=boost)
    return diff.mean()


def color_moment_loss(generated, reference):
    """Match per-channel mean/std (identity path: idt_B vs real_B)."""
    g_mean = generated.mean(dim=(2, 3))
    r_mean = reference.mean(dim=(2, 3))
    g_std = generated.var(dim=(2, 3), unbiased=False).add(EPS).sqrt()
    r_std = reference.var(dim=(2, 3), unbiased=False).add(EPS).sqrt()
    return (g_mean - r_mean).abs().mean() + (g_std - r_std).abs().mean()
