""" FID / KID (Fréchet / Kernel Inception Distance) utilities, numpy + torch.

Used by the SAR preprocessing order-search (preprocessing/optimize.py) and by
the CUT model-output evaluation (evaluation/) to compare a generated image set
against a real reference set (e.g. preprocessed SAR vs Optical, or fake_B vs
real EO). Degrades gracefully (returns None / raises informative errors) when
torch/torchvision or the InceptionV3 weights are unavailable — callers should
catch and fall back to non-FID metrics.

Offline / air-gapped support: InceptionV3 ImageNet weights
('inception_v3_google-0cc3c7bd.pth') are searched locally before falling back
to torchvision's cache/download. See docs/OFFLINE_FID.md.
"""

import os

import numpy as np

INCEPTION_FILENAME = 'inception_v3_google-0cc3c7bd.pth'


def fid_available():
    try:
        import torch  # noqa
        import torchvision  # noqa
        return True
    except Exception:
        return False


def _default_weights_paths():
    """Local locations searched for the InceptionV3 weights (offline support)."""
    paths = []
    env = os.environ.get('INCEPTION_WEIGHTS')
    if env:
        paths.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    for base in ('.', './weights', './models', here, repo,
                 os.path.join(repo, 'weights'), os.path.join(repo, 'models')):
        paths.append(os.path.join(base, INCEPTION_FILENAME))
    return paths


def resolve_inception_weights(explicit=None):
    """Return a local weights path if found (explicit -> env -> common dirs), else None."""
    for p in ([explicit] if explicit else []) + _default_weights_paths():
        if p and os.path.exists(p):
            return p
    return None


def load_inception(device='cpu', weights_path=None):
    """Return (net, transform). net outputs 2048-d pool features (fc replaced
    with Identity). Loads local weights if resolvable, else uses torchvision's
    cache / downloads once."""
    import torch
    import torchvision.transforms as T
    from torchvision.models import inception_v3, Inception_V3_Weights
    wp = resolve_inception_weights(weights_path)
    if wp:
        # offline: build without downloading, then load local weights
        try:
            net = inception_v3(weights=None, init_weights=False)
        except TypeError:
            net = inception_v3(weights=None)
        sd = torch.load(wp, map_location=device)
        try:
            net.load_state_dict(sd)
        except Exception:
            net.load_state_dict(sd, strict=False)
    else:
        # online: torchvision uses its cache or downloads once
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()      # 2048-d pool features
    net.eval().to(device)
    tf = T.Compose([T.ToPILImage(), T.Resize((299, 299)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return net, tf


def features_from_arrays(array_iter, net, tf, device='cpu', batch=16):
    """RGB uint8 array iterator -> (N, 2048) InceptionV3 pool features."""
    import torch
    feats, buf = [], []
    with torch.no_grad():
        for im in array_iter:
            if im is None:
                continue
            if im.ndim == 2:
                im = np.stack([im] * 3, -1)
            buf.append(tf(im.astype(np.uint8)))
            if len(buf) == batch:
                feats.append(net(torch.stack(buf).to(device)).cpu().numpy())
                buf = []
        if buf:
            feats.append(net(torch.stack(buf).to(device)).cpu().numpy())
    return np.concatenate(feats, 0) if feats else np.zeros((0, 2048))


def read_rgb_arrays(files):
    """Yield each image path as an RGB uint8 numpy array (skips unreadable files)."""
    from PIL import Image
    for p in files:
        try:
            yield np.asarray(Image.open(p).convert('RGB'))
        except Exception:
            continue


def _sqrtm_trace(s1, s2):
    """trace(sqrtm(s1 @ s2)) for symmetric PSD s1, s2 using only numpy (eigh)."""
    w, v = np.linalg.eigh(s1)
    w = np.clip(w, 0, None)
    s1_half = (v * np.sqrt(w)) @ v.T
    m = s1_half @ s2 @ s1_half
    ev = np.clip(np.linalg.eigvalsh(m), 0, None)
    return float(np.sqrt(ev).sum())


def fid_from_feats(fa, fb):
    """Fréchet Inception Distance between two feature sets (numpy-only).
    Lower = more similar (identical distributions -> 0)."""
    if fa.shape[0] < 2 or fb.shape[0] < 2:
        return float('nan')
    mu1, mu2 = fa.mean(0), fb.mean(0)
    sa = np.cov(fa, rowvar=False)
    sb = np.cov(fb, rowvar=False)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(sa) + np.trace(sb)
                 - 2.0 * _sqrtm_trace(sa, sb))


def kid_from_feats(fa, fb, subset_size=100, num_subsets=10, seed=42):
    """Kernel Inception Distance (unbiased polynomial-kernel MMD^2 estimator),
    averaged over random subsets. More stable than FID for small sample sizes.
    Lower = more similar (identical distributions -> ~0, can be slightly negative
    due to the unbiased estimator)."""
    m, n = fa.shape[0], fb.shape[0]
    s = min(int(subset_size), m, n)
    if s < 2:
        return float('nan')
    rng = np.random.default_rng(seed)
    d = fa.shape[1]

    def poly_kernel(x, y):
        return (x @ y.T / d + 1.0) ** 3

    vals = []
    for _ in range(int(num_subsets)):
        ia = rng.choice(m, s, replace=False)
        ib = rng.choice(n, s, replace=False)
        x, y = fa[ia], fb[ib]
        kxx, kyy, kxy = poly_kernel(x, x), poly_kernel(y, y), poly_kernel(x, y)
        term1 = (kxx.sum() - np.trace(kxx)) / (s * (s - 1))
        term2 = (kyy.sum() - np.trace(kyy)) / (s * (s - 1))
        term3 = kxy.sum() / (s * s)
        vals.append(term1 + term2 - 2 * term3)
    return float(np.mean(vals))


def compute_fid_kid(set_a_files, set_b_files, device=None, weights_path=None,
                    max_items=500, batch=16, want_kid=True, log=None):
    """High-level helper: read two image-path lists, extract InceptionV3
    features once each, return {'fid':..., 'kid':..., 'n_a':, 'n_b':}.
    Raises if torch/torchvision unavailable; callers should catch and degrade.
    """
    import torch
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net, tf = load_inception(dev, weights_path)
    fa_files = list(set_a_files)[:int(max_items or 0) or None]
    fb_files = list(set_b_files)[:int(max_items or 0) or None]
    if log:
        log(f'InceptionV3 특징 추출: A={len(fa_files)}장, B={len(fb_files)}장 (device={dev})')
    fa = features_from_arrays(read_rgb_arrays(fa_files), net, tf, dev, batch)
    fb = features_from_arrays(read_rgb_arrays(fb_files), net, tf, dev, batch)
    out = {'fid': fid_from_feats(fa, fb), 'n_a': int(fa.shape[0]), 'n_b': int(fb.shape[0])}
    if want_kid:
        out['kid'] = kid_from_feats(fa, fb)
    return out
