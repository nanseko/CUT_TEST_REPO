""" Generic (framework-agnostic) per-image comparison metrics, numpy-only.

Used both by the SAR preprocessing order-search (preprocessing/optimize.py,
comparing raw SAR vs preprocessed output) and by the CUT model-output
evaluation (evaluation/, comparing real_A vs fake_B and real_B vs idt_B).
Inputs are single-channel float arrays in [0, 1] unless noted otherwise.
"""

import numpy as np

EPS = 1e-8


def resize_to(g, shape):
    """Resize a [0,1] grayscale array to `shape` (H, W) via PIL bilinear."""
    from PIL import Image
    if g.shape == shape:
        return g
    im = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).resize(
        (shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(im).astype(np.float64) / 255.0


def to_luminance(arr):
    """RGB (H,W,3) uint8/float or grayscale (H,W) -> luminance in [0,1]."""
    a = np.asarray(arr).astype(np.float64)
    if a.max() > 1.0:
        a = a / 255.0
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return np.clip(a, 0, 1)


def psnr(ref, img):
    """Peak signal-to-noise ratio (dB), higher = more similar. Inputs in [0,1]."""
    mse = float(np.mean((ref - img) ** 2))
    if mse <= EPS:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def cc(ref, img):
    """Pearson correlation coefficient of pixel intensities. Higher = more similar."""
    a = ref.ravel() - ref.mean()
    b = img.ravel() - img.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / (d + EPS))


def _laplacian(x):
    xp = np.pad(x, 1, mode='reflect')
    return (xp[:-2, 1:-1] + xp[2:, 1:-1] + xp[1:-1, :-2] + xp[1:-1, 2:] - 4 * x)


def epi(ref, img):
    """Edge Preservation Index: Pearson correlation of the Laplacians (high-pass)
    of reference and processed image. ~1 = edges well preserved."""
    lr = _laplacian(ref).ravel()
    li = _laplacian(img).ravel()
    lr = lr - lr.mean()
    li = li - li.mean()
    d = float(np.sqrt((lr * lr).sum() * (li * li).sum()))
    return float((lr * li).sum() / (d + EPS))


def _box_filter(x, w):
    """Mean over a wxw window (reflect pad), pure NumPy via integral image."""
    w = int(w)
    if w <= 1:
        return x.copy()
    if w % 2 == 0:
        w += 1
    r = w // 2
    xpad = np.pad(x, r, mode='reflect')
    cs = np.cumsum(np.cumsum(xpad, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode='constant')
    H, W = x.shape
    S = (cs[w:w + H, w:w + W] - cs[0:H, w:w + W]
         - cs[w:w + H, 0:W] + cs[0:H, 0:W])
    return S / float(w * w)


def ssim(ref, img, window=7):
    """Structural Similarity Index (single-scale, windowed), inputs in [0,1].
    ~1 = identical structure/luminance/contrast."""
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    mu_x = _box_filter(ref, window)
    mu_y = _box_filter(img, window)
    sig_x2 = _box_filter(ref * ref, window) - mu_x * mu_x
    sig_y2 = _box_filter(img * img, window) - mu_y * mu_y
    sig_xy = _box_filter(ref * img, window) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x2 + sig_y2 + c2)
    return float(np.mean(num / (den + EPS)))


def composite_score(m):
    """Higher = better. Combines edge preservation, correlation and PSNR."""
    return (m.get('epi', 0.0) + m.get('cc', 0.0) + m.get('psnr', 0.0) / 40.0) / 3.0
