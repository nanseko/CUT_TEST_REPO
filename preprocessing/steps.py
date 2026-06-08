""" Modular SAR preprocessing steps for the CUT pipeline.

Each step implements the PreprocessStep interface and is registered in
STEP_REGISTRY. Steps operate on a single-channel float image in [0, 1]
(2-D ndarray) until ``channel_adapter`` expands it to 3 channels and
``normalize_for_cut`` produces the final uint8 image.

Implementations are pure NumPy/Pillow (no scipy/cv2 hard dependency) so they
run anywhere, including Colab. Optional extras (bm3d, cv2-CLAHE) degrade
gracefully when the package is missing.

Design follows docs/README_pipeline.md.
"""

import numpy as np

EPS = 1e-6


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_gray01(img):
    """Ensure a 2-D float32 image in [0, 1]."""
    a = np.asarray(img).astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    mx = float(a.max()) if a.size else 1.0
    if mx > 1.0:               # 8/16-bit -> normalise
        a = a / mx
    return a


def _box_filter(x, w):
    """Mean over a wxw window (reflect pad) via integral image. Pure NumPy."""
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


def _local_stats(x, w):
    mean = _box_filter(x, w)
    sq = _box_filter(x * x, w)
    var = np.maximum(sq - mean * mean, 0.0)
    return mean, var


def _estimate_enl(x):
    m = float(x.mean())
    v = float(x.var())
    enl = (m * m) / (v + EPS)
    return float(np.clip(enl, 1.0, 64.0))


def optical_like_v1_cdf(num_bins=1024):
    """Target tone curve: lift shadows, boost mid contrast, soft highlight."""
    x = np.linspace(0, 1, num_bins)
    y = np.power(x, 0.75)                       # SAR dark-heavy -> mid-tone
    y = 1.0 - np.power(1.0 - y, 1.15)           # highlight roll-off
    y = np.maximum.accumulate(y)                # monotonic
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    return x, y


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #

class PreprocessStep:
    name = 'base'

    def __init__(self, enabled=True, **params):
        self.enabled = enabled
        self.params = params

    def apply(self, image, context):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #

class ValidateImageStep(PreprocessStep):
    name = 'validate_image'

    def apply(self, image, context):
        img = _to_gray01(image)
        handle_nan = self.params.get('handle_nan', 'zero')
        if np.isnan(img).any():
            fill = 0.0 if handle_nan == 'zero' else float(np.nanmedian(img))
            img = np.nan_to_num(img, nan=fill)
        if np.isinf(img).any():
            img = np.nan_to_num(img, posinf=1.0, neginf=0.0)
        if self.params.get('drop_empty', True):
            if img.size == 0 or float(img.std()) < 1e-6:
                context['skip'] = True
                context['skip_reason'] = 'empty/constant image'
        return img, context


class SARIntensityTransformStep(PreprocessStep):
    name = 'sar_intensity_transform'

    def apply(self, image, context):
        img = _to_gray01(image)
        mode = self.params.get('mode', 'log1p')
        if mode == 'log1p':
            img = np.log1p(img / max(self.params.get('eps', 1e-6), 1e-9))
        elif mode == 'db':
            img = 10.0 * np.log10(img + self.params.get('eps', 1e-6))
        # renormalise to [0,1] for stable downstream
        lo, hi = float(img.min()), float(img.max())
        if hi - lo > EPS:
            img = (img - lo) / (hi - lo)
        return img.astype(np.float32), context


class SpeckleFilterStep(PreprocessStep):
    name = 'speckle_filter'

    def apply(self, image, context):
        x = _to_gray01(image)
        method = self.params.get('method', 'refined_lee')
        w = int(self.params.get('window_size', 7))
        enl_p = self.params.get('enl', 'auto')
        enl = _estimate_enl(x) if enl_p in ('auto', None) else float(enl_p)
        context.setdefault('stats', {})['enl'] = round(enl, 3)

        if method == 'lee':
            out = self._lee(x, w, enl)
        elif method == 'refined_lee':
            out = self._refined_lee(x, w, enl)
        elif method == 'gamma_map':
            out = self._gamma_map(x, w, enl)
        elif method == 'frost':
            out = self._frost(x, w, float(self.params.get('damping_factor', 2.0)))
        elif method == 'bm3d':
            out = self._bm3d(x, context)
        else:
            out = x
        return np.clip(out, 0.0, 1.0).astype(np.float32), context

    def _lee(self, x, w, enl):
        mean, var = _local_stats(x, w)
        ci2 = var / (mean * mean + EPS)
        cu2 = 1.0 / enl
        W = (ci2 - cu2) / (ci2 + EPS)
        W = np.clip(W, 0.0, 1.0)
        return mean + W * (x - mean)

    def _refined_lee(self, x, w, enl):
        # practical approximation: Lee + edge-preserving blend
        lee = self._lee(x, w, enl)
        gx = np.abs(np.gradient(x, axis=1))
        gy = np.abs(np.gradient(x, axis=0))
        g = gx + gy
        if g.max() > EPS:
            e = np.clip(g / (np.percentile(g, 95) + EPS), 0.0, 1.0)
        else:
            e = np.zeros_like(x)
        return e * x + (1.0 - e) * lee

    def _gamma_map(self, x, w, enl):
        mean, var = _local_stats(x, w)
        ci2 = var / (mean * mean + EPS)
        cu2 = 1.0 / enl
        out = mean.copy()
        mask = ci2 > cu2
        denom = (ci2 - cu2)
        alpha = (1.0 + cu2) / np.where(denom > EPS, denom, EPS)
        b = alpha - enl - 1.0
        disc = b * b * mean * mean + 4.0 * alpha * enl * mean * x
        disc = np.maximum(disc, 0.0)
        rhat = (b * mean + np.sqrt(disc)) / (2.0 * alpha + EPS)
        out = np.where(mask, rhat, mean)
        # very heterogeneous -> keep original
        out = np.where(ci2 > (cu2 * 6.0), x, out)
        return out

    def _frost(self, x, w, damping):
        if w % 2 == 0:
            w += 1
        r = w // 2
        mean, var = _local_stats(x, w)
        sigma2 = var / (mean * mean + EPS)        # per-pixel CoV^2
        xpad = np.pad(x, r, mode='reflect')
        H, W = x.shape
        acc = np.zeros_like(x)
        wsum = np.zeros_like(x)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dist = (dy * dy + dx * dx) ** 0.5
                shifted = xpad[r + dy:r + dy + H, r + dx:r + dx + W]
                wt = np.exp(-damping * sigma2 * dist)
                acc += wt * shifted
                wsum += wt
        return acc / (wsum + EPS)

    def _bm3d(self, x, context):
        try:
            import bm3d
            sigma = self.params.get('bm3d_sigma', 'auto')
            if sigma in ('auto', None):
                sigma = float(np.std(x - _box_filter(x, 3)))
            return bm3d.bm3d(x, sigma_psd=max(sigma, 1e-3))
        except Exception:
            context.setdefault('warnings', []).append(
                'bm3d 미설치 -> refined_lee로 대체 (pip install bm3d)')
            return self._refined_lee(x, 7, _estimate_enl(x))


class OutlierClippingStep(PreprocessStep):
    name = 'outlier_clipping'

    def apply(self, image, context):
        x = _to_gray01(image)
        minp = float(self.params.get('min_percentile', 0.2))
        maxp = float(self.params.get('max_percentile', 99.8))
        ignore_zero = self.params.get('ignore_zero', True)
        vals = x[x > 0] if ignore_zero and (x > 0).any() else x
        lo = np.percentile(vals, minp)
        hi = np.percentile(vals, maxp)
        if hi - lo < EPS:
            return x, context
        out = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        return out.astype(np.float32), context


class HistogramMappingStep(PreprocessStep):
    name = 'histogram_mapping'

    def apply(self, image, context):
        x = _to_gray01(image)
        bins = int(self.params.get('bins', 1024))
        mode = self.params.get('mode', 'sar_only')

        if mode == 'unpaired_optical_reference' and context.get('optical_target') is not None:
            tx, ty = context['optical_target']
        else:
            tx, ty = optical_like_v1_cdf(bins)

        # source CDF
        hist, edges = np.histogram(x.ravel(), bins=bins, range=(0.0, 1.0))
        cdf = np.cumsum(hist).astype(np.float64)
        cdf = cdf / (cdf[-1] + EPS)
        centers = (edges[:-1] + edges[1:]) * 0.5
        p = np.interp(x.ravel(), centers, cdf)        # rank in [0,1]
        mapped = np.interp(p, ty, tx)                 # invert target CDF
        out = mapped.reshape(x.shape).astype(np.float32)

        clahe = self.params.get('clahe', {}) or {}
        if clahe.get('enabled', False):
            out = self._clahe(out, clahe, context)
        return np.clip(out, 0.0, 1.0), context

    def _clahe(self, x, cfg, context):
        try:
            import cv2
            u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
            cl = cv2.createCLAHE(clipLimit=float(cfg.get('clip_limit', 2.0)),
                                 tileGridSize=tuple(cfg.get('tile_grid_size', [8, 8])))
            return cl.apply(u8).astype(np.float32) / 255.0
        except Exception:
            context.setdefault('warnings', []).append('CLAHE 건너뜀 (cv2 미설치)')
            return x


class ResizeOrTileStep(PreprocessStep):
    name = 'resize_or_tile'

    def apply(self, image, context):
        from PIL import Image
        x = _to_gray01(image)
        size = int(self.params.get('image_size', 256))
        mode = self.params.get('mode', 'resize')
        u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(u8)
        if mode == 'center_crop':
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((size, size), Image.BILINEAR)
        return (np.asarray(im).astype(np.float32) / 255.0), context


class ChannelAdapterStep(PreprocessStep):
    name = 'channel_adapter'

    def apply(self, image, context):
        x = np.asarray(image, dtype=np.float32)
        if x.ndim == 2:
            ch = int(self.params.get('output_channels', 3))
            x = np.stack([x] * ch, axis=-1)
        return x, context


class NormalizeForCUTStep(PreprocessStep):
    name = 'normalize_for_cut'

    def apply(self, image, context):
        x = np.asarray(image, dtype=np.float32)
        if x.ndim == 2:
            x = np.stack([x] * 3, axis=-1)
        out = np.clip(x, 0.0, 1.0)
        if self.params.get('output_range', 'uint8') == 'uint8':
            out = (out * 255.0).astype(np.uint8)
        return out, context


STEP_REGISTRY = {
    'validate_image': ValidateImageStep,
    'sar_intensity_transform': SARIntensityTransformStep,
    'speckle_filter': SpeckleFilterStep,
    'outlier_clipping': OutlierClippingStep,
    'histogram_mapping': HistogramMappingStep,
    'resize_or_tile': ResizeOrTileStep,
    'channel_adapter': ChannelAdapterStep,
    'normalize_for_cut': NormalizeForCUTStep,
}

# Recommended order (Phase 1 default)
DEFAULT_STEP_ORDER = [
    'validate_image', 'sar_intensity_transform', 'speckle_filter',
    'outlier_clipping', 'histogram_mapping', 'resize_or_tile',
    'channel_adapter', 'normalize_for_cut',
]

SPECKLE_METHODS = ['lee', 'frost', 'bm3d', 'refined_lee', 'gamma_map']
HISTOGRAM_MODES = ['sar_only', 'unpaired_optical_reference', 'preset']
INTENSITY_MODES = ['none', 'log1p', 'db']
