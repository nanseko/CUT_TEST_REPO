""" Per-epoch training-loss curves for the CUT pipeline.

train.py (via util/visualizer.py) already appends every logged iteration to
``<checkpoints_dir>/<name>/loss_log.txt`` in the fixed format:

    (epoch: 5, iters: 400, time: 0.124, data: 0.002) G_GAN: 0.259 D_real: 0.198 D_fake: 0.304 G: 3.257 NCE: 2.945 NCE_Y: 2.975

This module treats that file as the single source of truth: it parses it,
averages every loss per epoch, writes ``loss_history.csv`` and renders
``loss_curve.png`` next to it. Because it reads the log rather than hooking
into the training step, it works for GUI and CLI runs alike, for runs already
in progress, and needs no state kept in the trainer.

The three curves the user cares about ("D Loss, G Loss, NCE Loss") are drawn
prominently:
  - D  = (D_real + D_fake) / 2   (the actual discriminator objective; note the
         raw log only prints D_real and D_fake, so D is reconstructed here)
  - G  = total generator loss (the printed ``G``)
  - NCE = PatchNCE loss (``NCE``; NCE_Y for the identity path if present)

Optional component curves (G_GAN, D_real, D_fake, NCE_Y, and any structure/
colour losses like G_grad/G_lap/G_coherence/G_color) are drawn faintly in a
second panel so a stalled/diverging run is obvious at a glance.

matplotlib is optional: the CSV is always written; the PNG is skipped with a
message if matplotlib is unavailable.
"""

import os
import re
import csv
from collections import OrderedDict, defaultdict

# "(epoch: 5, iters: 400, time: ...) name: val name: val ..."
_RE_HEADER = re.compile(r'\(epoch:\s*(\d+),\s*iters:\s*(\d+)')
_RE_PAIR = re.compile(r'(\w+):\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')

# curves highlighted in the main panel
PRIMARY_ORDER = ['D', 'G', 'NCE']
# everything else shown faintly in the components panel
COMPONENT_ORDER = ['G_GAN', 'D_real', 'D_fake', 'NCE_Y',
                   'G_grad', 'G_lap', 'G_coherence', 'G_color']


def parse_loss_log(log_path):
    """Parse loss_log.txt -> (epochs, series).

    epochs: sorted list of epoch numbers seen.
    series: OrderedDict{loss_name: [per-epoch mean aligned to `epochs`]}, with
            a synthesized 'D' = mean(D_real, D_fake) when both are present.
    Missing values for an epoch are None (so a loss enabled mid-run still lines
    up on the epoch axis).
    """
    # sums[epoch][name] = (sum, count)
    sums = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    if not os.path.exists(log_path):
        return [], OrderedDict()
    with open(log_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            mh = _RE_HEADER.search(line)
            if not mh:
                continue
            epoch = int(mh.group(1))
            # only parse the "name: value" pairs AFTER the header parenthesis,
            # so time/data/iters inside the header are never treated as losses
            tail = line.split(')', 1)[-1]
            for name, val in _RE_PAIR.findall(tail):
                try:
                    v = float(val)
                except ValueError:
                    continue
                slot = sums[epoch][name]
                slot[0] += v
                slot[1] += 1

    epochs = sorted(sums.keys())
    # collect the set of all loss names, preserving a sensible order
    all_names = []
    for ep in epochs:
        for name in sums[ep]:
            if name not in all_names:
                all_names.append(name)

    series = OrderedDict()
    for name in all_names:
        series[name] = [
            (sums[ep][name][0] / sums[ep][name][1]) if sums[ep][name][1] else None
            for ep in epochs
        ]
    # synthesized combined discriminator loss
    if 'D_real' in series and 'D_fake' in series:
        dr, df = series['D_real'], series['D_fake']
        series['D'] = [
            ((a + b) / 2.0) if (a is not None and b is not None) else None
            for a, b in zip(dr, df)
        ]
    return epochs, series


def write_history_csv(epochs, series, csv_path):
    """Write one row per epoch: epoch + every loss series column."""
    names = list(series.keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['epoch'] + names)
        for i, ep in enumerate(epochs):
            w.writerow([ep] + [('' if series[n][i] is None else round(series[n][i], 6))
                               for n in names])


def render_plot(epochs, series, png_path, title=None):
    """Render the loss curves to png_path. Returns True on success, False if
    matplotlib is unavailable (CSV is still written by the caller)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if not epochs:
        return False

    def present(name):
        return name in series and any(v is not None for v in series[name])

    primaries = [n for n in PRIMARY_ORDER if present(n)]
    components = [n for n in COMPONENT_ORDER if present(n)]

    nrows = 2 if components else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), squeeze=False)
    ax0 = axes[0][0]

    colors = {'D': '#d62728', 'G': '#1f77b4', 'NCE': '#2ca02c'}
    for name in primaries:
        xs = [e for e, v in zip(epochs, series[name]) if v is not None]
        ys = [v for v in series[name] if v is not None]
        ax0.plot(xs, ys, label=name, color=colors.get(name), linewidth=2)
    ax0.set_xlabel('epoch')
    ax0.set_ylabel('loss (epoch mean)')
    ax0.set_title(title or 'Training losses (D / G / NCE)')
    ax0.legend(loc='upper right')
    ax0.grid(True, alpha=0.3)

    if components:
        ax1 = axes[1][0]
        for name in components:
            xs = [e for e, v in zip(epochs, series[name]) if v is not None]
            ys = [v for v in series[name] if v is not None]
            ax1.plot(xs, ys, label=name, linewidth=1, alpha=0.8)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss (epoch mean)')
        ax1.set_title('Components (G_GAN / D_real / D_fake / NCE_Y / structure losses)')
        ax1.legend(loc='upper right', ncol=2, fontsize=8)
        ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    try:
        fig.savefig(png_path, dpi=110)
    finally:
        plt.close(fig)
    return True


def update_loss_plot(checkpoints_dir, name, title=None):
    """Parse loss_log.txt for experiment <name> and (re)write loss_history.csv
    + loss_curve.png in the same folder. Returns a dict with the output paths
    and status; never raises (best-effort, safe to call every epoch)."""
    exp_dir = os.path.join(str(checkpoints_dir), str(name))
    log_path = os.path.join(exp_dir, 'loss_log.txt')
    csv_path = os.path.join(exp_dir, 'loss_history.csv')
    png_path = os.path.join(exp_dir, 'loss_curve.png')
    out = {'csv': None, 'png': None, 'epochs': 0, 'ok': False, 'message': ''}
    try:
        epochs, series = parse_loss_log(log_path)
        if not epochs:
            out['message'] = f'loss_log.txt에 파싱할 epoch 데이터가 없습니다: {log_path}'
            return out
        os.makedirs(exp_dir, exist_ok=True)
        write_history_csv(epochs, series, csv_path)
        out['csv'] = csv_path
        out['epochs'] = len(epochs)
        # keep plot text ASCII/English: matplotlib's default font has no Korean
        # glyphs, so a Korean title/label renders as tofu boxes on most PCs.
        if render_plot(epochs, series, png_path, title=title or 'Training losses: %s' % name):
            out['png'] = png_path
            out['message'] = f'{len(epochs)} epoch까지 손실 그래프 갱신: {png_path}'
        else:
            out['message'] = (f'{len(epochs)} epoch까지 CSV 저장(그래프는 matplotlib 미설치로 건너뜀): '
                              f'{csv_path}')
        out['ok'] = True
    except Exception as exc:
        out['message'] = f'손실 그래프 갱신 실패: {exc}'
    return out


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3:
        print(update_loss_plot(sys.argv[1], sys.argv[2])['message'])
    else:
        print('usage: python -m util.loss_plot <checkpoints_dir> <name>')
