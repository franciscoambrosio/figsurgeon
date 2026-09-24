"""Render the two synthetic demo figures the specs, tests and demos run against.

    timeseries_demo.png -- five line series, shaded update bands, a dash-dot split marker
                           (specs/timeseries_demo.py, tests/test_timeseries_demo.py)
    rebrand_source.png  -- viridis field at alpha 0.76 with a colorbar, rotated labels and a
                           two-column marker legend (examples/demo_rebrand.py, tests/test_rebrand.py)

Every value is drawn from a seeded random generator: neither figure shows real data. The
geometry printed at the end is what the spec and the rebrand boxes are measured from; if
you change the layout here, re-measure them.

Run: python tests/data/make_demo_figures.py   (writes both next to this script)
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _px(fig, bbox):
    """A display-space bbox as an (x0, x1, y0, y1) pixel box, y measured from the top."""
    h = fig.bbox.height
    return (round(bbox.x0), round(bbox.x1), round(h - bbox.y1), round(h - bbox.y0))


def timeseries(path=os.path.join(HERE, 'timeseries_demo.png')):
    rng = np.random.default_rng(7)
    n = 60000
    t = np.arange(n)
    levels = np.repeat(rng.uniform(1.0, 3.0, 40), n // 40)
    target = levels + 0.5 * rng.standard_normal(n)
    spikes = rng.choice(n, 150, replace=False)
    target[spikes] += rng.uniform(-5.0, 2.5, 150)
    online = levels + 0.15 * rng.standard_normal(n)
    online[rng.choice(n, 60, replace=False)] -= 1.5
    # Smooth enough that no column holds a 6+ px vertical run of `offline` ink at the split
    # marker, which the spec's ProtectRun would otherwise keep as a stub.
    offline = -0.3 + 0.1 * np.interp(t, np.linspace(0, n, 200), rng.standard_normal(200))
    linear = offline + 0.2 + 0.08 * rng.standard_normal(n)
    mlp = np.full(n, 0.6) + 0.01 * np.sin(t / 5000)

    fig = plt.figure(figsize=(11.66, 4.75), dpi=100)
    ax = fig.add_axes([0.079, 0.067, 0.895, 0.836])
    for x0 in rng.choice(n - 800, 25, replace=False):
        ax.axvspan(x0, x0 + rng.integers(200, 800), color='magenta', alpha=0.4, lw=0)
    ax.plot(t, target, color='black', alpha=0.4, lw=0.6, label='Target')
    ax.plot(t, linear, color='darkorange', lw=1.5, label='Linear model')
    ax.plot(t, online, color='royalblue', lw=1.2, label='Online model')
    ax.plot(t, mlp, color='green', lw=3, label='MLP')
    ax.plot(t, offline, color='black', lw=1.2, label='Offline model')
    ax.axvline(n // 2, color='black', ls='-.', lw=1.2)
    ax.set_xlim(0, n)
    ax.set_ylim(-4.7, 6)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([])
    ax.set_ylabel('Standardized units', fontsize=15)
    ax.tick_params(labelsize=13)
    title = ax.text(0.09, 0.9, 'Sensor signal', transform=ax.transAxes, fontsize=16)
    ticks = [ax.text(x, -4.6, s, ha=ha, fontsize=11) for x, s, ha in
             [(0, '0', 'left'), (n * 0.25, 'Validation', 'center'), (n // 2, '30000', 'center'),
              (n * 0.75, 'Test', 'center'), (n, '60000', 'right')]]
    leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=12,
                    frameon=True, handlelength=3)
    fig.savefig(path, dpi=100)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    print(f'{os.path.basename(path)}: {fig.canvas.get_width_height()}')
    print('  interior      ', _px(fig, ax.get_window_extent(r)))
    print('  legend        ', _px(fig, leg.get_frame().get_window_extent(r)))
    print('  title         ', _px(fig, title.get_window_extent(r)))
    tick_boxes = [_px(fig, tk.get_window_extent(r)) for tk in ticks]
    print('  tick labels   ', (min(b[0] for b in tick_boxes), max(b[1] for b in tick_boxes),
                              min(b[2] for b in tick_boxes), max(b[3] for b in tick_boxes)))
    split = ax.transData.transform((n // 2, 0))[0]
    print('  split marker x', round(split))
    for line, handle in zip(ax.get_lines(), leg.legend_handles):
        print(f'  swatch {line.get_label():14s}', _px(fig, handle.get_window_extent(r)))
    plt.close(fig)


def rebrand(path=os.path.join(HERE, 'rebrand_source.png')):
    rng = np.random.default_rng(11)
    fig = plt.figure(figsize=(8.96, 3.79), dpi=100)
    ax = fig.add_axes([0.115, 0.335, 0.705, 0.655])
    cax = fig.add_axes([0.865, 0.335, 0.015, 0.655])

    xs, ys = np.meshgrid(np.linspace(9.5, 22.5, 260), np.linspace(5.4, 7.1, 120))
    score = 600 - 22 * (xs - 9.5) + 180 * (ys - 5.4)
    lower = 5.5 + 0.03 * (xs - 9.5)
    upper = 7.0 - 0.02 * (xs - 9.5)
    field = np.where((ys > lower) & (ys < upper) & (xs > 10) & (xs < 22), score, np.nan)
    im = ax.pcolormesh(xs, ys, field, cmap='viridis', alpha=0.76, shading='auto')
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Predicted score', fontsize=13)

    fx = np.linspace(10, 21.5, 20)
    ax.plot(fx, 5.5 + 0.03 * (fx - 9.5), color='red', lw=2, zorder=3)
    ax.scatter(fx, 5.5 + 0.03 * (fx - 9.5), color='red', s=40, zorder=4)
    px, py = rng.uniform(12, 21.5, 54), rng.uniform(5.9, 6.9, 54)
    ax.scatter(px, py, c=600 - 22 * (px - 9.5) + 180 * (py - 5.4), cmap='viridis',
               marker='^', s=60, edgecolors='black', linewidths=0.5, vmin=np.nanmin(field),
               vmax=np.nanmax(field), zorder=5)
    ax.scatter(rng.uniform(13, 21, 6), rng.uniform(6.8, 7.0, 6), color='#b0b0e8', marker='v',
               s=60, edgecolors='grey', linewidths=0.5, zorder=5)

    ax.set_xlim(9.5, 22.5)
    ax.set_ylim(5.4, 7.1)
    ax.set_xticks(range(10, 23, 2))
    ax.set_yticks(np.arange(5.5, 7.01, 0.25))
    ax.yaxis.set_major_formatter('{x:.2f}')
    ax.tick_params(labelsize=13)
    ax.set_xlabel('Material use [kg]', fontsize=15)
    ax.set_ylabel('Batch time [s]', fontsize=15)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color='#4a90d0'),
        plt.Line2D([], [], color='red', lw=2),
        plt.Line2D([], [], color='red', marker='o', ls='', markersize=7),
        plt.Line2D([], [], color='#d8c040', marker='^', ls='', markersize=8,
                   markeredgecolor='grey'),
        plt.Line2D([], [], color='#b0b0e8', marker='v', ls='', markersize=8,
                   markeredgecolor='grey'),
    ]
    labels = ['Feasible region', 'Frontier', 'Frontier points (n=20)',
              'Sample (in band, n=54)', 'Sample (outside band, n=6)']
    leg = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.46, 0.0), ncol=2,
                     fontsize=10, frameon=True)
    fig.savefig(path, dpi=100)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    print(f'{os.path.basename(path)}: {fig.canvas.get_width_height()}')
    print('  data area     ', _px(fig, ax.get_window_extent(r)))
    print('  colorbar      ', _px(fig, cax.get_window_extent(r)))
    print('  y label       ', _px(fig, ax.yaxis.label.get_window_extent(r)))
    print('  cbar label    ', _px(fig, cax.yaxis.label.get_window_extent(r)))
    print('  legend        ', _px(fig, leg.get_frame().get_window_extent(r)))
    plt.close(fig)


if __name__ == '__main__':
    timeseries()
    rebrand()
