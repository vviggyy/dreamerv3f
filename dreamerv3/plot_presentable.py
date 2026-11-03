"""plot_presentable.py — lean "Tier-1" presentable figure styling.

Ports the house-style rcParams + palette from the `plot-style` skill
(`figkit.style`) into a SELF-CONTAINED, dependency-free module so any plotting
script in this repo can opt into a consistent publication/presentation look via
a `--presentable` flag, WITHOUT importing the skill or rewriting its figures.

Tier 1 (this module) = global rcParams that retroactively restyle existing
plotting code: Arial/Helvetica typography at a single small print-scale, no
top/right spines, no grid, outward low-count ticks, the colorblind-safe
Okabe-Ito color cycle, and vector-friendly output (pdf.fonttype 42). It does NOT
touch layout/gridspec/panel-geometry (Tier 2) — those stay per-figure.

Usage in a script:
    from dreamerv3 import plot_presentable as pp   # or: import plot_presentable as pp
    pp.add_presentable_arg(parser)
    args = parser.parse_args()
    pp.maybe_apply(args)        # call BEFORE creating any figure
    ...
    pp.save(fig, out_path)      # svg + pdf; or keep the script's own savefig

Palette / rc values are lifted from the plot-style skill's figkit.py so figures
built here match the skill's examples.
"""
import matplotlib as mpl
from matplotlib import ticker

# --- Okabe-Ito colorblind-safe palette (from figkit.OKABE/CYCLE) ---
OKABE = {
    'black': '#000000', 'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73',
    'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00',
    'purple': '#CC79A7', 'gray': '#999999',
}
CYCLE = [OKABE[k] for k in ('blue', 'vermillion', 'green', 'orange', 'purple', 'sky')]

# --- Base house-style rcParams (figkit.BASE_RC) ---
_BASE_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Nimbus Sans', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'legend.frameon': False,
    'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'lines.linewidth': 1.2,
    'figure.dpi': 120,
    'savefig.dpi': 600,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
}

# --- The ONE type scale, in points at final print size (figkit.TYPE) ---
_TYPE = {
    'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'legend.fontsize': 9,
}

# Single-column print width is the sensible small default; scripts that pass an
# explicit figsize keep theirs (rcParams only affects figures that DON'T).
_DEFAULT_FIGSIZE = (3.5, 2.6)

# Module-level flag so helpers (save, etc.) can no-op gracefully if never applied.
_APPLIED = False


def apply_presentable_style(small_figsize=True, prop_cycle=True, **extra):
    """Apply the Tier-1 house style globally via rcParams.

    Call ONCE before creating any figure. Affects every subsequent plot drawn
    into the active rcParams (fonts, spines, ticks, sizes, color cycle).

    small_figsize: set a single-column (3.5x2.6in) default figsize. Only affects
        figures created WITHOUT an explicit figsize — scripts that hardcode a
        size keep it (shrink those at the call site if desired).
    prop_cycle: install the Okabe-Ito color cycle as the default. Figures with
        hardcoded colors keep them; only default-cycled series change.
    extra: any additional rcParams to override (e.g. figure.figsize=(7.2, 4)).
    """
    global _APPLIED
    rc = {**_BASE_RC, **_TYPE}
    if small_figsize:
        rc['figure.figsize'] = _DEFAULT_FIGSIZE
    if prop_cycle:
        rc['axes.prop_cycle'] = mpl.cycler(color=CYCLE)
    rc.update(extra)
    mpl.rcParams.update(rc)
    _APPLIED = True
    return rc


def is_applied():
    return _APPLIED


def kfmt(x, pos=None):
    """Compact big-number tick label: 2500 -> '2.5k', 5_000_000 -> '5M', 0 -> '0'."""
    ax = abs(x)
    if ax >= 1e6:
        s = f'{x / 1e6:.1f}M'
    elif ax >= 1e3:
        s = f'{x / 1e3:.1f}k'
    else:
        return f'{x:.0f}' if float(x).is_integer() else f'{x:g}'
    # strip trailing '.0' before the suffix: '2.0k' -> '2k'
    return s.replace('.0M', 'M').replace('.0k', 'k')


def nice_ticks(ax, n=3, axis='both', kfmt_axis=None):
    """2-3 round outward ticks per axis (MaxNLocator), optionally kfmt-formatted.

    ax: a matplotlib Axes.
    n: target max number of ticks per axis.
    axis: 'x', 'y', or 'both'.
    kfmt_axis: 'x', 'y', or 'both' to apply the compact kfmt big-number formatter.
    """
    do_x = axis in ('x', 'both')
    do_y = axis in ('y', 'both')
    if do_x:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(n, prune=None))
    if do_y:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(n, prune=None))
    if kfmt_axis in ('x', 'both'):
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(kfmt))
    if kfmt_axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(kfmt))
    return ax


def save(fig, path, exts=('svg', 'pdf'), dpi=600, tight=True):
    """Save a figure as vector formats (svg + pdf by default) at the house dpi.

    path may include or omit an extension; the stem is reused for each ext.
    Returns the list of written paths.
    """
    import os
    stem = os.path.splitext(str(path))[0]
    d = os.path.dirname(os.path.abspath(stem))
    if d:
        os.makedirs(d, exist_ok=True)
    kw = {'dpi': dpi}
    if tight:
        kw['bbox_inches'] = 'tight'
    written = []
    for ext in exts:
        out = f'{stem}.{ext}'
        fig.savefig(out, **kw)
        written.append(out)
    return written


def add_presentable_arg(parser, default=False):
    """Add a standard --presentable flag to an argparse parser."""
    parser.add_argument(
        '--presentable', action='store_true', default=default,
        help='Apply the Tier-1 presentable house style (Arial, Okabe-Ito, no '
             'top/right spines, small print-scale fonts, vector-friendly). '
             'Global rcParams only; layout is unchanged.')
    return parser


def maybe_apply(args, **kwargs):
    """Apply the style iff args.presentable is set. Call before any figure."""
    if getattr(args, 'presentable', False):
        apply_presentable_style(**kwargs)
        return True
    return False
