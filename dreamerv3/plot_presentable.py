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
# Scaled ~1.6x from the original single-column (3.5x2.6in) calibration because
# the plotters pass small_figsize=False and keep their larger hardcoded figsizes
# (e.g. 6x5in scatters, multi-panel grids) — at the old 10pt the type read small
# on those bigger canvases. Titles stay a touch above labels.
_TYPE = {
    'font.size': 16, 'axes.labelsize': 16, 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'legend.fontsize': 14, 'axes.titlesize': 17,
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


# When set (via --out_fmt), every figure save is rewritten to this extension so
# plotters that hardcode .svg/.pdf transparently emit e.g. PNG instead (handy for
# PowerPoint, which scales imported SVGs awkwardly). None = keep native formats.
_FMT_OVERRIDE = None
_FMT_DPI = None
_VECTOR_EXTS = {'svg', 'pdf', 'png', 'eps', 'ps', 'jpg', 'jpeg', 'tif', 'tiff'}
_RASTER_EXTS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}


def _install_format_override(fmt, dpi=None):
    """Monkeypatch Figure.savefig so any `*.svg/.pdf/...` path is rewritten to
    `*.{fmt}`, optionally at a fixed `dpi` for raster targets. Applied
    process-wide, once. File-object targets are left alone.

    dpi matters for PowerPoint: PP re-renders every element, so a flat raster
    (png) at a modest dpi (~150-200) is far snappier than a dense scatter SVG
    (thousands of vector points) OR a 600-dpi png (tens of megapixels)."""
    global _FMT_OVERRIDE, _FMT_DPI
    _FMT_OVERRIDE = fmt.lower().lstrip('.') if fmt else None
    _FMT_DPI = dpi
    if not _FMT_OVERRIDE:
        return
    import os
    import matplotlib.figure as mfig
    if getattr(mfig.Figure, '_pp_savefig_patched', False):
        return
    _orig = mfig.Figure.savefig

    def _savefig(self, fname, *a, **kw):
        if _FMT_OVERRIDE is not None:
            try:
                s = os.fspath(fname)
            except TypeError:
                s = None  # file object / buffer — don't touch
            if s is not None:
                stem, ext = os.path.splitext(s)
                if ext.lower().lstrip('.') in _VECTOR_EXTS:
                    fname = f'{stem}.{_FMT_OVERRIDE}'
                    kw.pop('format', None)
                    if _FMT_DPI and _FMT_OVERRIDE in _RASTER_EXTS:
                        kw['dpi'] = _FMT_DPI
        return _orig(self, fname, *a, **kw)

    mfig.Figure.savefig = _savefig
    mfig.Figure._pp_savefig_patched = True


def save(fig, path, exts=('svg', 'pdf'), dpi=600, tight=True):
    """Save a figure as vector formats (svg + pdf by default) at the house dpi.

    path may include or omit an extension; the stem is reused for each ext.
    Returns the list of written paths. Honors the --out_fmt override (collapses
    to the single chosen format).
    """
    import os
    if _FMT_OVERRIDE:
        exts = (_FMT_OVERRIDE,)
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
    # Registered defensively: some plotters (e.g. manifold_analysis) already
    # define their own --fmt, so this uses a distinct name and skips silently
    # if a conflicting option is somehow already present.
    try:
        parser.add_argument(
            '--out_fmt', default=None, choices=['png', 'svg', 'pdf'],
            help='Override EVERY figure output to this format regardless of what '
                 'each plotter hardcodes (e.g. --out_fmt png for PowerPoint, which '
                 'lags on dense scatter SVGs). Default: keep each native format.')
        parser.add_argument(
            '--out_dpi', type=int, default=None,
            help='DPI for --out_fmt raster output. Default 200 for png/jpg (good '
                 'for slides; avoids the 600-dpi bitmap bloat that lags PowerPoint). '
                 'Raise for print, lower (~120) for the lightest decks.')
    except Exception:
        pass
    return parser


def maybe_apply(args, **kwargs):
    """Apply the presentable style iff args.presentable, and install the --out_fmt
    output-format override iff set. Call once before creating any figure."""
    applied = False
    if getattr(args, 'presentable', False):
        apply_presentable_style(**kwargs)
        applied = True
    fmt = getattr(args, 'out_fmt', None)
    if fmt:
        dpi = getattr(args, 'out_dpi', None)
        if dpi is None and fmt in _RASTER_EXTS:
            dpi = 200  # slide-friendly default; 600 vector-dpi bloats bitmaps
        _install_format_override(fmt, dpi)
    return applied
