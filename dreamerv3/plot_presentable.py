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
import os
import glob
import matplotlib as mpl
from matplotlib import ticker

# --- Font registration ------------------------------------------------------
# Arial/Helvetica are usually absent on HPC; matplotlib then falls back to
# DejaVu Sans, which (a) looks clunkier than the clean sans in the Cosyne figs
# and (b) ships ONLY Regular(400)+Bold(700) — no true medium weight, so a poster
# "medium" look is impossible. We bundle/register Source Sans 3 (Regular 400,
# Medium 500, Bold 700) from the user font dir if present, giving a clean,
# Arial-like sans with a genuine in-between weight. Self-contained: registered
# per-process via addfont so every repo plotter that imports this module gets it
# without depending on the global matplotlib font cache.
_PREFERRED_SANS = 'Source Sans 3'
_USER_FONT_DIRS = [
    os.path.expanduser('~/.local/share/fonts'),
    os.path.expanduser('~/.fonts'),
]
_FONTS_REGISTERED = False


def _register_fonts():
    """addfont() any Source Sans 3 TTFs found in the user font dirs. Idempotent."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    from matplotlib import font_manager as fm
    have = {f.fname for f in fm.fontManager.ttflist}
    for d in _USER_FONT_DIRS:
        for ttf in glob.glob(os.path.join(d, 'SourceSans3-*.ttf')):
            if ttf not in have:
                try:
                    fm.fontManager.addfont(ttf)
                except Exception:
                    pass
    _FONTS_REGISTERED = True


def _sans_available(name):
    from matplotlib import font_manager as fm
    return any(f.name == name for f in fm.fontManager.ttflist)

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

# --- Tier-2 POSTER type scale (figkit._TYPE * ~1.75) ---------------------------
# Posters are viewed from ~1-1.5 m, so type must read much larger than the
# print-scale Tier-1 above. These are points at final panel size; pair with a
# correspondingly larger per-panel figsize (square spatial panels ~4-5 in).
# DELIBERATELY NOT heavy-bold: the Cosyne figures' bold was a paper-PDF
# convention; a poster wants LARGE but not HEAVY text (plot-style's weight at
# poster size). Weight is set by apply_presentable_style(weight=...), default
# 'medium' for poster; only the panel-letter helper stays bold. Heavier
# lines/spines (not text) help thin strokes survive distance viewing.
_POSTER_TYPE = {
    'font.size': 24, 'axes.labelsize': 26, 'xtick.labelsize': 22,
    'ytick.labelsize': 22, 'legend.fontsize': 22, 'axes.titlesize': 28,
    'axes.linewidth': 1.4, 'lines.linewidth': 2.2,
    'xtick.major.width': 1.4, 'ytick.major.width': 1.4,
    'xtick.major.size': 5, 'ytick.major.size': 5,
}
# Default text weight per mode: poster -> 'medium' (between plot-style's regular
# and Cosyne's heavy bold); Tier-1 keeps matplotlib's normal.
_POSTER_WEIGHT = 'medium'

# Single-column print width is the sensible small default; scripts that pass an
# explicit figsize keep theirs (rcParams only affects figures that DON'T).
_DEFAULT_FIGSIZE = (3.5, 2.6)
_POSTER_FIGSIZE = (5.0, 4.2)

# Module-level flag so helpers (save, etc.) can no-op gracefully if never applied.
_APPLIED = False
_POSTER = False


def apply_presentable_style(small_figsize=True, prop_cycle=True, poster=False,
                            scale=None, weight=None, **extra):
    """Apply the house style globally via rcParams. Call ONCE before any figure.

    Tier 1 (default): print-scale fonts that retroactively restyle existing code
    (fonts, spines, ticks, sizes, color cycle).

    poster=True: Tier-2 POSTER scale — much larger, bold sans type + heavier
        lines/spines to match the Cosyne figure look, intended for large-format
        panels assembled into a poster. Combine with the Tier-2 layout helpers
        below (endpoint_ticks, highlow_colorbar, panel_letter, square_panel).

    small_figsize: set a default figsize for figures created WITHOUT an explicit
        figsize (single-column 3.5x2.6in; poster -> 5.0x4.2in). Scripts that
        hardcode a size keep it.
    prop_cycle: install the Okabe-Ito color cycle as the default.
    scale: optional multiplier applied on top of the chosen type scale (e.g.
        scale=1.2 to nudge all fonts up 20% without picking new absolute sizes).
    weight: title/label font weight ('normal', 'medium', 'semibold', 'bold', or a
        numeric 100-900). Poster default is 'medium' (between plot-style's regular
        and Cosyne's heavy bold); pass 'normal' for the lightest look or 'semibold'
        to push toward Cosyne. Tier-1 leaves matplotlib's default unless set.
    extra: any additional rcParams to override (e.g. figure.figsize=(7.2, 4)).
    """
    global _APPLIED, _POSTER
    _register_fonts()
    type_rc = dict(_POSTER_TYPE if poster else _TYPE)
    # prefer the clean multi-weight sans when it's available (else keep the
    # Arial/Helvetica/DejaVu fallback chain from _BASE_RC untouched)
    if _sans_available(_PREFERRED_SANS):
        type_rc['font.sans-serif'] = (
            [_PREFERRED_SANS] + list(_BASE_RC['font.sans-serif']))
    if scale and scale != 1.0:
        for k in ('font.size', 'axes.labelsize', 'xtick.labelsize',
                  'ytick.labelsize', 'legend.fontsize', 'axes.titlesize'):
            if k in type_rc:
                type_rc[k] = round(type_rc[k] * scale, 1)
    w = weight if weight is not None else (_POSTER_WEIGHT if poster else None)
    if w is not None:
        type_rc['axes.labelweight'] = w
        type_rc['axes.titleweight'] = w
    rc = {**_BASE_RC, **type_rc}
    if small_figsize:
        rc['figure.figsize'] = _POSTER_FIGSIZE if poster else _DEFAULT_FIGSIZE
    if prop_cycle:
        rc['axes.prop_cycle'] = mpl.cycler(color=CYCLE)
    rc.update(extra)
    mpl.rcParams.update(rc)
    _APPLIED = True
    _POSTER = bool(poster)
    return rc


def is_poster():
    return _POSTER


# ============================================================================
# Tier-2 poster layout helpers (the Cosyne "format that works")
# ----------------------------------------------------------------------------
# Six rules distilled from the Cosyne Fig1-3 panels:
#   1. big bold sans type            -> apply_presentable_style(poster=True)
#   2. endpoint-only ticks           -> endpoint_ticks(ax)
#   3. despined data / clean-box img -> rcParams (data) / box_only(ax) (images)
#   4. short labels                  -> caller supplies
#   5. High/Low qualitative colorbar -> highlow_colorbar(...)
#   6. square panels + bold letters  -> square_panel(ax), panel_letter(ax, 'A')
# These touch a single Axes/Figure each; they never set global state, so a script
# can adopt them panel-by-panel without disturbing the rest of the figure.
# ============================================================================

def endpoint_ticks(ax, axis='both', x=None, y=None, xlabels=None, ylabels=None,
                   fmt=None):
    """Reduce an axis to just its two endpoint ticks (min & max), Cosyne-style.

    By default reads each axis's current data limits and places one tick at each
    end. Pass explicit `x`/`y` (2-tuples/lists of tick positions) to override,
    and `xlabels`/`ylabels` to set custom strings (e.g. ['0', '1.0M']).

    fmt: optional callable(value)->str applied to auto endpoint labels when no
        explicit labels are given (e.g. pp.kfmt for compact big numbers).
    """
    def _apply(a, lo_hi, labels, setter, labeler):
        if lo_hi is None:
            lo, hi = a.get_view_interval()
            ticks = [lo, hi]
        else:
            ticks = list(lo_hi)
        setter(ticks)
        if labels is not None:
            labeler([str(t) for t in labels])
        elif fmt is not None:
            labeler([fmt(t) for t in ticks])
    if axis in ('x', 'both'):
        _apply(ax.xaxis, x, xlabels, ax.set_xticks, ax.set_xticklabels)
    if axis in ('y', 'both'):
        _apply(ax.yaxis, y, ylabels, ax.set_yticks, ax.set_yticklabels)
    return ax


def box_only(ax):
    """Restore all 4 spines (a clean frame) with no ticks — for image/heatmap
    panels like the Cosyne Fig2B spatial maps (despined looks wrong on images)."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(mpl.rcParams.get('axes.linewidth', 1.6))
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def square_panel(ax):
    """Force a square data box (equal aspect on the box, not the data) — for
    spatial/activation panels that should read as a square tile."""
    ax.set_box_aspect(1)
    return ax


def panel_letter(ax, letter, x=-0.18, y=1.08, size=None, weight='bold', **kw):
    """Draw a bold panel-letter tag (A, B, ...) in axes-fraction coords, placed
    just outside the top-left of the Axes like the Cosyne panels.

    x/y are in axes fraction (default slightly left of and above the axes).
    size defaults to ~1.3x the title size so letters dominate.
    """
    if size is None:
        size = mpl.rcParams.get('axes.titlesize', 30)
        try:
            size = float(size) * 1.3
        except (TypeError, ValueError):
            size = 34
    return ax.text(x, y, letter, transform=ax.transAxes, fontsize=size,
                   fontweight=weight, va='bottom', ha='right', **kw)


def highlow_colorbar(mappable, ax=None, cax=None, label_high='High',
                     label_low='Low', fraction=0.046, pad=0.04, weight='bold',
                     size=None):
    """A qualitative colorbar labeled only 'High' (top) / 'Low' (bottom) with no
    numeric ticks — the Cosyne Fig2B/Fig3B style for maps where the absolute
    scale is not the point. Returns the Colorbar.
    """
    import matplotlib.pyplot as plt
    fig = (ax.figure if ax is not None else
           (cax.figure if cax is not None else plt.gcf()))
    cb_kw = {} if cax is not None else {'ax': ax, 'fraction': fraction, 'pad': pad}
    cb = fig.colorbar(mappable, cax=cax, **cb_kw)
    cb.set_ticks([])
    cb.outline.set_linewidth(mpl.rcParams.get('axes.linewidth', 1.6))
    if size is None:
        size = mpl.rcParams.get('ytick.labelsize', 24)
    cb.ax.text(0.5, 1.02, label_high, transform=cb.ax.transAxes, ha='center',
               va='bottom', fontsize=size, fontweight=weight)
    cb.ax.text(0.5, -0.02, label_low, transform=cb.ax.transAxes, ha='center',
               va='top', fontsize=size, fontweight=weight)
    return cb


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
