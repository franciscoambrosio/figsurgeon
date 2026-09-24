"""Declarative description of one figure.

Everything figure-specific lives here; the engine in `compose` reads it and holds no
constants of its own.  Geometry is in pixel coordinates, inclusive on both ends.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence

Box = tuple  # (x0, x1, y0, y1) inclusive


@dataclass(frozen=True)
class Series:
    """One plotted artist.

    colour: the artist's own colour, before any alpha is applied.
    alpha:  the alpha it was drawn with.  matplotlib's `alpha=0.4` on a black line renders
            as 0.6 * background, NOT as a grey constant -- getting this wrong is what makes
            such a series indistinguishable from a real grey line.
    """
    name: str
    colour: tuple
    alpha: float = 1.0
    legend_swatch: Optional[Box] = None


@dataclass(frozen=True)
class Bands:
    """Shaded vertical spans (matplotlib axvspan) drawn BEHIND the lines.

    Assumes full-height and column-constant.  Horizontal spans, partial-height spans, or
    spans drawn over the lines are NOT handled -- `analyze.check_band_assumptions` tests
    this against the actual pixels before the engine trusts it.
    """
    darken: float = 0.25   # 1.0 keeps a band's full weight when neutralised; 0.25 renders a
                           # single band near 230 so it stays behind a 204 line


@dataclass(frozen=True)
class ProtectRun:
    """Columns holding a near-vertical annotation (e.g. a dash-dot split marker).

    Pixels of `series` inside these columns are preserved only when they sit in a vertical
    run of at least `min_run` -- so a line crossing the annotation horizontally still gets
    recoloured instead of leaving a stub.
    """
    x0: int
    x1: int
    series: str
    min_run: int = 6


@dataclass
class FigureSpec:
    path: str
    interior: Box                              # inside the axes
    series: Sequence[Series]
    legend: Optional[Box] = None               # legend frame, if it sits inside the axes
    protect: Sequence[Box] = field(default_factory=tuple)   # in-axes text: title, tick labels
    protect_runs: Sequence[ProtectRun] = field(default_factory=tuple)
    bands: Optional[Bands] = None

    grey: tuple = (204, 204, 204)
    tol: float = 42.0        # colour distance that still counts as a match
    seed_margin: float = 0.15  # coverage above a translucent sibling that seeds the opaque one
    halo: int = 2            # px reach of a line's anti-aliasing, for the spatial split
    swatch_ink: float = 0.35  # min coverage to edit inside a legend swatch box

    def by_name(self, name):
        for s in self.series:
            if s.name == name:
                return s
        raise KeyError(f'{name!r} not in spec; have {[s.name for s in self.series]}')

    @property
    def swatches(self):
        return [s.legend_swatch for s in self.series if s.legend_swatch]
