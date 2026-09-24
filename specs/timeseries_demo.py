"""Spec for the synthetic time-series demo figure (1166x475), rendered by make_demo_figures.py.

Geometry is what make_demo_figures.py prints for the rendered figure, confirmed against the
pixels with analyze.probe_interior.  Notable: this figure has NO axes spines, so the interior
bounds come from the axes extent, not from a drawn frame.
"""
import os
from figsurgeon import FigureSpec, Series, Bands, ProtectRun

# Tests point this at the repo copy via conftest.py; the bare default only resolves when run
# from the repo root.
IMAGE_PATH = os.environ.get('FIGSURGEON_DEMO_IMAGE', 'tests/data/timeseries_demo.png')

SPEC = FigureSpec(
    path=IMAGE_PATH,
    interior=(92, 1136, 46, 443),
    legend=(311, 917, 339, 395),
    protect=[
        (180, 338, 64, 96),       # chart title "Sensor signal"
        (92, 1136, 424, 443),     # in-axes tick labels / Validation / Test
    ],
    protect_runs=[ProtectRun(609, 619, 'offline', min_run=6)],   # dash-dot split marker
    bands=Bands(darken=0.25),     # magenta update regions
    series=[
        # alpha 0.40 black -- renders as 0.6*background, NOT as a grey line.  This makes it
        # collinear with `offline`; the engine splits them spatially.
        Series('target',  (0, 0, 0),      alpha=0.40, legend_swatch=(316, 370, 345, 361)),
        Series('linear',  (255, 140, 0),  legend_swatch=(316, 370, 371, 387)),
        Series('online',  (65, 105, 225), legend_swatch=(522, 576, 345, 361)),
        Series('mlp',     (0, 128, 0),    legend_swatch=(522, 576, 371, 387)),
        Series('offline', (0, 0, 0),      legend_swatch=(732, 786, 345, 361)),
    ],
)
