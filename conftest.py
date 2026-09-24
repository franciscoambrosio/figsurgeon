"""Make `pytest tests/` work from a clean checkout, on any machine.

`specs/` is not an installed package, so `from timeseries_demo import SPEC` fails at
collection unless it's on `sys.path`. The chart demo spec also resolves its image path
relative to the working directory, so it needs an absolute default. Both are fixed here
rather than by editing every test.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

for path in (ROOT, os.path.join(ROOT, 'specs')):
    if path not in sys.path:
        sys.path.insert(0, path)

# The demo spec reads this env var, defaulting to a bare relative filename that only
# resolves when pytest happens to be run from the repo root.
os.environ.setdefault('FIGSURGEON_DEMO_IMAGE',
                      os.path.join(ROOT, 'tests', 'data', 'timeseries_demo.png'))
