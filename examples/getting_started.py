"""Getting started with your own figure.

Run this two ways:
    python examples/getting_started.py measure path/to/your_figure.png
        -> prints the numbers you need to write a FigureSpec (colours, candidate geometry).
           Nothing is edited in this step.

    python examples/getting_started.py edit path/to/your_spec.py "highlight the X line"
        -> loads a FigureSpec you've written (see specs/timeseries_demo.py for a fully
           annotated example) and applies the instruction.

This script is intentionally thin -- it is the on-ramp, not the library.  Everything it
calls is public API in figsurgeon.analyze / figsurgeon.describe.
"""
import sys
import importlib.util


def measure(path):
    from figsurgeon import load, census
    from figsurgeon.analyze import probe_interior

    a = load(path)
    print(f'Image size: {a.shape[1]} x {a.shape[0]} (width x height)\n')

    c = census(a)
    print('Dominant CHROMATIC colours (candidate series) -- pick the ones that matter:')
    for rgb, n in c['chromatic'][:12]:
        print(f'  RGB{rgb}  ~{n} px')
    print('\nDominant NEUTRAL colours (grey/black/white -- candidate lines or background):')
    for rgb, n in c['neutral'][:8]:
        print(f'  RGB{rgb}  ~{n} px')

    print('\nGeometry probe (a SUGGESTION -- confirm by eye, do not trust blindly):')
    for k, v in probe_interior(a).items():
        print(f'  {k}: {v}')

    print('\nNext step: copy specs/timeseries_demo.py, replace the path, series colours,')
    print('and interior/legend boxes with the numbers above and what you see in the image.')


def edit(spec_path, instruction):
    from figsurgeon import apply_text, load, report, assert_clean
    from PIL import Image

    modname = spec_path.rsplit('/', 1)[-1].removesuffix('.py')
    spec_mod = importlib.util.spec_from_file_location(modname, spec_path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    fig_spec = mod.SPEC

    a = load(fig_spec.path)
    out, verb, kwargs, info = apply_text(a, fig_spec, instruction)
    print(f'Parsed as: {verb}({kwargs})')

    keep = kwargs.get('keep', [kwargs.get('target')])[0] if kwargs.get('keep') else kwargs.get('target')
    if keep:
        rep = report(a, out, fig_spec, keep)
        try:
            assert_clean(rep)
            print('Verification: passed (nothing changed outside the axes; kept series intact)')
        except AssertionError as e:
            print(f'Verification WARNING: {e}')

    out_path = 'edited.png'
    Image.fromarray(out).save(out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('measure', 'edit'):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == 'measure':
        measure(sys.argv[2])
    else:
        edit(sys.argv[2], sys.argv[3])
