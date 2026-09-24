"""Run rebrand.remap_colormap + verify_photo.check_colormap_remapped on the audit corpus
and dump one JSON record per case. Whole-figure remap (box=None) is the primary case; a
few cases carry an explicit box to stress-test the outside-box note."""
import json
import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from figsurgeon import rebrand, verify_photo  # noqa: E402

CACHE = os.path.join(ROOT, 'evals', '_corpus', 'audit_remap')
OUT = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'audit_remap')
os.makedirs(os.path.join(OUT, 'renders'), exist_ok=True)

BRAND = ('#00407A', '#52BDEC')
BRAND_DIV = ('#7A0000', '#FFFFFF', '#00407A')

# case: name, file, source_cmap, box (or None), extra kwargs, note about what the case tests
CASES = [
    # -- primary claim test: correctly-identified matplotlib colormap, whole figure --
    ('gcd_heatmap', 'gcd_heatmap.png', 'viridis', None, {}, 'known-good viridis heatmap'),
    ('ackley_contour', 'ackley_contour.png', 'viridis', None, {}, 'known-good viridis contour'),
    ('photon_jet', 'photon_jet.png', 'jet', None, {}, "known band defect (non-mpl jet)"),
    ('photon_mfp_jet', 'photon_mfp_jet.png', 'jet', None, {}, 'same series as photon_jet'),
    ('photon_mac_jet', 'photon_mac_jet.png', 'jet', None, {}, 'same series as photon_jet'),
    ('collatz_fractal', 'collatz_fractal.png', 'inferno', None, {}, 'measured best-fit inferno, dist 4.5'),
    ('tidal_desert_turbo', 'tidal_desert_turbo.png', 'turbo', None, {}, 'measured best-fit turbo, dist 5.5'),
    ('aquaplanet_turbo', 'aquaplanet_turbo.png', 'turbo', None, {}, 'measured best-fit turbo, dist 4.8'),
    ('cloudless_desert_turbo', 'cloudless_desert_turbo.png', 'turbo', None, {}, 'measured best-fit turbo, dist 7.7; only 1 of 4 panels carries it'),
    ('gliese_turbo', 'gliese_turbo.png', 'turbo', None, {}, 'measured best-fit turbo, dist 11.8'),
    ('gaussian_surface', 'gaussian_surface.png', 'cividis', None, {}, 'measured best-fit cividis, dist 1.05'),
    ('gw170608_spectro', 'gw170608_spectro.png', 'viridis', None, {}, 'measured best-fit viridis, dist 1.3; 2 panels + colorbar, whole figure'),
    ('gw170817_spectro', 'gw170817_spectro.png', 'viridis', None, {}, 'measured best-fit viridis dist 7.5; NO colorbar in the image at all'),
    ('gw_transient_catalog', 'gw_transient_catalog.png', 'afmhot', None, {}, 'measured best-fit afmhot, dist 9.2; 11 panels, no colorbar'),
    ('wiki_depth_scatter', 'wiki_depth_scatter.png', 'viridis', None, {}, 'scatter+colorbar, visual viridis guess'),
    ('rho_oph_scatter', 'rho_oph_scatter.png', 'viridis', None, {}, 'scatter+colorbar, visual viridis guess'),
    ('hunter_gatherer_pop', 'hunter_gatherer_pop.png', 'viridis', None, {}, 'map+colorbar, visual viridis guess'),
    ('fertile_crescent_viridis', 'fertile_crescent_viridis.png', 'viridis', None, {}, 'map, no colorbar, visual viridis guess'),
    ('jan_rain_viridis', 'jan_rain_viridis.png', 'viridis', None, {}, 'map, visual viridis guess'),
    ('cvd_strips_box', 'cvd_strips.png', 'viridis', (0, 217, 861, 380), {}, 'viridis strip ONLY boxed; cividis+parula strips deliberately outside'),

    # -- caller names the WRONG colormap (real-world "not matplotlib" / diverging-mistaken cases) --
    ('cretaceous_jet', 'cretaceous_jet.png', 'jet', None, {}, 'poor global fit (dist 15.6); visually a diverging blue-yellow-red scale, caller guesses jet'),
    ('kangerlussuaq_jet', 'kangerlussuaq_jet.png', 'jet', None, {}, 'poor global fit (dist 40.2); visually diverging blue-white-orange-red, caller guesses jet'),
    ('gw_transient_wrong', 'gw_transient_catalog.png', 'inferno', None, {}, 'real cmap is afmhot (dist 9.2); caller guesses inferno (dist 58.2)'),
    ('rossmo_custom', 'rossmo_custom.png', 'viridis', None, {}, "custom Rossmo geoprofiling scale, NOT matplotlib; caller guesses viridis"),
    ('radar_reflectivity_custom', 'radar_reflectivity_custom.png', 'jet', None, {}, 'NWS radar categorical scale, NOT matplotlib; caller guesses jet'),
    ('gliese_profile_custom', 'gliese_profile_viridis.png', 'viridis', None, {}, 'custom magenta/cyan scale, NOT matplotlib; caller guesses viridis'),
    ('harappan_no_cmap', 'harappan_minerals.png', 'viridis', None, {}, 'terrain basemap with categorical markers, no real colormap present; caller guesses viridis'),
    ('hr_diagram_no_cmap', 'hr_diagram.png', 'viridis', None, {}, 'no colormap/colorbar present, just a few coloured line segments; caller guesses viridis'),

    # -- outside-box note stress tests --
    ('gw170608_boxed_panel1', 'gw170608_spectro.png', 'viridis', (0, 150, 1280, 660), {},
     'box the TOP panel+colorbar only; bottom panel legitimately carries the SAME viridis, deliberately left out'),
]


def run(case):
    name, fname, cmap, box, kwargs, note = case
    path = os.path.join(CACHE, fname)
    before = Image.open(path).convert('RGB')
    try:
        after, match_frac, alpha = rebrand.remap_colormap(before, box=box, source_cmap=cmap,
                                                            new_colours=BRAND, **kwargs)
    except Exception as e:
        return {'name': name, 'file': fname, 'cmap': cmap, 'box': box, 'note': note,
                'error': f'{type(e).__name__}: {e}'}
    ok, detail = verify_photo.check_colormap_remapped(before, after, box=box, source_cmap=cmap,
                                                        new_colours=BRAND)
    outside_fired = 'NOTE:' in detail
    render_path = os.path.join(OUT, 'renders', name + '.png')
    after.save(render_path)
    return {
        'name': name, 'file': fname, 'cmap': cmap, 'box': box, 'note': note,
        'match_fraction': match_frac, 'alpha': alpha,
        'verdict': ok, 'detail': detail, 'outside_note_fired': outside_fired,
        'render': render_path,
    }


if __name__ == '__main__':
    results = [run(c) for c in CASES]
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    for r in results:
        if 'error' in r:
            print(f"{r['name']:28s} ERROR: {r['error']}")
        else:
            print(f"{r['name']:28s} verdict={str(r['verdict']):6s} note={r['outside_note_fired']!s:5s}  {r['detail'][:100]}")
