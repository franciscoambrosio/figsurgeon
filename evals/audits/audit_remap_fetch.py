"""Fetch the audit corpus from Wikimedia Commons, same pattern as evals/chart_remap.py /
evals/real_corpus.py (User-Agent required, cached, credited)."""
import os
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from evals import real_corpus  # noqa: E402

CACHE = os.path.join(ROOT, 'evals', '_corpus', 'audit_remap')
os.makedirs(CACHE, exist_ok=True)

# name -> (Commons title, fetch width, believed colormap, category, note)
CORPUS = {
    'gcd_heatmap': ('Heatmap of GCD Matrix.png', 1200, 'viridis', 'heatmap',
                     'known-good from evals/chart_remap.py'),
    'ackley_contour': ('Ackley 2d.png', 1200, 'viridis', 'contour',
                        'known-good from evals/chart_remap.py'),
    'photon_jet': ('Photon Cross Sections.png', 1200, 'jet', 'heatmap+lines',
                    "known non-mpl jet from evals/chart_remap.py"),
    'photon_mfp_jet': ('Photon Mean Free Path.png', 1200, 'jet', 'heatmap+lines',
                        'same series as photon_jet, presumed same non-mpl jet'),
    'photon_mac_jet': ('Photon Mass Attenuation Coefficients.png', 1200, 'jet',
                        'heatmap+lines', 'same series, presumed same non-mpl jet'),
    'cvd_strips': ('CVD-friendly sequential colormaps.png', 1400, 'viridis', 'colorbar-strips',
                    'a labelled demo of several sequential colormaps side by side'),
    'collatz_fractal': ('Collatz Fractal.png', 1200, 'inferno', 'heatmap',
                         'fractal render, title suggests inferno-like palette'),
    'wiki_depth_scatter': ("Wikipedias' article depth vs number of articles.png", 1200,
                            'viridis', 'scatter+colorbar', 'guess from typical mpl default'),
    'rho_oph_scatter': ('Rho ophiuchi region L1688 star mass vs dust disk mass.png', 1200,
                         'viridis', 'scatter+colorbar', 'guess from typical mpl default'),
    'hr_diagram': ('HR diagram 2msun feh0 1.png', 1200, 'viridis', 'scatter+colorbar',
                    'guess'),
    'gliese_turbo': ('Temperature of gliese 12b as locked desert planet 1.png', 1200,
                      'turbo', 'map', 'exoplanet climate-model map, guess turbo/jet-like'),
    'aquaplanet_turbo': ('Aquaplanet temperature distribution 1 1 1 1.png', 1200, 'turbo',
                          'map', 'same series, guess turbo/jet-like'),
    'tidal_desert_turbo': ('Tidally locked desert planet surface temperature 1 1 1 1.png',
                            1200, 'turbo', 'map', 'same series'),
    'cloudless_desert_turbo': ('Cloudless desert planet surface temperature 1 1 1 1.png',
                                1200, 'turbo', 'map', 'same series'),
    'kangerlussuaq_jet': ('T surface july kangerlussuaq area 2000-2021 2.png', 1200, 'jet',
                           'map', 'climate reanalysis map, guess jet'),
    'cretaceous_jet': ('Cretaceous 90ma co2 900 annual temperature 2.png', 1200, 'jet', 'map',
                        'paleoclimate map, guess jet, possibly non-mpl'),
    'fertile_crescent_viridis': ('Fertile crescent precipitation 1.png', 1200, 'viridis',
                                  'map', 'guess viridis'),
    'gliese_profile_viridis': ('Gliese 12 b temperature profile if rotating ocean planet 1.png',
                                1200, 'viridis', 'line+colorbar', 'guess'),
    'jan_rain_viridis': ('January Rain Middle East 1.png', 1200, 'viridis', 'map', 'guess'),
    'rossmo_custom': ('Kyllikki saari murder site rossmo 2 1 1 1.png', 1200, 'NOT-MPL',
                       'heatmap', 'geoprofiling (Rossmo) heat surface: almost certainly a '
                       'custom red/yellow scale, not any matplotlib colormap'),
    'galactic_corr': ('Galactic disk stellar elements correlation matrix xfe 1.png', 1200,
                       'viridis', 'heatmap', 'guess'),
    'radar_reflectivity_custom': ('Composite Reflectivity San Juan radar 92017.png', 1200,
                                   'NOT-MPL', 'map', 'NWS radar reflectivity scale: a fixed '
                                   'meteorological palette, not a matplotlib colormap'),
    'hunter_gatherer_pop': ('Simple estimation of hunter gatherer population of ancient near '
                             'east from rainfall and land wetness estimation 17000 bp 1.png',
                             1200, 'viridis', 'map', 'guess'),
    'gaussian_surface': ('Gaussian 2d surface.png', 1200, 'viridis', 'surface', 'guess'),
}


def fetch(name):
    title, width, cmap, cat, note = CORPUS[name]
    path = os.path.join(CACHE, name + '.png')
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    info = real_corpus._api({'action': 'query', 'format': 'json',
                             'titles': f'File:{title}', 'prop': 'imageinfo',
                             'iiprop': 'url|extmetadata', 'iiurlwidth': str(width)})
    page = next(iter(info['query']['pages'].values()))
    if 'imageinfo' not in page:
        print(f'  MISSING: {title!r} not found on Commons', file=sys.stderr)
        return None
    ii = page['imageinfo'][0]
    req = urllib.request.Request(ii.get('thumburl') or ii['url'],
                                 headers={'User-Agent': real_corpus.UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as fh:
            data = fh.read()
    except Exception as e:
        print(f'  FETCH FAILED: {title!r}: {e}', file=sys.stderr)
        return None
    with open(path, 'wb') as out:
        out.write(data)
    meta = ii.get('extmetadata', {})
    with open(os.path.join(CACHE, 'CREDITS.txt'), 'a') as fh:
        fh.write(f'{name}: File:{title}\n'
                 f'    artist:  {real_corpus._strip_html(meta.get("Artist", {}).get("value"))}\n'
                 f'    licence: {real_corpus._strip_html(meta.get("LicenseShortName", {}).get("value"))}\n'
                 f'    source:  https://commons.wikimedia.org/wiki/File:'
                 f'{urllib.parse.quote(title.replace(" ", "_"))}\n')
    return path


if __name__ == '__main__':
    for name in CORPUS:
        p = fetch(name)
        print(name, '->', p)
