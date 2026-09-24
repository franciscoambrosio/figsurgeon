"""A corpus of real photographs, charts and screenshots, fetched from Wikimedia Commons.

Other evals mostly run on scikit-image teaching samples: small, clean, uncompressed. Real
input differs in ways that change what the code has to do: much larger (megapixels, not
kilopixels), JPEG compression artifacts, EXIF orientation flags, and real subjects (hair,
motion blur, shadows, backgrounds that share the subject's colour).

Images are downloaded once and cached in `evals/_corpus/` (gitignored: other people's work
under their own licences). Attribution and licence for every file are recorded in
`_corpus/CREDITS.txt` by the fetch.

    python evals/real_corpus.py          # fetch everything, print the manifest
"""
import json
import os
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_corpus')

# Commons asks for a descriptive User-Agent identifying the tool and a contact.
UA = ('figsurgeon-eval/1.0 '
      '(https://github.com/franciscoambrosio/figsurgeon; image-editing test suite)')

# Curated by looking at each one, not taken from a search ranking. `width` is the cached
# render width -- large where being a big image is the point, smaller where the content
# (a chart, a screenshot) has a natural size.
CORPUS = {
    'food_pizza': ('Margherita pizza on plate.jpg', 2400,
                   'phone food photo: warm wood, JPEG artifacts, shallow DoF already'),
    'product_shoe': ('Product RED Converse (4034111954).jpg', 2000,
                     'red shoes on black with large text: a product shot where the '
                     'background is NOT plain and a hue mask has an easy target'),
    'product_white': ('Superga White new 05.jpg', 1024,
                      'white sneakers on a mid-grey textured floor -- a WHITE product on a '
                      'neutral ground, which is where cutouts actually get hard'),
    'car_red': ('Red car parked in front of Wupaochun Bakery Taichung on 29 January 2021.jpg',
                2400, 'the classic "make the car blue": the car is SMALL in a busy street '
                      'scene, which is the realistic version of that request'),
    'portrait_studio': ('Russia, Moscow Oblast. Young woman with umbrella, studio portrait.jpg',
                        1600, 'studio portrait: hair detail, and a BLUE umbrella against a '
                              'BLUE backdrop -- hue matching cannot separate them'),
    'street_people': ('Pedestrians on Caroline Street (daytime), Saratoga Springs, New York.jpg',
                      2400, 'street scene: many objects, no single subject'),
    'flower_macro': ('Gazania petals (macro) - Ashbury, 2007 1.jpg', 2000,
                     'macro flower: saturated colour, real bokeh'),
    'night_city': ('Édifice Price at night, Quebec city, Canada.jpg', 2400,
                   'night photograph: real sensor noise, deep shadows, point lights'),
    'ui_screenshot': ('Files (software) screenshot.png', 1600,
                      'real application screenshot: flat UI, small text, no subject'),
    'chart_bar': ('Fish-catch-sector-bar (OWID 0438).png', 850,
                  'real published bar chart with a legend and axis labels'),
    'chart_health': ('Health indicators Bar chart.png', 600,
                     'a second real chart, lower resolution'),
    'city_wide': ('Midtown Manhattan from Jersey City September 2020 HDR.jpg', 2400,
                  'wide cityscape: fine repeated structure, sky gradient'),
}


def _api(params):
    url = 'https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.load(fh)


def _strip_html(value):
    import re
    return re.sub(r'<[^>]+>', '', value or '').strip()


def fetch(name, verbose=True):
    """Return the local path for one corpus image, downloading it if not already cached."""
    title, width, _ = CORPUS[name]
    ext = os.path.splitext(title)[1].lower()
    path = os.path.join(CACHE, f'{name}{ext}')
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path

    os.makedirs(CACHE, exist_ok=True)
    info = _api({'action': 'query', 'format': 'json', 'titles': f'File:{title}',
                 'prop': 'imageinfo', 'iiprop': 'url|extmetadata|size',
                 'iiurlwidth': str(width)})
    pages = info.get('query', {}).get('pages', {})
    page = next(iter(pages.values()))
    if 'imageinfo' not in page:
        raise LookupError(f'Commons has no file named {title!r}')
    ii = page['imageinfo'][0]
    meta = ii.get('extmetadata', {})

    req = urllib.request.Request(ii.get('thumburl') or ii['url'], headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=120) as fh:
        data = fh.read()
    with open(path, 'wb') as out:
        out.write(data)

    credit = (f'{name}: File:{title}\n'
              f'    artist:  {_strip_html(meta.get("Artist", {}).get("value"))}\n'
              f'    licence: {_strip_html(meta.get("LicenseShortName", {}).get("value"))}\n'
              f'    source:  https://commons.wikimedia.org/wiki/File:'
              f'{urllib.parse.quote(title.replace(" ", "_"))}\n')
    with open(os.path.join(CACHE, 'CREDITS.txt'), 'a') as fh:
        fh.write(credit)
    if verbose:
        print(f'  fetched {name:17s} {len(data)/1e6:5.1f} MB  '
              f'{_strip_html(meta.get("LicenseShortName", {}).get("value"))}')
    return path


def load(name):
    """The corpus image as a PIL image, with EXIF orientation applied.

    A portrait photo is often stored landscape with an orientation flag; PIL's `open`
    doesn't apply it, so without this the image would be rotated 90 degrees from what a
    person sees, and a box read off it would be wrong in a way nothing downstream detects.
    """
    from PIL import Image, ImageOps
    img = Image.open(fetch(name, verbose=False))
    return ImageOps.exif_transpose(img).convert('RGB')


def fetch_all():
    print(f'fetching {len(CORPUS)} real images into {CACHE}')
    paths = {}
    for name in CORPUS:
        try:
            paths[name] = fetch(name)
        except Exception as e:
            print(f'  FAILED  {name}: {type(e).__name__}: {e}')
    return paths


if __name__ == '__main__':
    paths = fetch_all()
    print(f'\n{len(paths)}/{len(CORPUS)} cached. Manifest:')
    from PIL import Image, ImageOps
    for name, path in sorted(paths.items()):
        img = Image.open(path)
        oriented = ImageOps.exif_transpose(img)
        flag = ' (EXIF-rotated)' if oriented.size != img.size else ''
        mp = oriented.size[0] * oriented.size[1] / 1e6
        print(f'  {name:17s} {oriented.size[0]:5d}x{oriented.size[1]:<5d} '
              f'{mp:4.1f} MP  {img.format:4s}{flag}  -- {CORPUS[name][2]}')
    print(f'\ncredits in {os.path.join(CACHE, "CREDITS.txt")}')
    sys.exit(0 if len(paths) == len(CORPUS) else 1)
