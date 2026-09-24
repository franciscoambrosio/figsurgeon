"""Eight boxes drawn by hand around eight obvious subjects on photographs the package
had never seen, each segmented and the mask looked at as an overlay, since passing the
coverage check does not mean the mask is the object. Run after any change to
segmentation and look at the sheets; the printed numbers alone cannot tell you that.

    python evals/segmentation_models/eight_boxes.py      # needs figsurgeon[grounding]

SlimSAM (`backend='auto'`) got seven of eight; GrabCut got one of eight. The exception
is the guitar, where the mask also takes the player's fretting hand and forearm (a
second object inside the box, no tighter box available). Also runs the phrase check
(`verify_photo.check_object_recoloured`) on each edit.
"""
import os
import sys
import urllib.parse
import urllib.request

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from evals import real_corpus                                            # noqa: E402

CACHE = os.path.join(ROOT, 'evals', '_corpus', 'eight_boxes')
SHEETS = os.path.join(HERE, 'sheets_eight_boxes')

# Boxes below are in these thumbnail coordinates.
WORKING = 1200

# title on Commons, the box, and the caller's words for what is in it.
CASES = [
    ('dog_shepherd', '20110425 German Shepherd Dog 8505.jpg',
     (150, 30, 1060, 740), 'the dog'),
    ('kingfisher', 'Common kingfisher on a branch opening its wings.jpg',
     (480, 130, 1010, 600), 'the bird'),
    ('backpack_red', 'Fjallraven, OutDoor 2018, Friedrichshafen (1X7A0438).jpg',
     (110, 25, 880, 1160), 'the backpack'),
    ('motorcycle', 'Honda CMX500 Rebel parked at Hogs for Hospice, Leamington, Ontario, 2025-08-02.jpg',
     (175, 60, 1060, 745), 'the motorcycle'),
    ('boat_sunset', 'Sailing boat at sunset, Ionian Sea, Albania.jpg',
     (35, 415, 345, 840), 'the sailing boat'),
    ('guitarist', 'Sam Amidon-1160199.jpg',
     (150, 570, 740, 890), 'the guitar'),
    ('chairs_blue', 'Santorin (GR), Fira -- 2017 -- 2598.jpg',
     (705, 430, 975, 700), 'the chair on the right'),
    ('horse_white', 'White horse in field.jpg',
     (245, 140, 640, 1075), 'the horse'),
]

# What each mask was judged to contain, to catch regressions as disagreements, not drift.
LOOKED_AT = {
    'dog_shepherd': 'the dog, tail to front paws',
    'kingfisher': 'the bird, open beak and both wings; the branch excluded',
    'backpack_red': 'the bag, both handles, the background between them resolved',
    'motorcycle': 'the whole bike incl. spoked wheels, plate and mirror; railing excluded',
    'boat_sunset': 'hull, mast, boom and most of the rigging, plus a few specks of sky',
    'guitarist': 'THE GUITAR PLUS THE FRETTING HAND AND FOREARM -- the one failure',
    'chairs_blue': 'that chair only; the second chair and the table left alone',
    'horse_white': 'the horse down to the hooves, mane and tail included',
}


def image(name, title):
    """The photograph, cached under evals/_corpus/eight_boxes/ (gitignored, like the rest).

    Deliberately NOT added to `real_corpus.CORPUS`: several evals iterate that corpus, and
    eight more images would silently change what they measure.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + '.jpg')
    if not (os.path.exists(path) and os.path.getsize(path) > 1000):
        info = real_corpus._api({'action': 'query', 'format': 'json',
                                 'titles': f'File:{title}', 'prop': 'imageinfo',
                                 'iiprop': 'url|extmetadata', 'iiurlwidth': str(WORKING)})
        page = next(iter(info['query']['pages'].values()))
        if 'imageinfo' not in page:
            raise LookupError(f'Commons has no file named {title!r}')
        ii = page['imageinfo'][0]
        req = urllib.request.Request(ii.get('thumburl') or ii['url'],
                                     headers={'User-Agent': real_corpus.UA})
        with urllib.request.urlopen(req, timeout=120) as fh:
            data = fh.read()
        with open(path, 'wb') as out:
            out.write(data)
        meta = ii.get('extmetadata', {})
        with open(os.path.join(CACHE, 'CREDITS.txt'), 'a') as fh:
            fh.write(f'{name}: File:{title}\n'
                     f'    artist:  {real_corpus._strip_html(meta.get("Artist", {}).get("value"))}\n'
                     f'    licence: {real_corpus._strip_html(meta.get("LicenseShortName", {}).get("value"))}\n'
                     f'    source:  https://commons.wikimedia.org/wiki/File:'
                     f'{urllib.parse.quote(title.replace(" ", "_"))}\n')
    img = Image.open(path).convert('RGB')
    img.thumbnail((WORKING, WORKING), Image.LANCZOS)
    return img


def run(grade=True):
    from figsurgeon import locate, objects as O
    os.makedirs(SHEETS, exist_ok=True)
    print(f'{"case":14s} {"backend":9s} {"cover":>6s} {"outside":>8s}  what the mask took')
    for name, title, box, subject in CASES:
        img = image(name, title)
        backend = O.choose_backend(img)
        seg = O.segment_object(img, box)
        mask, coverage = seg

        # Crop to the box: judging from full-frame thumbnails got masks backwards before.
        x0, y0, x1, y1 = box
        pad = 40
        sheet = locate.mask_overlay(img, mask).crop(
            (max(0, x0 - pad), max(0, y0 - pad),
             min(img.size[0], x1 + pad), min(img.size[1], y1 + pad)))
        sheet.save(os.path.join(SHEETS, f'{name}.jpg'), quality=85, optimize=True)
        print(f'{name:14s} {backend:9s} {coverage:6.2f} {seg.outside_frac:7.1%}   '
              f'{LOOKED_AT[name]}')

        if grade:
            out, _ = O.recolour_object(img, box=box, to_rgb=(40, 90, 200))
            out.save(os.path.join(SHEETS, f'{name}_recoloured.jpg'), quality=85)
            from figsurgeon import verify_photo as V
            ok, note = V.check_object_recoloured(img, out, box, subject=subject)
            verdict = {True: 'verified', False: 'FAILED', None: 'no verdict'}[ok]
            print(f'{"":14s} {"":9s} {verdict:>15s}   {note}')

    print(f'\nsheets in {SHEETS} -- LOOK AT THEM. One of eight was the object in 2026-08 '
          f'(GrabCut); seven of eight on 2026-09-15 (SlimSAM).')


if __name__ == '__main__':
    run(grade='--no-grade' not in sys.argv)
