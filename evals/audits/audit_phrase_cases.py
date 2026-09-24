import os, sys
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from evals import real_corpus

EIGHT_CACHE = os.path.join(ROOT, 'evals', '_corpus', 'eight_boxes')
AUDIT_CACHE = os.path.join(ROOT, 'evals', '_corpus', 'audit_phrase')

EIGHT_NAMES = {
    'dog_shepherd': 'dog_shepherd.jpg', 'kingfisher': 'kingfisher.jpg',
    'backpack_red': 'backpack_red.jpg', 'motorcycle': 'motorcycle.jpg',
    'boat_sunset': 'boat_sunset.jpg', 'guitarist': 'guitarist.jpg',
    'chairs_blue': 'chairs_blue.jpg', 'horse_white': 'horse_white.jpg',
}
AUDIT_NAMES = {
    'bicycle': 'bicycle.jpg', 'cat_grass': 'cat_grass.jpg',
    'sheep_fence': 'sheep_fence.jpg', 'winter_branches': 'winter_branches.jpg',
}


def load_real(name):
    return real_corpus.load(name)


def load_eight(name):
    path = os.path.join(EIGHT_CACHE, EIGHT_NAMES[name])
    img = Image.open(path).convert('RGB')
    img.thumbnail((1200, 1200), Image.LANCZOS)
    return img


def load_audit(name):
    path = os.path.join(AUDIT_CACHE, AUDIT_NAMES[name])
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert('RGB')
    img.thumbnail((1200, 1200), Image.LANCZOS)
    return img


IMAGE_SOURCE = {}
for n in real_corpus.CORPUS:
    IMAGE_SOURCE[n] = ('real', n)
for n in EIGHT_NAMES:
    IMAGE_SOURCE[n] = ('eight', n)
for n in AUDIT_NAMES:
    IMAGE_SOURCE[n] = ('audit', n)


_LOADERS = {'real': load_real, 'eight': load_eight, 'audit': load_audit}
_cache = {}


def get_image(name):
    if name not in _cache:
        kind, key = IMAGE_SOURCE[name]
        _cache[name] = _LOADERS[kind](key)
    return _cache[name]


# (case_id, image, phrase, hard_kind, expected_referent[y/n], note-before-measuring)
CASES = [
    # --- real_corpus, two phrases each ---
    ('pizza_1', 'food_pizza', 'the pizza', 'plain', 'y', 'clear single subject, fills much of frame'),
    ('pizza_2', 'food_pizza', 'the plate', 'part-surround', 'y', 'plate visible under/around pizza'),
    ('shoe_1', 'product_shoe', 'the shoes', 'plain', 'y', 'clear product shot'),
    ('shoe_2', 'product_shoe', 'the shoelace', 'part-thin', 'y', 'thin part of a known object'),
    ('shoew_1', 'product_white', 'the sneakers', 'plain', 'y', 'clear product shot, white on grey'),
    ('shoew_2', 'product_white', 'the shoe sole', 'part-thin', 'y', 'thin/flat part, low contrast'),
    ('car_1', 'car_red', 'the car', 'small', 'y', 'car is small in a busy street scene, <5% of frame expected'),
    ('car_2', 'car_red', "the car's wheel", 'part-thin', 'y', 'small thin part of a small object'),
    ('portrait_1', 'portrait_studio', 'the woman', 'large', 'y', 'clear single subject, large in frame'),
    ('portrait_2', 'portrait_studio', 'the umbrella', 'plain', 'y', 'blue umbrella on blue background: hard to separate by colour but phrase should still work'),
    ('street_1', 'street_people', 'the pedestrians', 'ambiguous-plural', 'y', 'many people, plural phrase should get several of them'),
    ('street_2', 'street_people', 'the elephant', 'nothing', 'n', 'no elephant in a street scene: should match nothing'),
    ('flower_1', 'flower_macro', 'the flower', 'large', 'y', 'macro flower fills most of the frame, >60% expected'),
    ('flower_2', 'flower_macro', 'the flower centre', 'part-tricky', 'maybe', 'known-hard per code comments: a part of a thing that fills the frame'),
    ('night_1', 'night_city', 'the tower', 'plain', 'y', 'clear illuminated tower subject'),
    ('night_2', 'night_city', 'the clock', 'part-small', 'maybe', 'small part, may or may not be visible/legible'),
    ('ui_1', 'ui_screenshot', 'the icon', 'ambiguous-plural', 'maybe', 'many icons in a UI screenshot, ambiguous which one'),
    ('ui_2', 'ui_screenshot', 'the sidebar', 'plain', 'maybe', 'a UI region, not an object -- tests whether phrase grounding handles UI chrome'),
    ('chart_1', 'chart_bar', 'the bar', 'ambiguous-plural', 'maybe', 'many bars in a bar chart, ambiguous which one'),
    ('chart_2', 'chart_bar', 'the legend', 'part-small', 'maybe', 'small legend box within a chart'),
    ('city_1', 'city_wide', 'the sky', 'large-stuff', 'y', 'stuff category, should take a large contiguous region'),
    ('city_2', 'city_wide', 'the building', 'ambiguous-plural', 'maybe', 'many buildings in a wide cityscape, ambiguous which one'),

    # --- eight_boxes photographs, extra phrases beyond the original set ---
    ('dog_1', 'dog_shepherd', 'the dog', 'plain', 'y', 'known-good case from eight_boxes'),
    ('dog_2', 'dog_shepherd', "the dog's ear", 'part-thin', 'y', 'small part of a known object'),
    ('bird_1', 'kingfisher', 'the bird', 'plain', 'y', 'known-good case from eight_boxes'),
    ('bird_2', 'kingfisher', 'the branch', 'part-thin', 'y', 'thin branch the bird sits on'),
    ('backpack_1', 'backpack_red', 'the backpack', 'plain', 'y', 'known-good case from eight_boxes'),
    ('backpack_2', 'backpack_red', 'the zipper', 'part-tiny', 'maybe', 'tiny part of an object'),
    ('moto_1', 'motorcycle', 'the motorcycle', 'thin-skeletal', 'y', 'spoked wheels + railing behind at similar distance, known-hard'),
    ('moto_2', 'motorcycle', 'the motorcycle wheel', 'part-thin-skeletal', 'y', 'thin spoked wheel, part of object'),
    ('boat_1', 'boat_sunset', 'the sailing boat', 'thin-skeletal-small', 'y', 'mostly rigging, ~2% of frame, known-hard per docstring'),
    ('boat_2', 'boat_sunset', 'the mast', 'part-thin', 'y', 'thin vertical part of the boat'),
    ('guitar_1', 'guitarist', 'the guitar', 'occluded', 'y', "fretting hand/forearm occlude/overlap, known-hard (SlimSAM's one failure in eight_boxes)"),
    ('guitar_2', 'guitarist', 'the guitarist', 'large', 'y', 'the whole person, large in frame'),
    ('chairs_1', 'chairs_blue', 'the chair on the right', 'identical-multi', 'y', 'one of several similar chairs, disambiguated only by position'),
    ('chairs_2', 'chairs_blue', 'the table', 'plain', 'maybe', 'a table among the chairs'),
    ('horse_1', 'horse_white', 'the horse', 'large', 'y', 'known-good case from eight_boxes'),
    ('horse_2', 'horse_white', "the horse's tail", 'part-thin', 'y', 'thin part of a known object'),

    # --- new fetches for hard kinds not otherwise covered ---
    ('bike_1', 'bicycle', 'the bicycle', 'thin-skeletal', 'y', 'thin frame/spokes, clear single subject'),
    ('bike_2', 'bicycle', 'the bicycle wheel', 'part-thin-skeletal', 'y', 'spoked wheel, thin part of a thin object'),
    ('cat_1', 'cat_grass', 'the cat', 'plain', 'y', 'clear single subject'),
    ('cat_2', 'cat_grass', "the cat's ear", 'part-tiny', 'y', 'tiny part of a known object -- explicit hard case requested'),
    ('sheep_1', 'sheep_fence', 'the sheep', 'occluded', 'y', 'mostly behind a wire fence -- explicit hard case requested'),
    ('sheep_2', 'sheep_fence', 'the fence', 'thin-skeletal', 'y', 'thin wire fence in front of the sheep'),
    ('branch_1', 'winter_branches', 'the bare branches', 'thin-skeletal', 'y', 'bare branches against the sky -- explicit hard case requested'),
    ('branch_2', 'winter_branches', 'the sky', 'large-stuff', 'y', 'sky visible through the branches'),
]

if __name__ == '__main__':
    print(len(CASES), 'cases')
