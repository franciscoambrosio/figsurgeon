"""Demo for figsurgeon.objects -- run: python examples/demo_objects.py
Requires: pip install "figsurgeon[demo,advanced]"
"""
import os
from skimage import data
from PIL import Image
from figsurgeon import objects as O

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT, exist_ok=True)

# 1. recolour_object: the astronaut's suit turns blue. The box (60,180,420,512) also
# contains a corner of the American flag's red stripes (a red-stripe pixel at x=69,y=180
# falls inside this box) -- replace_colour (hue-based) would recolour the flag too;
# recolour_object (spatial, GrabCut-segmented) only touches what it finds inside the box
# as the object, so the flag survives untouched.
astro = Image.fromarray(data.astronaut())
recoloured, coverage = O.recolour_object(astro, box=(60, 180, 420, 512), to_rgb=(30, 60, 180))
print(f'  recolour_object: {coverage:.0%} of the box classified as the suit')
astro.save(os.path.join(OUT, 'obj_1_astronaut_orig.png'))
recoloured.save(os.path.join(OUT, 'obj_1_astronaut_edit.png'))

# 2a. erase_object: a thin mast against smooth sky, ~1.5% of the frame. No size warning;
# inspect obj_2a_edit.png and the mast is gone.
rocket = Image.fromarray(data.rocket())
mast_erased, frac_a, warning_a = O.erase_object(rocket, box=(150, 80, 230, 340))
print(f'  erase_object (mast): {frac_a:.1%} of frame, warning={warning_a!r}')
rocket.save(os.path.join(OUT, 'obj_2a_rocket_orig.png'))
mast_erased.save(os.path.join(OUT, 'obj_2a_rocket_edit.png'))

# 2b. erase_object: same function, a coffee cup at ~34% of the frame. Inpainting has no
# information about what was behind a large object, so this smears visibly (a radial
# streak pattern).
coffee = Image.fromarray(data.coffee())
cup_erased, frac_b, warning_b = O.erase_object(coffee, box=(80, 30, 420, 340))
print(f'  erase_object (cup): {frac_b:.1%} of frame, warning={"set" if warning_b else None}')
coffee.save(os.path.join(OUT, 'obj_2b_coffee_orig.png'))
cup_erased.save(os.path.join(OUT, 'obj_2b_coffee_edit.png'))

print('done -- see examples/output/obj_*.png')
