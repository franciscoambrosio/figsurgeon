"""Demo: "replace the red with the orange of her shirt" -- a text parser cannot resolve
this alone, since it references a region rather than a colour name. Shows how it's done
instead.
Run: python examples/demo_replace_colour.py
"""
import os
from skimage import data
from PIL import Image
from figsurgeon import photo as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT, exist_ok=True)

im = Image.fromarray(data.astronaut())

# Step 1: the text parser cannot resolve a region reference, so it raises instead of guessing.
from figsurgeon.photo_describe import parse
try:
    parse("I want the red replaced by the orange of her shirt")
except ValueError as e:
    print('Text parser (correctly) cannot resolve this:')
    print(' ', e)
    print()

# Step 2: what actually resolves it -- sampling the real region.
# (The box comes from looking at the photo, the same way a person would point.)
suit_colour = P.sample_colour(im, box=(140, 250, 220, 320))
print(f'Sampled suit colour: {suit_colour}')

out, mask = P.replace_colour(im, from_colour=(200, 30, 30), to_colour=suit_colour, hue_tol=0.05)
im.save(os.path.join(OUT, 'demo_replace_orig.png'))
out.save(os.path.join(OUT, 'demo_replace_edit.png'))
print('Saved examples/output/demo_replace_orig.png / demo_replace_edit.png')
