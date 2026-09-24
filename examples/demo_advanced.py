"""Demo for figsurgeon.advanced -- run: python examples/demo_advanced.py
Requires: pip install "figsurgeon[demo,advanced]"
"""
import os
from skimage import data
from PIL import Image
import numpy as np
from figsurgeon import advanced as A

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT, exist_ok=True)

# 1. background removal, shown on a checkerboard so transparency is visible
im = Image.fromarray(data.chelsea())
cutout = A.remove_background(im)
checker = Image.new('RGB', cutout.size, (255, 255, 255))
px = np.array(checker)
for y in range(0, cutout.size[1], 16):
    for x in range(0, cutout.size[0], 16):
        if (x // 16 + y // 16) % 2 == 0:
            px[y:y + 16, x:x + 16] = (210, 210, 210)
Image.fromarray(px).save(os.path.join(OUT, 'adv_1_orig.png'))
comp = Image.fromarray(px)
comp.paste(cutout, (0, 0), cutout)
comp.save(os.path.join(OUT, 'adv_1_edit.png'))

# 2. replace_background
replaced = A.replace_background(im, (245, 235, 220))
im.save(os.path.join(OUT, 'adv_2_orig.png'))
replaced.save(os.path.join(OUT, 'adv_2_edit.png'))

# 3. object removal: the thin mast in the rocket photo, against smooth sky
rocket = Image.fromarray(data.rocket())
out = A.remove_object(rocket, [(192, 90), (207, 340)], radius=7)
rocket.save(os.path.join(OUT, 'adv_3_orig.png'))
out.save(os.path.join(OUT, 'adv_3_edit.png'))

# 4. denoise -- noise is added synthetically first, so the 'before' is a real comparison
clean = Image.fromarray(data.chelsea())
arr = np.array(clean).astype(float)
noisy = np.clip(arr + np.random.default_rng(0).normal(0, 22, arr.shape), 0, 255).astype(np.uint8)
noisy_im = Image.fromarray(noisy)
denoised = A.denoise(noisy_im, strength=10)
noisy_im.save(os.path.join(OUT, 'adv_4_orig.png'))
denoised.save(os.path.join(OUT, 'adv_4_edit.png'))

# 5a. white balance: neutral-content photo, synthetic tungsten cast -- succeeds
arr2 = np.array(clean).astype(float)
tungsten = arr2.copy()
tungsten[..., 0] *= 1.35
tungsten[..., 1] *= 1.05
tungsten[..., 2] *= 0.55
tungsten = np.clip(tungsten, 0, 255).astype(np.uint8)
tungsten_im = Image.fromarray(tungsten)
corrected, warn = A.auto_white_balance(tungsten_im)
print('  wb (neutral-content photo) warning:', warn)
tungsten_im.save(os.path.join(OUT, 'adv_5b_orig.png'))
corrected.save(os.path.join(OUT, 'adv_5b_edit.png'))

# 5b. white balance: naturally warm-dominated photo, same function -- fails
coffee = Image.fromarray(data.coffee())
carr = np.array(coffee).astype(float)
casted = carr.copy()
casted[..., 2] *= 1.5
casted[..., 0] *= 0.8
casted = np.clip(casted, 0, 255).astype(np.uint8)
casted_im = Image.fromarray(casted)
corrected2, warn2 = A.auto_white_balance(casted_im)
print('  wb (warm-dominant photo) warning:', warn2)
casted_im.save(os.path.join(OUT, 'adv_5_orig.png'))
corrected2.save(os.path.join(OUT, 'adv_5_edit.png'))

print('done -- see examples/output/adv_*.png')
