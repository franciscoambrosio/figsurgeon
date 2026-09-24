 
from skimage import data
from PIL import Image
import numpy as np
from figsurgeon.tools import dispatch
from figsurgeon import verify_photo as V

def get(n):
    a = getattr(data, n)()
    if a.ndim==2: a = np.stack([a]*3,axis=-1)
    if a.shape[2]==4: a = a[:,:,:3]
    return Image.fromarray(a.astype('uint8'))

IMAGES = ['astronaut','chelsea','coffee','rocket','coins','immunohistochemistry',
          'hubble_deep_field','retina','colorwheel','brick','clock','camera','moon']

CHECKS = [
 ('stylise', {'effect':'grayscale'}, lambda b,a: V.check_grayscale(b,a)),
 ('stylise', {'effect':'vignette','strength':0.6}, lambda b,a: V.check_vignette(b,a)),
 ('adjust', {'brightness':1.4}, lambda b,a: V.check_brightness(b,a,'up',factor=1.4)),
 ('adjust', {'brightness':0.6}, lambda b,a: V.check_brightness(b,a,'down',factor=0.6)),
 ('adjust', {'contrast':1.5}, lambda b,a: V.check_contrast(b,a,'up',factor=1.5)),
 ('adjust', {'saturation':1.6}, lambda b,a: V.check_saturation(b,a,'up',factor=1.6)),
 ('adjust', {'saturation':0.4}, lambda b,a: V.check_saturation(b,a,'down',factor=0.4)),
 ('denoise', {'strength':8}, lambda b,a: V.check_denoise(b,a)),
]

results = {}
for img_name in IMAGES:
    im = get(img_name)
    for tool, args, checker in CHECKS:
        key = f"{tool}:{args.get('effect') or list(args.keys())[0]}={list(args.values())[0]}"
        out, note = dispatch(tool, args, im)
        try:
            ok, detail = checker(im, out)
        except Exception as e:
            ok, detail = False, f'{type(e).__name__}: {e}'
        results.setdefault(key, []).append((img_name, ok, detail))

for key, rows in results.items():
    bad = [(n,d) for n,ok,d in rows if ok is False]
    skipped = [n for n,ok,d in rows if ok is None]
    status = 'OK ' if not bad else 'FAIL'
    print(f'{status} {key:34s} {len(rows)-len(bad)-len(skipped)}/{len(rows)} pass, {len(skipped)} skip')
    for n,d in bad[:4]:
        print(f'        FAILED on {n}: {d}')
