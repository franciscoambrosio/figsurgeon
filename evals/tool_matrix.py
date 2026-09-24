 
from skimage import data
from PIL import Image
import numpy as np
from figsurgeon.tools import dispatch

def get(n):
    a = getattr(data, n)()
    if a.ndim==2: a = np.stack([a]*3,axis=-1)
    if a.shape[2]==4: a = a[:,:,:3]
    return Image.fromarray(a.astype('uint8'))

IMAGES = ["astronaut","chelsea","coffee","rocket","coins","immunohistochemistry","hubble_deep_field","retina","colorwheel","brick","clock","camera","moon","text"]

CALLS = [
 ('isolate_colour', {'target_rgb':[220,30,30]}),
 ('replace_colour', {'from_rgb':[220,30,30],'to_rgb':[40,90,200]}),
 ('sample_colour', {'box':[10,10,60,60]}),
 ('blur_background', {'focus':'subject','radius':12}),
 ('remove_background', {}),
 ('replace_background', {'colour_rgb':[255,255,255]}),
 ('remove_object', {'box':[10,10,40,40]}),
 ('adjust', {'contrast':1.3,'saturation':1.2}),
 ('stylise', {'effect':'sepia'}),
 ('stylise', {'effect':'sketch'}),
 ('stylise', {'effect':'vignette','strength':0.6}),
 ('denoise', {'strength':6}),
 ('auto_white_balance', {}),
 ('crop', {'box':[10,10,100,100]}),
 ('rotate', {'angle':15}),
 ('flip', {'axis':'horizontal'}),
 ('resize', {'scale':0.5}),
 ('add_text', {'text':'test','position':[5,5]}),
]

fails, errs = [], []
for img_name in IMAGES:
    try: im = get(img_name)
    except Exception as e:
        errs.append((img_name,'LOAD',str(e)[:60])); continue
    for tool, args in CALLS:
        try:
            out, note = dispatch(tool, args, im)
            if note.startswith('error'):
                fails.append((img_name, tool, note[:70]))
            elif out is None:
                fails.append((img_name, tool, 'returned None'))
            elif out.size != im.size and tool not in ('remove_background','crop','rotate','resize'):
                fails.append((img_name, tool, f'size changed {im.size}->{out.size}'))
        except Exception as e:
            errs.append((img_name, tool, f'{type(e).__name__}: {str(e)[:60]}'))

print(f'ran {len(IMAGES)}x{len(CALLS)} = {len(IMAGES)*len(CALLS)} combinations')
print(f'graceful failures (note=error): {len(fails)}')
for f in fails[:15]: print('   ', f)
print(f'UNCAUGHT EXCEPTIONS: {len(errs)}')
for e in errs[:20]: print('   ', e)
