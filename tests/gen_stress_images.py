"""Generates a stress-test image set: real photos, uncommon chart types and a
dark-background figure, spanning categories outside the package's normal test coverage."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from PIL import Image
rng = np.random.default_rng(11)

def save(a, n):
    if a.ndim==2: a = np.stack([a]*3,axis=-1)
    if a.shape[2]==4: a=a[:,:,:3]
    Image.fromarray(a.astype('uint8')).save(n)

# real photos (varied subjects/lighting)
save(data.astronaut(), 'p1_portrait.png')
save(data.coffee(), 'p2_object.png')
save(data.chelsea(), 'p3_animal.png')
save(data.rocket(), 'p4_lowlight.png')
save(data.hubble_deep_field(), 'p5_astro.png')
save(data.immunohistochemistry(), 'p6_medical.png')

# chart types outside normal coverage
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
x = np.arange(6)
ax.barh(x, rng.uniform(2,9,6), color='#8B0000')
ax.set_yticks(x); ax.set_yticklabels([f'Item {i}' for i in range(6)])
ax.set_xlabel('Score'); ax.set_title('Horizontal bars')
fig.tight_layout(); fig.savefig('c1_barh.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(5,5), dpi=100)
ax.pie(rng.uniform(1,5,5), labels=['A','B','C','D','E'], autopct='%1.0f%%')
ax.set_title('Pie chart')
fig.savefig('c2_pie.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(6,4), dpi=100)
m = rng.normal(0,1,(12,12))
im_ = ax.imshow(m, cmap='RdBu_r'); fig.colorbar(im_)
ax.set_title('Diverging heatmap (RdBu_r)')
fig.tight_layout(); fig.savefig('c3_heatmap.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(7,3), dpi=100)
t = np.linspace(0,20,500)
for lab,c in [('alpha','#e41a1c'),('beta','#377eb8'),('gamma','#4daf4a')]:
    ax.plot(t, np.sin(t*rng.uniform(0.5,1.5))+rng.normal(0,0.1,500), label=lab, color=c)
ax.legend(ncol=3, loc='upper center'); ax.set_title('Dense timeseries')
fig.tight_layout(); fig.savefig('c4_timeseries.png'); plt.close(fig)

# dark-background figure: other tests assume a white background
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
ax.scatter(rng.normal(0,1,200), rng.normal(0,1,200), c='#00FFAA', s=18)
ax.set_title('Dark theme scatter'); ax.set_xlabel('x'); ax.set_ylabel('y')
fig.tight_layout(); fig.savefig('c5_dark.png'); plt.close(fig)
plt.style.use('default')
print('generated')
