"""Ask a hosted model to edit an image, through OpenRouter's one OpenAI-shaped endpoint.

Kept apart from `background_drift.py` because the two answer different questions: this one
gets a picture back from a vendor, that one grades it. Every return is written to disk with
the request that produced it, so a number in a table can be traced to the exact call.

The key comes from OPENROUTER_API_KEY, or from a gitignored `.env.local` beside the package
-- the same precedence `agent.py` uses for Anthropic. It is never printed or logged.
"""
import base64
import io
import json
import os
import time
import urllib.request

from PIL import Image

ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The cheap end of the image editors, and the default the hybrid path runs on: priced at a
# fifth of `openai/gpt-5.4-image-2`, and (unlike the gpt-image family) returns the frame it
# was given rather than a fixed size, which `hybrid.py` relies on to re-register the result.
DEFAULT_MODEL = 'google/gemini-3.1-flash-lite-image'


def _key():
    k = os.environ.get('OPENROUTER_API_KEY')
    if k:
        return k
    path = os.path.join(ROOT, '.env.local')
    if os.path.exists(path):
        for line in open(path):
            name, _, value = line.strip().partition('=')
            if name == 'OPENROUTER_API_KEY':
                return value.strip().strip('"\'')
    raise SystemExit('no OPENROUTER_API_KEY in the environment or .env.local')


def edit(image, instruction, model=DEFAULT_MODEL, seed=None, timeout=600, fmt='JPEG'):
    """Return (edited_image, meta). `image` is a PIL image; the return is whatever came back.

    No resizing is done here. What the model chose to return -- its size, its aspect ratio --
    is evidence, and normalising it at the door would destroy the most visible finding.
    """
    buf = io.BytesIO()
    image.convert('RGB').save(buf, fmt, **({'quality': 88} if fmt == 'JPEG' else {}))
    b64 = base64.b64encode(buf.getvalue()).decode()

    body = {'model': model, 'modalities': ['image', 'text'],
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': instruction},
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/{fmt.lower()};base64,' + b64}}]}]}
    if seed is not None:
        body['seed'] = seed

    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + _key(),
                                          'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        return None, {'model': model, 'error': f'HTTP {e.code}',
                      'detail': e.read()[:500].decode('utf-8', 'replace')}
    # A 200 doesn't mean a picture: a refusal, provider error or moderation block all come
    # back shaped differently, and reaching straight for ['choices'] would raise KeyError.
    if 'choices' not in r:
        return None, {'model': model, 'seconds': round(time.time() - t0, 1),
                      'error': str(r.get('error', r))[:500]}
    msg = r['choices'][0]['message']
    images = msg.get('images') or []
    meta = {'model': model, 'seed': seed, 'instruction': instruction,
            'seconds': round(time.time() - t0, 1),
            'usage': r.get('usage', {}), 'sent': list(image.size),
            'refusal': msg.get('refusal'),
            'text': (msg.get('content') or '')[:400]}
    if not images:
        return None, meta
    raw = base64.b64decode(images[0]['image_url']['url'].split(',', 1)[1])
    out = Image.open(io.BytesIO(raw))
    meta['returned'] = list(out.size)
    return out, meta


def run(image_path, instruction, models, out_dir, seed=None):
    """Edit one image with several models, saving each return and its meta next to it."""
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(os.path.expanduser(image_path)).convert('RGB')
    stem = os.path.splitext(os.path.basename(image_path))[0]
    done = []
    for model in models:
        slug = model.replace('/', '_')
        edited, meta = edit(img, instruction, model, seed=seed)
        if edited is None:
            why = meta.get('error') or meta.get('refusal') or meta.get('text') or 'no image'
            print(f'{model}: NO IMAGE -- {str(why)[:200]}')
            json.dump(meta, open(os.path.join(
                out_dir, f'{stem}__{model.replace("/", "_")}.failed.json'), 'w'), indent=2)
            continue
        path = os.path.join(out_dir, f'{stem}__{slug}.png')
        edited.save(path)
        json.dump(meta, open(path.replace('.png', '.json'), 'w'), indent=2)
        cost = meta['usage'].get('cost')
        print(f'{model}: {meta["returned"][0]}x{meta["returned"][1]} in {meta["seconds"]}s, '
              f'${cost} -> {path}')
        done.append((model, path, meta))
    return done
