"""Local playground for figsurgeon: edit a photograph, in plain language or by box, and
see each operation's own verdict next to the picture.

Start with (from the repo root): .venv/bin/python -m playground.server
then open http://127.0.0.1:8420/. Localhost only -- no auth, no CORS.

The session is an `ImageWorkspace`, held in one module global (single-user tool).
Plain language runs the real agent loop (`figsurgeon.agent.edit`) over OpenRouter on
that same workspace; needs OPENROUTER_API_KEY in the environment or `.env.local`.

Endpoints:
  GET  /api/photos            -> {"photos": [...]}
  GET  /api/state?name=X      -> the session for that photo (creating it if needed)
  POST /api/edit               {"image", "box", "op", "subject", "to_rgb"}
  POST /api/instruct           {"image", "text", "model"}
  POST /api/undo               {"image", "steps"}
  POST /api/redo               {"image", "steps"}
  POST /api/reset              {"image"}
"""
import base64
import io
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Make sure the package imported is THIS checkout's copy, not anything on a global path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from figsurgeon import workspace

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'evals', '_corpus')
WORKING_SIZE = 1200  # long side, px
DEFAULT_MODEL = 'z-ai/glm-5.3-flash'

_CACHE = {}                          # name -> the downscaled original
# `redo` holds Steps popped off the workspace's history by undo, newest last, so
# going forward replays a stored image rather than re-running the tool.
_SESSION = {'name': None, 'ws': None, 'redo': []}

# Operations the box UI can build a call for; plain language reaches every tool.
_OPS = ('erase_object', 'recolour_object', 'isolate_object', 'extract_object',
       'scale_object', 'generative_fill')


def _list_photos():
    out = []
    for root, _dirs, files in os.walk(CORPUS):
        for f in sorted(files):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                out.append(os.path.relpath(os.path.join(root, f), CORPUS))
    return sorted(out)


def _load_working(name):
    """The downscaled original, at the size the browser displays -- a box dragged on
    screen is already in the library's coordinate space, no transform needed."""
    if name not in _CACHE:
        path = os.path.abspath(os.path.join(CORPUS, name))
        if not path.startswith(os.path.abspath(CORPUS) + os.sep):
            raise ValueError('bad photo name')
        img = Image.open(path).convert('RGB')
        w, h = img.size
        scale = WORKING_SIZE / max(w, h)
        if scale < 1:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                             Image.LANCZOS)
        _CACHE[name] = img
    return _CACHE[name]


def _session(name):
    """The workspace for `name`, created on first use. Switching photographs starts a new
    one -- history belongs to the picture it was made on."""
    if _SESSION['name'] != name or _SESSION['ws'] is None:
        _SESSION['name'] = name
        _SESSION['ws'] = workspace.ImageWorkspace(_load_working(name))
        _SESSION['redo'].clear()
    return _SESSION['ws']


def do_undo(ws, steps=1):
    """Walk back, keeping what was undone so it can be walked forward again."""
    steps = max(1, int(steps))
    going = list(ws.history[-steps:])            # oldest first, as history stores them
    note = ws.undo(steps).note
    for step in reversed(going):                 # newest ends up on top of the stack
        _SESSION['redo'].append(step)
    return _state(ws, note=note)


def do_redo(ws, steps=1):
    """Re-apply an undone step by restoring the image it produced, not recomputing it --
    the Step carries its own `after` and verdict, so a non-deterministic tool can't
    come back different."""
    n = 0
    for _ in range(max(1, int(steps))):
        if not _SESSION['redo']:
            break
        step = _SESSION['redo'].pop()
        ws.image = step.after
        ws.history.append(step)
        n += 1
    if not n:
        return _state(ws, note='nothing to redo')
    return _state(ws, note=f'redid {n} edit{"" if n == 1 else "s"}; '
                           f'forward to {step.name!r}')


def _png_data_uri(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')      # alpha survives (extract_object)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


# Verdict states the panel renders: PASS/FAIL from ok is True/False; ok is None splits
# into "no verdict" (check didn't run) vs "undecided" (a designed abstention).
_NO_VERDICT_MARKERS = (
    'is not checked', 'no automatic check', 'no check for', 'could not run',
    'no region given', 'no usable region', 'check skipped', 'not checked',
)


def _classify(ok, detail):
    if ok is True:
        return 'PASS'
    if ok is False:
        return 'FAIL'
    if any(m in (detail or '') for m in _NO_VERDICT_MARKERS):
        return 'NO_VERDICT'
    return 'UNDECIDED'


def _state(ws, note=None):
    """Everything the page needs to redraw, after any action."""
    return {
        'ws_id': id(ws), 'pid': os.getpid(),
        'image': _png_data_uri(ws.image),
        'width': ws.image.width,
        'height': ws.image.height,
        'note': note,
        'redo': len(_SESSION['redo']),
        'history': [{'name': s.name, 'args': s.args, 'note': s.note, 'ok': s.ok,
                     'state': _classify(s.ok, s.detail), 'detail': s.detail,
                     'trace': getattr(s, 'trace', None)}
                    for s in ws.history],
    }


def do_edit(ws, box, op, subject, to_rgb, scale=None, instruction=None,
           mask_to_object=None):
    if op not in _OPS:
        raise ValueError(f'unknown op {op!r}')
    args = {'box': [int(v) for v in box]}
    subject = (subject or '').strip()
    if subject:
        # so the box path and the LLM path are graded by the same evidence
        args['subject'] = subject
    if op == 'recolour_object':
        if not to_rgb:
            raise ValueError('recolour needs a target colour')
        args['to_rgb'] = [int(v) for v in to_rgb]
    elif op == 'scale_object':
        args.pop('subject', None)     # scale_object is box-only, no subject arg
        args['scale'] = float(scale) if scale not in (None, '') else 1.0
    elif op == 'generative_fill':
        if not (instruction or '').strip():
            raise ValueError('generative fill needs an instruction')
        args.pop('subject', None)
        args['instruction'] = instruction.strip()
        if mask_to_object:
            args['mask_to_object'] = True
    result = ws.apply(op, args)
    # An edit after undo abandons the forward history -- redo would paste a step from
    # a different past onto this one.
    _SESSION['redo'].clear()
    return _state(ws, note=result.note)


def _collapse(text, inner, transcript):
    """Collapse the instruction's surviving tool calls into ONE undoable step, so one
    sentence of plain language costs one undo rather than one per tool call. The verdict
    is the worst of the survivors, not the last one, so a failed check isn't buried under
    a later PASS on a different tool."""
    ok = False if any(s.ok is False for s in inner) else (
        True if any(s.ok is True for s in inner) else None)
    failed = next((s for s in inner if s.ok is False), None)
    worst = failed or inner[-1]
    step = workspace.Step(text, {'instruction': text},
                          f'{len(inner)} edit(s) from: {text!r}',
                          inner[0].before, inner[-1].after, ok,
                          f'{worst.name}: {worst.detail}' if worst.detail else '')
    # Not a Step field: the box path has no trace. `redo` re-pushes this object,
    # so the trace travels forward with it.
    step.trace = [{'kind': name, 'text': note} if name in ('assistant', 'refused')
                  else {'kind': 'tool', 'name': name, 'args': args, 'text': note}
                  for name, args, note in transcript]
    return step


def do_instruct(ws, text, model):
    """The real agent loop, on this session's workspace, over OpenRouter."""
    from figsurgeon import openrouter
    from figsurgeon.agent import edit as agent_edit

    # `_key()` raises SystemExit (a BaseException), which would otherwise skip past
    # the handler's `except Exception` and kill the worker thread.
    try:
        client = openrouter.client()
        client.messages  # noqa: B018 -- touch it so a missing key fails here, not mid-loop
        openrouter._key()
    except SystemExit as e:
        raise RuntimeError(f'{e} -- plain language needs an OpenRouter key; the box '
                           f'controls work without one') from None

    # The LAST step object before the call, not its index: `apply()` pop(0)s the oldest
    # step past `max_history`, which shifts indices but not identity.
    last_old = ws.history[-1] if ws.history else None
    _SESSION['redo'].clear()          # same divergence rule as the box path
    _img, transcript = agent_edit(ws.image, text, client=client,
                                  model=model or DEFAULT_MODEL, workspace=ws)
    said = [row[2] for row in transcript if row[0] in ('assistant', 'refused')]
    if last_old is None:
        inner = list(ws.history)
    else:
        try:
            idx = next(i for i, s in enumerate(ws.history) if s is last_old)
            inner = ws.history[idx + 1:]
        except StopIteration:
            # boundary step itself was evicted by max_history -- everything left is new
            inner = list(ws.history)
    if inner:
        del ws.history[len(ws.history) - len(inner):]
        ws.history.append(_collapse(text, inner, transcript))
    note = f'{len(inner)} edit(s) from: {text!r}'
    if said:
        note += ' | model: ' + ' '.join(said)
    return _state(ws, note=note)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ('/', '/index.html'):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')
            with open(path, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == '/api/photos':
            self._json({'photos': _list_photos(), 'default_model': DEFAULT_MODEL})
            return
        if parsed.path == '/api/state':
            name = (parse_qs(parsed.query).get('name') or [''])[0]
            try:
                self._json(_state(_session(name)))
            except Exception as e:
                self._json({'error': f'{type(e).__name__}: {e}'}, status=400)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        route = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        try:
            payload = json.loads(self.rfile.read(length) or b'{}')
            ws = _session(payload.get('image'))
            if route == '/api/edit':
                out = do_edit(ws, payload['box'], payload['op'], payload.get('subject'),
                              payload.get('to_rgb'), payload.get('scale'),
                              payload.get('instruction'), payload.get('mask_to_object'))
            elif route == '/api/instruct':
                out = do_instruct(ws, payload.get('text', ''), payload.get('model'))
            elif route == '/api/undo':
                out = do_undo(ws, payload.get('steps', 1))
            elif route == '/api/redo':
                out = do_redo(ws, payload.get('steps', 1))
            elif route == '/api/reset':
                note = ws.reset().note
                _SESSION['redo'].clear()
                out = _state(ws, note=note)
            else:
                self.send_response(404)
                self.end_headers()
                return
        except Exception as e:
            traceback.print_exc()
            self._json({'error': f'{type(e).__name__}: {e}'}, status=400)
            return
        self._json(out)


def main():
    port = int(os.environ.get('PLAYGROUND_PORT', 8420))
    print(f'figsurgeon playground: http://127.0.0.1:{port}/')
    print('first model-backed op loads CLIPSeg + SlimSAM weights -- that run is slow.')
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


if __name__ == '__main__':
    main()
