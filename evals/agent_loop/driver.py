"""One tool call at a time, logged identically for every driver, so transcripts are
comparable and no driver grades its own work (`finish` records what it says it did;
pictures are judged separately).

    from evals.agent_loop.driver import Run
    run = Run('door')
    print(run.call('show_grid'))
    print(run.call('recolour_object', {'box': [...], 'to_rgb': [20, 70, 40],
                                       'subject': 'the front door'}))
    run.finish('recoloured the door, left the postbox alone')
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'out')
LOGS = os.path.join(HERE, 'logs')


def _cases():
    import sys
    sys.path.insert(0, ROOT)
    from evals.agent_loop.cases import CASES
    return {c['id']: c for c in CASES}


class Run:
    """Resumed from disk if one is already open for this case: pickled after every call
    so a driver can look at the picture between calls (a new process) without re-running
    everything before it."""

    def __init__(self, case_id, restart=False):
        import sys
        sys.path.insert(0, ROOT)
        from PIL import Image
        from figsurgeon.workspace import ImageWorkspace
        case = _cases()[case_id]
        self.case = case
        self.path = os.path.join(ROOT, case['image'])
        os.makedirs(OUT, exist_ok=True)
        os.makedirs(LOGS, exist_ok=True)
        self.state_path = os.path.join(LOGS, f'{case_id}.state')
        if os.path.exists(self.state_path) and not restart:
            import pickle
            with open(self.state_path, 'rb') as fh:
                self.ws, self.steps, self.started = pickle.load(fh)
            print(f'case {case_id}: {case["request"]!r}')
            print(f'resumed after {len(self.steps)} calls; image is '
                  f'{self.ws.image.size} {self.ws.image.mode}')
            return
        img = Image.open(self.path)
        img.load()
        self.ws = ImageWorkspace(img.convert('RGBA') if img.mode == 'RGBA' else img.convert('RGB'))
        self.steps = []
        self.started = time.time()
        self._checkpoint()
        print(f'case {case_id}: {case["request"]!r}')
        print(f'image {self.path} {self.ws.image.size} {self.ws.image.mode}')

    def _checkpoint(self):
        import pickle
        with open(self.state_path, 'wb') as fh:
            pickle.dump((self.ws, self.steps, self.started), fh)

    def call(self, tool, args=None):
        """Run one tool call. Returns the note; a preview is saved and its path printed."""
        t0 = time.time()
        result = self.ws.apply(tool, args or {})
        note = result[1] if isinstance(result, tuple) else str(result)
        verified = (getattr(result, 'data', None) or {}).get('verified', None)
        preview_path = None
        preview = getattr(result, 'preview', None)
        if preview is not None:
            preview_path = os.path.join(OUT, f'{self.case["id"]}_step{len(self.steps) + 1}_{tool}.png')
            preview.save(preview_path)
        self.steps.append({'step': len(self.steps) + 1, 'tool': tool, 'args': args or {},
                           'note': note, 'verified': verified,
                           'seconds': round(time.time() - t0, 1), 'preview': preview_path})
        self._checkpoint()
        if preview_path:
            print(f'[preview saved: {preview_path} -- LOOK at it]')
        return note

    def save(self, label='current'):
        path = os.path.join(OUT, f'{self.case["id"]}_{label}.png')
        self.ws.image.save(path)
        print(f'[saved: {path}]')
        return path

    def finish(self, said):
        """`said` is the driver's own account of what it did -- recorded, not believed."""
        final = self.save('final')
        log = {'case': self.case['id'], 'request': self.case['request'],
               'image': self.case['image'],
               'driver_said': said, 'final_image': final,
               'total_seconds': round(time.time() - self.started, 1),
               'steps': self.steps}
        with open(os.path.join(LOGS, f'{self.case["id"]}.json'), 'w') as fh:
            json.dump(log, fh, indent=2)
        print(f'[log written: {LOGS}/{self.case["id"]}.json -- {len(self.steps)} calls, '
              f'{log["total_seconds"]}s]')
        return final
