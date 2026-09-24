"""Drive a real model through the whole agent loop and log it like `driver.py` does.

    .venv/bin/python evals/agent_loop/run_models.py <case_id> <model> [max_turns]

Reimplements `figsurgeon.agent.edit`'s loop body (same system prompt/encoding, imported
not copied) to add per-call instrumentation: seconds, verdict state, saved previews,
running token/cost totals. Log is `driver.py`'s schema plus 'model', 'usage', 'refused',
and 'truncated' (ran out of output tokens mid-turn, else looks like a considered stop).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'out')
LOGS = os.path.join(HERE, 'logs')
sys.path.insert(0, ROOT)

from PIL import Image                                                    # noqa: E402

from figsurgeon.agent import SYSTEM_PROMPT, _image_block       # noqa: E402
from figsurgeon.tools import TOOL_SCHEMAS                               # noqa: E402
from figsurgeon.workspace import ImageWorkspace                         # noqa: E402
from evals.agent_loop.cases import CASES                                 # noqa: E402

MAX_TURNS = 10


def slug(model):
    return model.replace('/', '_')


def run(case_id, model, client, max_turns=MAX_TURNS, verbose=True):
    case = {c['id']: c for c in CASES}[case_id]
    img = Image.open(os.path.join(ROOT, case['image']))
    img.load()
    ws = ImageWorkspace(img.convert('RGBA') if img.mode == 'RGBA' else img.convert('RGB'))

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    tag = f'{case_id}__{slug(model)}'
    started = time.time()
    steps = []
    usage_totals = {'prompt_tokens': 0, 'completion_tokens': 0, 'cost': 0.0}
    said_parts = []
    refused = None
    truncated = None

    messages = [{
        'role': 'user',
        'content': [
            _image_block(ws.image),
            {'type': 'text',
             'text': f'Image size: {ws.image.size[0]}x{ws.image.size[1]} pixels.\n\n'
                     f'Instruction: {case["request"]}'},
        ],
    }]

    turns_used = 0
    for turn in range(max_turns):
        turns_used = turn + 1
        resp = client.messages.create(
            model=model, max_tokens=16000,
            system=[{'type': 'text', 'text': SYSTEM_PROMPT,
                     'cache_control': {'type': 'ephemeral'}}],
            tools=TOOL_SCHEMAS, messages=messages)

        u = getattr(resp, 'usage', None) or {}
        usage_totals['prompt_tokens'] += u.get('prompt_tokens') or 0
        usage_totals['completion_tokens'] += u.get('completion_tokens') or 0
        usage_totals['cost'] += u.get('cost') or 0.0

        if getattr(resp, 'stop_reason', None) == 'refusal':
            why = getattr(resp, 'stop_details', None)
            refused = f'the model declined this instruction ' \
                      f'({getattr(why, "category", "no category given")})'
            if verbose:
                print(f'[{tag}] REFUSED: {refused}')
            break

        tool_uses = [b for b in resp.content if b.type == 'tool_use']
        for t in [b.text for b in resp.content if b.type == 'text']:
            said_parts.append(t)
            if verbose:
                print(f'[{tag}] [model] {t[:200]}')

        if not tool_uses:
            # Record explicitly: a max_tokens stop otherwise looks like a considered stop.
            if getattr(resp, 'stop_reason', None) == 'max_tokens':
                truncated = f'the model ran out of output tokens on turn {turn + 1}; it ' \
                            f'did not choose to stop here'
                if verbose:
                    print(f'[{tag}] TRUNCATED: {truncated}')
            break

        messages.append({'role': 'assistant', 'content': resp.content})
        results, images, changed = [], [], False
        for tu in tool_uses:
            t0 = time.time()
            result = ws.apply(tu.name, tu.input)
            elapsed = round(time.time() - t0, 1)
            verified = (getattr(result, 'data', None) or {}).get('verified', None)
            preview_path = None
            if result.preview is not None:
                preview_path = os.path.join(OUT, f'{tag}_step{len(steps) + 1}_{tu.name}.png')
                result.preview.save(preview_path)
            steps.append({'step': len(steps) + 1, 'tool': tu.name, 'args': tu.input,
                          'note': result.note, 'verified': verified, 'seconds': elapsed,
                          'preview': preview_path})
            if verbose:
                print(f'[{tag}] [tool] {tu.name}({json.dumps(tu.input)[:80]}) -> '
                      f'{result.note[:110]}')
            results.append({'type': 'tool_result', 'tool_use_id': tu.id,
                            'content': result.note})
            if result.preview is not None:
                images.append((f'Result of {tu.name}:', result.preview))
            changed = changed or result.mutates

        for caption, preview in images:
            results.append({'type': 'text', 'text': caption})
            results.append(_image_block(preview))
        if changed:
            results.append({'type': 'text', 'text': 'Current state of the image being edited:'})
            results.append(_image_block(ws.image))
        messages.append({'role': 'user', 'content': results})

    final_path = os.path.join(OUT, f'{tag}_final.png')
    ws.image.save(final_path)
    said = ' '.join(said_parts).strip() or (refused or '(no closing statement)')
    log = {'case': case_id, 'request': case['request'], 'image': case['image'],
           'model': model, 'driver_said': said, 'refused': refused,
           'truncated': truncated, 'turns_used': turns_used, 'max_turns': max_turns,
           'final_image': final_path, 'total_seconds': round(time.time() - started, 1),
           'usage': usage_totals, 'steps': steps}
    with open(os.path.join(LOGS, f'{tag}.json'), 'w') as fh:
        json.dump(log, fh, indent=2)
    print(f'[{tag}] log written: {len(steps)} calls, {log["total_seconds"]}s, '
          f'~${usage_totals["cost"]:.4f}, '
          f'{usage_totals["prompt_tokens"]}+{usage_totals["completion_tokens"]} tok')
    return log


if __name__ == '__main__':
    from figsurgeon import openrouter
    case_id = sys.argv[1]
    model = sys.argv[2]
    max_turns = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_TURNS
    run(case_id, model, openrouter.client(), max_turns=max_turns)
