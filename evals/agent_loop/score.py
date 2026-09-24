"""Did the verdicts change what the driver did, and did the pictures come out right?

    .venv/bin/python evals/agent_loop/score.py

Reads every transcript in `logs/` (original single-file-per-case logs, plus the
multi-model `<case>__<model_slug>.json` logs from `run_models.py`) and reports: verdict
counts and whether a non-pass verdict changed what the driver did next; the correct/
honest-stop/incomplete/wrong rate per model and pooled, judged by eye and stored in
`labels.py` (never inferred from the log or the driver's own words); and how each run
ended (turn budget spent, output truncated, or stopped on its own).
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOGS = os.path.join(HERE, 'logs')
sys.path.insert(0, ROOT)

from evals.agent_loop.cases import CASES                                 # noqa: E402
from evals.agent_loop.kinds import KIND                                  # noqa: E402

try:
    from evals.agent_loop.labels import LABEL
except ImportError:                       # before anyone has looked at the pictures
    LABEL = {}

CASE_IDS = {c['id'] for c in CASES}
EDITS = None          # every tool that mutates; filled from tools.py on first use


def is_edit(step):
    global EDITS
    if EDITS is None:
        from figsurgeon.workspace import LOOKING
        EDITS = LOOKING
    return step['tool'] not in EDITS and step['tool'] not in ('undo', 'reset')


def changed_course(step, nxt):
    """Narrow on purpose: undo, a different tool, or the same tool argued differently."""
    if nxt is None:
        return 'stopped'
    if nxt['tool'] in ('undo', 'reset'):
        return 'undid'
    if nxt['tool'] != step['tool']:
        return 'changed tool'
    if nxt['args'] != step['args']:
        return 'retried differently'
    return 'repeated it'


def label_for(case, model):
    """(case, model) label if one was written, else the bare-case-id label (the original
    eight, one picture each, no model recorded), else 'unjudged'. Always returns a 3-tuple
    (verdict, why, self-report-match) -- older 2-tuple entries pad with '' for the third."""
    entry = LABEL.get((case, model)) or LABEL.get(case) or ('unjudged', '')
    return tuple(entry) + ('',) * (3 - len(entry))


def ending(run):
    """How the loop stopped, from the log alone -- never inferred from the pictures."""
    if run.get('refused'):
        return f'refused: {run["refused"]}'
    if run.get('truncated'):
        return f'TRUNCATED -- {run["truncated"]}'
    if 'turns_used' not in run:
        return 'not recorded'
    if run['turns_used'] >= run.get('max_turns', 0):
        return f'turn budget spent ({run["turns_used"]}/{run["max_turns"]} turns)'
    return f'stopped of its own accord after {run["turns_used"]} turns'


def load():
    """Only `<case>.json` (the original run) and `<case>__<model_slug>.json`
    (`run_models.py`) -- not sidecar files sharing a case id, which would otherwise be
    double-counted as a second run."""
    out = []
    for path in sorted(glob.glob(os.path.join(LOGS, '*.json'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        case = stem.split('__', 1)[0]
        if case not in CASE_IDS or not (stem == case or stem.startswith(case + '__')):
            continue
        with open(path) as fh:
            run = json.load(fh)
        if run.get('case') != case:
            continue
        run.setdefault('model', 'original-run')
        out.append(run)
    return out


def main():
    runs = load()
    if not runs:
        print(f'no transcripts in {LOGS} yet')
        return
    print(f'{"case":16s} {"model":26s} {"kind":10s} {"calls":>5s} {"edits":>5s} {"secs":>6s}  '
          f'{"verdicts (v/f/?)":>16s}  {"picture":11s} {"after a bad verdict"}')
    totals = {}   # model -> dict of counters, plus 'ALL'
    for run in runs:
        model = run['model']
        for key in (model, 'ALL'):
            totals.setdefault(key, {'verified': 0, 'failed': 0, 'inconclusive': 0,
                                    'acted': 0, 'ignored': 0})
        steps = run['steps']
        edits = [s for s in steps if is_edit(s)]
        v = sum(1 for s in edits if s['verified'] is True)
        f = sum(1 for s in edits if s['verified'] is False)
        q = sum(1 for s in edits if s['verified'] is None)
        reactions = []
        for s in edits:
            if s['verified'] is True:
                continue
            nxt = next((n for n in steps if n['step'] > s['step']), None)
            reaction = changed_course(s, nxt)
            reactions.append(f'{s["tool"]}->{reaction}')
            bucket = None
            if reaction in ('undid', 'changed tool', 'retried differently'):
                bucket = 'acted'
            elif reaction == 'repeated it' or (reaction == 'stopped' and s['verified'] is False):
                bucket = 'ignored'
            if bucket:
                for key in (model, 'ALL'):
                    totals[key][bucket] += 1
        for key in (model, 'ALL'):
            totals[key]['verified'] += v
            totals[key]['failed'] += f
            totals[key]['inconclusive'] += q
        kind = KIND.get(run['case'], ('?', ''))[0]
        label = label_for(run['case'], model)[0]
        print(f'{run["case"]:16s} {model:26s} {kind:10s} {len(steps):5d} {len(edits):5d} '
              f'{run["total_seconds"]:6.0f}  {f"{v}/{f}/{q}":>16s}  {label:11s} '
              f'{", ".join(reactions) if reactions else "-"}')

    print()
    for model, t in totals.items():
        tag = 'POOLED (all models)' if model == 'ALL' else model
        print(f'{tag}: edits {t["verified"]} verified, {t["failed"]} failed, '
              f'{t["inconclusive"]} inconclusive -- of the non-pass verdicts, '
              f'{t["acted"]} changed what happened next, {t["ignored"]} did not')

    if not LABEL:
        print('\nno labels yet: look at every final image in out/ and write labels.py')
        return

    print('\nTHE RATE -- correct / honest-stop / incomplete / wrong, judged by eye, per model:')
    by_model = {}
    for run in runs:
        by_model.setdefault(run['model'], []).append(run['case'])

    def rate_line(tag, labs):
        n = len(labs)
        correct = labs.count('correct')
        stop = labs.count('honest stop')
        incomplete = labs.count('incomplete')
        wrong = labs.count('wrong')
        unjudged = n - correct - stop - incomplete - wrong
        extra = f', {unjudged} unjudged' if unjudged else ''
        print(f'  {tag:26s} {correct}/{n} correct, {stop}/{n} honest stop, '
              f'{incomplete}/{n} incomplete, {wrong}/{n} wrong{extra}')

    for model, cases in sorted(by_model.items(), key=lambda kv: (kv[0] == 'ALL', kv[0])):
        rate_line(model, [label_for(c, model)[0] for c in cases])
    rate_line('POOLED', [label_for(r['case'], r['model'])[0] for r in runs])

    endings = [(r, ending(r)) for r in runs]
    older = sum(1 for _, e in endings if e == 'not recorded')
    print('\nhow each run ENDED:')
    for run, how in endings:
        if how != 'not recorded':
            print(f'  {run["case"]:16s} {run["model"]:26s} {how}')
    if older:
        print(f'  ({older} of {len(runs)} logs predate run_models.py recording this and are '
              f'not guessed at)')

    mismatches = [(r['case'], r['model'], label_for(r['case'], r['model'])[2]) for r in runs
                  if label_for(r['case'], r['model'])[2]
                  and not label_for(r['case'], r['model'])[2].lower().startswith('matches')]
    if mismatches:
        print(f'\nself-report vs picture: {len(mismatches)}/{len(runs)} driver accounts '
              f'oversold, undersold or misled by omission relative to what the picture shows:')
        for case, model, note in mismatches:
            print(f'  {case:16s} {model:26s} {note}')

    kinds = {}
    for run in runs:
        k = KIND.get(run['case'], ('?',))[0]
        lab = label_for(run['case'], run['model'])[0]
        kinds.setdefault(k, []).append(lab)
    print('\npictures, by what the case was chosen to test (pooled across models):')
    for k, labs in sorted(kinds.items()):
        counts = {lab: labs.count(lab) for lab in sorted(set(labs))}
        print(f'  {k:10s} ' + ', '.join(f'{n} {lab}' for lab, n in counts.items()))

    print('\nwhat each driver said, against what the picture shows:')
    for run in runs:
        lab, why, match = label_for(run['case'], run['model'])
        print(f'  {run["case"]:16s} {run["model"]:26s} [{lab}] said: {run["driver_said"]}')
        if why:
            print(f'  {"":16s} {"":26s}  seen: {why}')
        if match:
            print(f'  {"":16s} {"":26s}  self-report: {match}')


if __name__ == '__main__':
    main()
