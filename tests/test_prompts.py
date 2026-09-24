"""Prompt coverage battery with expected outcomes.

Asserting the expected function matters more than asserting "it parsed": a prompt can parse
successfully and still do the opposite of what was asked (e.g. "black and white but leave the
red" routed to full grayscale). Those cases are marked below.
"""
from figsurgeon import grounding
from figsurgeon.photo_describe import parse

# A subject-naming isolation request ("the flowers") resolves with grounding installed,
# and refuses with an install hint without it -- pinning one outcome would break the other.
SUBJECT = 'isolate_subject' if grounding.available() else None

# (prompt, expected function name or None meaning "should raise with a helpful message")
CASES = [
    # --- selective colour isolation ---
    ('colour pop the red', 'isolate_colour'),
    ('color pop the blue', 'isolate_colour'),
    ('make everything black and white except the red', 'isolate_colour'),
    ('keep only the green', 'isolate_colour'),
    ('isolate the orange', 'isolate_colour'),
    ('grey out everything except the blue', 'isolate_colour'),
    ('desaturate everything but the red', 'isolate_colour'),      # not global adjust()
    ('make it black and white but leave the red', 'isolate_colour'),  # not global grayscale()
    ('keep the red in colour', 'isolate_colour'),
    # The loose "keep <word>" rule stays permissive: this is not an isolation request at
    # all, and must still reach plain grayscale rather than the subject-noun refusal.
    ('keep it black and white', 'grayscale'),
    # --- background ---
    ('remove the background', 'remove_background'),
    ('cut out the subject', 'remove_background'),
    ('make the background transparent', 'remove_background'),
    ('put this on a white background', 'replace_background'),
    ('replace the background with white', 'replace_background'),
    ('change the background to black', 'replace_background'),
    ('blur the background', 'blur_background'),
    # --- cleanup / correction ---
    ('remove the noise', 'denoise'),
    ('clean up the grain', 'denoise'),
    ('fix the white balance', 'auto_white_balance'),
    ('correct the colour cast', 'auto_white_balance'),
    ('it looks too orange', 'auto_white_balance'),
    ('sharpen it', 'adjust'),
    # --- stylise ---
    ('sepia tone', 'sepia'),
    ('make it look vintage', 'sepia'),
    ('turn it into a pencil sketch', 'sketch'),
    ('make it black and white', 'grayscale'),
    ('add a vignette', 'vignette'),
    ('add a subtle vignette', 'vignette'),
    ('add a strong vignette', 'vignette'),
    # --- tone ---
    ('brighten it', 'adjust'),
    ('make it darker', 'adjust'),
    ('increase the contrast', 'adjust'),
    ('reduce the contrast', 'adjust'),
    ('make the colours more vivid', 'adjust'),
    ('desaturate it a bit', 'adjust'),
    ('make it pop', 'adjust'),
    # --- geometry ---
    ('rotate 90 degrees', 'crop_and_rotate'),
    ('flip it horizontally', 'crop_and_rotate'),
    ('mirror it', 'crop_and_rotate'),
    # --- colour replacement ---
    ('replace the red with orange', 'replace_colour'),
    ('change red to blue', 'replace_colour'),
    ('turn the red into orange', 'replace_colour'),
    # --- should raise, with an actionable message rather than a wrong guess ---
    # Naming a subject instead of a colour must not fall through to global grayscale.
    ('make it black and white except the flowers', SUBJECT),
    ('desaturate everything but her dress', SUBJECT),
    ('colour pop the sunset', SUBJECT),
    ('keep the roses in colour', SUBJECT),
    ('I want the red replaced by the orange of her shirt', None),
    ('remove the object in the middle', None),
    ('crop the top half', None),
]


def run():
    ok = wrong = raised = 0
    problems = []
    for prompt, expected in CASES:
        try:
            fn, kwargs = parse(prompt)
            got = fn.__name__
            if expected is None:
                problems.append(f'SHOULD HAVE RAISED: {prompt!r} -> {got}')
                wrong += 1
            elif got != expected:
                problems.append(f'WRONG FN: {prompt!r} -> {got}, expected {expected}')
                wrong += 1
            else:
                ok += 1
        except ValueError:
            if expected is None:
                ok += 1          # raised as intended, with a helpful message
            else:
                problems.append(f'FAILED TO PARSE: {prompt!r} (expected {expected})')
                raised += 1
    total = len(CASES)
    print(f'{ok}/{total} correct   |   {wrong} wrong function   |   {raised} failed to parse')
    for p in problems:
        print('   ', p)
    return ok, total


if __name__ == '__main__':
    run()


def test_all_prompts_route_correctly():
    ok, total = run()
    assert ok == total, f'{total - ok} of {total} prompts routed to the wrong function'
