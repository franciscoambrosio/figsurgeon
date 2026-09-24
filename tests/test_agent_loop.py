"""The agent loop, driven by a stub model instead of the API: what needs testing is the
wiring around the model, not the model itself.
"""
import os

import numpy as np
import pytest
from PIL import Image

from figsurgeon.agent import edit
from figsurgeon.workspace import ImageWorkspace

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'evals', '_corpus')


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class StubClient:
    """Replays a scripted sequence of model turns and records what it was sent, mimicking
    only the `client.messages.create(...)` surface `agent.edit` touches."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        blocks = self.script.pop(0) if self.script else [_Block(type='text', text='done')]
        return _Block(content=blocks)


def _tool_use(name, args, uid='t1'):
    return _Block(type='tool_use', name=name, input=args, id=uid)


@pytest.fixture(scope='module')
def cat():
    from skimage import data
    return Image.fromarray(data.chelsea())


def _images_sent(client):
    """Every image block the loop sent to the model, across all requests."""
    out = []
    for req in client.requests:
        for msg in req['messages']:
            content = msg['content']
            if isinstance(content, list):
                out += [b for b in content
                        if isinstance(b, dict) and b.get('type') == 'image']
    return out


def test_a_looking_tool_returns_a_preview_and_leaves_the_image_alone(cat):
    """`show_grid`'s answer is its preview, sent back rather than pasted onto the image."""
    client = StubClient([
        [_Block(type='text', text='looking first'), _tool_use('show_grid', {})],
        [_Block(type='text', text='the eye is around x=140..210')],
    ])
    result, transcript = edit(cat, 'where is the eye?', client=client)

    assert np.array_equal(np.array(result), np.array(cat)), 'looking changed the image'
    assert any(name == 'show_grid' for name, _, _ in transcript)
    assert len(_images_sent(client)) > 1, 'the preview was never sent to the model'


def test_an_edit_sends_back_the_new_working_image(cat):
    client = StubClient([
        [_tool_use('adjust', {'brightness': 1.4})],
        [_Block(type='text', text='brightened')],
    ])
    result, transcript = edit(cat, 'brighten it', client=client)
    assert np.array(result).mean() > np.array(cat).mean()
    note = [n for name, _, n in transcript if name == 'adjust'][0]
    assert 'verified' in note, 'the verification verdict never reached the model'


def test_the_model_can_undo_a_bad_edit(cat):
    """Undo, or the only response to a bad edit is another edit on top of it."""
    client = StubClient([
        [_tool_use('adjust', {'brightness': 3.0})],
        [_Block(type='text', text='too much'), _tool_use('undo', {}, uid='t2')],
        [_Block(type='text', text='reverted')],
    ])
    result, transcript = edit(cat, 'brighten it a lot', client=client)
    assert np.array_equal(np.array(result), np.array(cat)), 'undo did not reach the workspace'
    assert any(name == 'undo' for name, _, _ in transcript)


def test_a_failed_check_is_reported_back_to_the_model():
    from skimage import data
    grey = Image.fromarray(np.stack([data.camera()] * 3, axis=-1))
    client = StubClient([
        [_tool_use('adjust', {'saturation': 2.0})],
        [_Block(type='text', text='that image has no colour')],
    ])
    _, transcript = edit(grey, 'make the colours pop', client=client)
    note = [n for name, _, n in transcript if name == 'adjust'][0]
    assert 'CHECK FAILED' in note
    assert 'undo' in note, 'the note should tell the model what to do about it'


def test_a_bad_argument_does_not_kill_the_session(cat):
    """The model must be able to read the problem and correct itself on the next turn."""
    client = StubClient([
        [_tool_use('crop', {'box': [900, 900, 1000, 1000]})],
        [_tool_use('crop', {'box': [0, 0, 100, 100]}, uid='t2')],
        [_Block(type='text', text='cropped')],
    ])
    result, transcript = edit(cat, 'crop it', client=client)
    assert result.size == (100, 100)
    assert transcript[0][2].startswith('error')


def test_the_loop_stops_when_the_model_stops_calling_tools(cat):
    client = StubClient([[_Block(type='text', text='nothing to do')]])
    result, transcript = edit(cat, 'do nothing', client=client, max_turns=8)
    assert len(client.requests) == 1, 'the loop kept going after the model finished'
    assert np.array_equal(np.array(result), np.array(cat))


def test_max_turns_bounds_a_model_that_never_stops(cat):
    """A model looping forever must cost a bounded number of calls."""
    class Endless(StubClient):
        def create(self, **kwargs):
            self.requests.append(kwargs)
            return _Block(content=[_tool_use('show_grid', {})])

    client = Endless([])
    edit(cat, 'look forever', client=client, max_turns=3)
    assert len(client.requests) == 3


def test_a_workspace_can_be_carried_across_calls(cat):
    """A later `undo` can revert an edit made in an earlier call sharing the same workspace."""
    ws = ImageWorkspace(cat)
    edit(cat, 'brighten', client=StubClient([
        [_tool_use('adjust', {'brightness': 1.5})], [_Block(type='text', text='ok')]]),
        workspace=ws)
    assert len(ws.history) == 1
    edit(ws.image, 'undo that', client=StubClient([
        [_tool_use('undo', {})], [_Block(type='text', text='ok')]]), workspace=ws)
    assert np.array_equal(np.array(ws.image), np.array(cat))


def test_oversized_images_are_downscaled_before_sending(cat):
    """Never below what the model is told the size is, or its reported boxes are corrupted."""
    from figsurgeon.agent import _encode
    import base64
    import io
    big = cat.resize((3000, 2000))
    decoded = Image.open(io.BytesIO(base64.b64decode(_encode(big)[0])))
    assert max(decoded.size) == 1568
    small = Image.open(io.BytesIO(base64.b64decode(_encode(cat)[0])))
    assert small.size == cat.size, 'a normal image must be sent at its true size'


def test_the_cheaper_encoding_wins_and_flat_images_stay_lossless():
    """Both encodings are tried and the smaller wins, but only by a margin, so a chart is
    not silently traded for JPEG ringing on the 1 px lines and text this package edits."""
    from figsurgeon.agent import _encode
    import base64

    photo = Image.open(os.path.join(CORPUS, 'car_red.jpg')).convert('RGB')
    photo.thumbnail((1200, 1200), Image.LANCZOS)
    data, media = _encode(photo)
    assert media == 'image/jpeg', 'a photograph should not be sent as PNG'
    assert len(base64.b64decode(data)) < 0.5e6, 'the whole point is that it is smaller'

    flat = Image.new('RGB', (900, 700), 'white')
    for x in range(0, 900, 7):
        for y in range(700):
            flat.putpixel((x, y), (20, 20, 20))
    _data, media = _encode(flat)
    assert media == 'image/png', 'a flat-colour image must stay lossless'


def test_a_refusal_stops_the_loop_and_says_so(cat):
    """A refusal comes back HTTP 200 with no tool calls, indistinguishable from the model
    finishing normally unless the difference reaches the caller."""
    class Refusing(StubClient):
        def create(self, **kwargs):
            self.requests.append(kwargs)
            return _Block(content=[], stop_reason='refusal',
                          stop_details=_Block(type='refusal', category='cyber'))

    client = Refusing([])
    result, transcript = edit(cat, 'do something declined', client=client, max_turns=5)
    assert len(client.requests) == 1, 'the loop kept asking after a refusal'
    assert any(row[0] == 'refused' for row in transcript), transcript
    assert 'cyber' in [row[2] for row in transcript if row[0] == 'refused'][0]
    assert np.array_equal(np.array(result), np.array(cat))


def test_a_truncated_turn_is_not_recorded_as_a_considered_stop(cat):
    """A truncated turn returns HTTP 200 with nothing in it, the same shape as the model
    finishing, and must not be recorded as a deliberate stop."""
    class Truncating(StubClient):
        def create(self, **kwargs):
            self.requests.append(kwargs)
            return _Block(content=[], stop_reason='max_tokens')

    client = Truncating([])
    result, transcript = edit(cat, 'do something long', client=client, max_turns=5)
    assert len(client.requests) == 1, 'the loop kept asking after a truncation'
    assert any(row[0] == 'truncated' for row in transcript), transcript
    assert np.array_equal(np.array(result), np.array(cat))


def test_a_stub_without_stop_reason_still_drives_the_loop(cat):
    """The loop's contract is any object exposing `messages.create`, so `stop_reason` must
    be read tolerantly rather than assumed present."""
    client = StubClient([[_tool_use('show_grid', {})],
                         [_Block(type='text', text='done')]])
    edit(cat, 'look once', client=client, max_turns=5)
    assert len(client.requests) == 2, 'the loop stopped early on a stub with no stop_reason'


def test_the_stable_prefix_is_cached(cat):
    """`SYSTEM_PROMPT` and the tool schemas are resent every turn and need a cache breakpoint."""
    client = StubClient([[_tool_use('show_grid', {})],
                         [_Block(type='text', text='done')]])
    edit(cat, 'look once', client=client, max_turns=5)
    system = client.requests[0]['system']
    assert isinstance(system, list), 'a bare string cannot carry a cache breakpoint'
    assert system[-1].get('cache_control') == {'type': 'ephemeral'}, system[-1]
    assert client.requests[0]['max_tokens'] >= 16000, 'a low cap truncates a multi-tool turn'


def test_the_default_model_matches_the_default_client(monkeypatch, cat):
    """With no `client`, the loop builds an Anthropic SDK client and must send it a model
    the Anthropic API serves, not an OpenRouter id."""
    import sys
    import types
    sent = {}

    class FakeSDK:
        def __init__(self, api_key=None):
            self.messages = self

        def create(self, **kwargs):
            sent.update(kwargs)
            return _Block(content=[_Block(type='text', text='done')])

    monkeypatch.setitem(sys.modules, 'anthropic', types.SimpleNamespace(Anthropic=FakeSDK))
    edit(cat, 'do nothing', api_key='x')
    assert sent['model'].startswith('claude-'), sent['model']


def test_running_out_of_turns_is_recorded(cat):
    """Hitting max_turns must be distinguishable from the model choosing to stop."""
    class Endless(StubClient):
        def create(self, **kwargs):
            self.requests.append(kwargs)
            return _Block(content=[_tool_use('show_grid', {})])

    _, transcript = edit(cat, 'look forever', client=Endless([]), max_turns=2)
    assert transcript[-1][0] == 'max_turns', transcript[-1]
