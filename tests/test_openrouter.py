"""The OpenRouter adapter, driven without a key or a network call.

What needs testing is the translation, not OpenRouter: the loop speaks the Anthropic shape
and the endpoint speaks the OpenAI shape, and a silent mistranslation would look exactly
like a model behaving oddly. So `_post` is replaced with a scripted stub that also RECORDS
what it was sent, and the assertions are on both directions of the conversion.
"""
import base64
import json

import numpy as np
import pytest
from PIL import Image

from figsurgeon import openrouter
from figsurgeon.agent import edit


class Endpoint:
    """Replays scripted OpenAI-shaped replies and keeps every request body."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def __call__(self, body, timeout):
        self.bodies.append(body)
        if self.replies:
            return self.replies.pop(0)
        return _reply('done')


def _reply(text=None, calls=None, finish='stop'):
    msg = {'content': text}
    if calls:
        msg['tool_calls'] = [
            {'id': cid, 'type': 'function',
             'function': {'name': name, 'arguments': json.dumps(args)}}
            for cid, name, args in calls]
    return {'choices': [{'message': msg, 'finish_reason': finish}]}


@pytest.fixture
def endpoint(monkeypatch):
    def install(replies):
        ep = Endpoint(replies)
        monkeypatch.setattr(openrouter, '_post', ep)
        return ep
    return install


@pytest.fixture(scope='module')
def cat():
    from skimage import data
    return Image.fromarray(data.chelsea())


def test_a_tool_call_survives_the_round_trip(endpoint, cat):
    """An OpenAI `tool_calls` entry must arrive in the loop as a `tool_use` block, run the
    real tool, and go back out as a `role: tool` message."""
    ep = endpoint([_reply(calls=[('c1', 'stylise', {'effect': 'grayscale'})],
                          finish='tool_calls'),
                   _reply('done')])
    result, transcript = edit(cat, 'make it grey', client=openrouter.client(),
                              model='test/model')
    assert any(row[0] == 'stylise' for row in transcript), transcript
    arr = np.array(result.convert('RGB'))
    assert np.allclose(arr[:, :, 0], arr[:, :, 1], atol=2), 'the tool never actually ran'

    # second request replays the assistant turn and the tool result
    replay = ep.bodies[1]['messages']
    assistant = [m for m in replay if m['role'] == 'assistant']
    call = assistant[0]['tool_calls'][0]['function']
    assert assistant and call['name'] == 'stylise'
    assert json.loads(call['arguments']) == {'effect': 'grayscale'}, 'arguments were lost'
    assert [m for m in replay if m['role'] == 'tool'][0]['tool_call_id'] == 'c1'


def test_the_image_becomes_a_data_url(endpoint, cat):
    ep = endpoint([_reply('nothing to do')])
    edit(cat, 'look', client=openrouter.client(), model='test/model')
    parts = ep.bodies[0]['messages'][-1]['content']
    urls = [p for p in parts if p['type'] == 'image_url']
    assert urls, parts
    head = urls[0]['image_url']['url']
    # The adapter must carry whatever media type it was handed (PNG or JPEG), not assume one.
    assert head.startswith(('data:image/png;base64,', 'data:image/jpeg;base64,'))
    raw = base64.b64decode(head.split(',', 1)[1])
    assert Image.open(__import__('io').BytesIO(raw)).size == cat.size


def test_the_tool_schemas_are_translated(endpoint, cat):
    ep = endpoint([_reply('nothing to do')])
    edit(cat, 'look', client=openrouter.client(), model='test/model')
    tools = ep.bodies[0]['tools']
    assert tools and all(t['type'] == 'function' for t in tools)
    fn = tools[0]['function']
    assert set(fn) == {'name', 'description', 'parameters'}, fn
    assert 'input_schema' not in fn, 'the Anthropic key name leaked through'


def test_the_system_prompt_is_flattened_and_the_breakpoint_dropped(endpoint, cat):
    """OpenAI takes a plain string and would reject `cache_control`, so it must be dropped
    while the system prompt text still arrives."""
    ep = endpoint([_reply('nothing to do')])
    edit(cat, 'look', client=openrouter.client(), model='test/model')
    system = [m for m in ep.bodies[0]['messages'] if m['role'] == 'system']
    assert len(system) == 1
    assert isinstance(system[0]['content'], str) and len(system[0]['content']) > 50
    assert 'cache_control' not in json.dumps(ep.bodies[0])


def test_a_content_filter_reaches_the_loops_refusal_branch(endpoint, cat):
    """`content_filter` must map to 'refusal', not look like a model that just stopped
    calling tools."""
    ep = endpoint([_reply(None, finish='content_filter')])
    result, transcript = edit(cat, 'something declined', client=openrouter.client(),
                              model='test/model', max_turns=4)
    assert any(row[0] == 'refused' for row in transcript), transcript
    assert len(ep.bodies) == 1, 'the loop kept asking after a refusal'
    assert np.array_equal(np.array(result), np.array(cat))


def test_unparseable_tool_arguments_are_named(endpoint, cat):
    """Non-JSON tool arguments must fail here naming the tool, not surface as an unrelated
    error deeper in the operation."""
    endpoint([{'choices': [{'message': {'content': None, 'tool_calls': [
        {'id': 'c1', 'type': 'function',
         'function': {'name': 'stylise', 'arguments': '{not json'}}]},
        'finish_reason': 'tool_calls'}]}])
    with pytest.raises(RuntimeError, match='stylise'):
        edit(cat, 'make it grey', client=openrouter.client(), model='test/model')


def test_an_http_error_does_not_leak_the_key(endpoint, cat, monkeypatch):
    """The failure path stringifies a response body; the Authorization header must not be
    anywhere in what it raises."""
    import urllib.error

    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-secret-value')

    def boom(req, timeout=None):
        raise urllib.error.HTTPError('u', 402, 'Payment Required', {},
                                     __import__('io').BytesIO(b'{"error":"no credit"}'))

    monkeypatch.setattr(openrouter.urllib.request, 'urlopen', boom)
    with pytest.raises(RuntimeError) as e:
        openrouter.client().messages.create(model='m', messages=[
            {'role': 'user', 'content': 'hi'}])
    assert 'no credit' in str(e.value)
    assert 'sk-or-secret-value' not in str(e.value)


def test_the_loops_default_model_resolves_on_openrouter(endpoint, cat):
    """A default model id already prefixed with its vendor must reach OpenRouter untouched;
    a bare id must still get a vendor prefix, or OpenRouter rejects it as unknown."""
    ep = endpoint([_reply('nothing to do')])
    edit(cat, 'look', client=openrouter.client())          # no model= -- the loop's default
    assert ep.bodies[0]['model'] == 'z-ai/glm-5.3-flash', 'the default was rewritten'

    ep = endpoint([_reply('nothing to do')])
    edit(cat, 'look', client=openrouter.client(), model='claude-sonnet-5')
    assert ep.bodies[0]['model'] == 'anthropic/claude-sonnet-5'

    ep = endpoint([_reply('nothing to do')])
    edit(cat, 'look', client=openrouter.client(), model='openai/gpt-5.4')
    assert ep.bodies[0]['model'] == 'openai/gpt-5.4', 'a vendor-prefixed id was rewritten'


def test_a_transient_error_is_retried(monkeypatch):
    """A 503 or a dropped connection must be retried; other errors must not."""
    import io
    import urllib.error
    calls = []

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError('u', 503, 'busy', {}, io.BytesIO(b'busy'))
        if len(calls) == 2:
            raise urllib.error.URLError('connection reset')
        return io.BytesIO(json.dumps(_reply('ok')).encode())

    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr(openrouter.urllib.request, 'urlopen', flaky)
    monkeypatch.setattr(openrouter.time, 'sleep', lambda s: None)
    assert openrouter._post({'x': 1}, timeout=5)['choices'][0]['message']['content'] == 'ok'
    assert len(calls) == 3


def test_a_client_error_is_not_retried_and_a_network_failure_is_a_runtime_error(monkeypatch):
    import io
    import urllib.error
    calls = []

    def bad(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError('u', 400, 'bad', {}, io.BytesIO(b'bad request'))

    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr(openrouter.urllib.request, 'urlopen', bad)
    monkeypatch.setattr(openrouter.time, 'sleep', lambda s: None)
    with pytest.raises(RuntimeError, match='400'):
        openrouter._post({}, timeout=5)
    assert len(calls) == 1

    def down(req, timeout):
        raise urllib.error.URLError('name resolution failed')
    monkeypatch.setattr(openrouter.urllib.request, 'urlopen', down)
    with pytest.raises(RuntimeError, match='could not reach OpenRouter'):
        openrouter._post({}, timeout=5)


def test_a_missing_key_is_a_runtime_error_not_a_system_exit(monkeypatch, tmp_path):
    """SystemExit from library code would end the host process instead of reaching the
    caller's error handling."""
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.setattr(openrouter, 'ROOT', str(tmp_path))
    with pytest.raises(RuntimeError, match='OPENROUTER_API_KEY'):
        openrouter._key()
