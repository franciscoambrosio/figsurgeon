"""Drive `agent.edit` through OpenRouter instead of the Anthropic SDK.

    from figsurgeon.agent import edit
    from figsurgeon import openrouter
    result, transcript = edit(image, 'make the red car blue',
                              client=openrouter.client(),
                              model='anthropic/claude-sonnet-4.5')

`agent.edit` already accepts any object exposing `messages.create(...)` -- that contract
exists so the loop can be tested offline, and it is what makes this file possible without
touching the loop at all. What is here is purely translation: OpenRouter speaks the
OpenAI shape (`tool_calls`, `image_url`, a flat `role: tool` message) and the loop speaks
the Anthropic shape (`tool_use` content blocks, `image` blocks with a `source`), so this
converts each way and nothing else.

The key comes from OPENROUTER_API_KEY or a gitignored `.env.local`, the same precedence
`evals/openrouter_edit.py` uses. It is never printed or logged.

What this does not give you: prompt caching is dropped. `cache_control` is an Anthropic
field and the loop's one breakpoint has no OpenAI-shaped equivalent, so a ten-turn edit
here pays for the system prompt and tool schemas on every turn. Going through OpenRouter to
reach an Anthropic model therefore costs strictly more than the SDK path. Use this for a
model you cannot otherwise reach.
"""
import json
import os
import time
import urllib.error
import urllib.request

ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# OpenAI's finish_reason -> the stop_reason `agent.edit` reads. `content_filter` maps to
# 'refusal' so a declined request takes the loop's refusal branch rather than looking like
# a model that simply stopped asking for tools.
_STOP = {'tool_calls': 'tool_use', 'stop': 'end_turn', 'length': 'max_tokens',
         'content_filter': 'refusal', 'function_call': 'tool_use'}


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
    raise RuntimeError('no OPENROUTER_API_KEY in the environment or .env.local')


class Block:
    """One content block, with the attribute names `agent.edit` reads off the SDK's blocks.

    The loop puts these straight back into `messages` on the next turn, so they have to
    survive the round trip -- `_messages` below recognises them on the way back in.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f'Block({self.__dict__})'


def _system(system):
    """The loop sends a list of text blocks carrying a cache breakpoint; OpenAI wants one
    string. The breakpoint is dropped -- see the module docstring."""
    if isinstance(system, str):
        return system
    return '\n\n'.join(b['text'] for b in system if b.get('type') == 'text')


def _tools(tools):
    return [{'type': 'function',
             'function': {'name': t['name'], 'description': t.get('description', ''),
                          'parameters': t['input_schema']}}
            for t in tools]


def _part(block):
    """One non-tool_result block of a user message, in OpenAI's shape."""
    if block['type'] == 'text':
        return {'type': 'text', 'text': block['text']}
    src = block['source']
    return {'type': 'image_url',
            'image_url': {'url': f"data:{src['media_type']};base64,{src['data']}"}}


def _assistant(blocks):
    """An assistant turn the loop is replaying: our own Block objects from last time."""
    text = ''.join(b.text for b in blocks if b.type == 'text')
    calls = [{'id': b.id, 'type': 'function',
              'function': {'name': b.name, 'arguments': json.dumps(b.input)}}
             for b in blocks if b.type == 'tool_use']
    msg = {'role': 'assistant', 'content': text or None}
    if calls:
        msg['tool_calls'] = calls
    return msg


def _messages(messages):
    """Anthropic-shaped `messages` -> OpenAI-shaped.

    The one structural difference that is not a rename: a single user message here can
    carry BOTH tool results and images (the loop sends the verdicts and the new working
    image together), while OpenAI wants each tool result as its own `role: tool` message
    and allows no image in one. So a mixed message becomes several: the tool results
    first, then whatever text and images were alongside them.
    """
    out = []
    for m in messages:
        content = m['content']
        if m['role'] == 'assistant':
            out.append(_assistant(content))
            continue
        if isinstance(content, str):
            out.append({'role': 'user', 'content': content})
            continue
        results = [b for b in content if b.get('type') == 'tool_result']
        rest = [b for b in content if b.get('type') != 'tool_result']
        for r in results:
            out.append({'role': 'tool', 'tool_call_id': r['tool_use_id'],
                        'content': r['content']})
        if rest:
            out.append({'role': 'user', 'content': [_part(b) for b in rest]})
    return out


# Statuses where the request was not served and trying again is the remedy. A timeout is
# deliberately NOT retried: the request may have run -- an image generation is billed --
# and sending it again could pay twice.
_RETRY_STATUS = {429, 502, 503, 504}
_RETRY_DELAYS = (2, 8, 30)


def _post(body, timeout):
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + _key(),
                                          'Content-Type': 'application/json'})
    for attempt, delay in enumerate(_RETRY_DELAYS + (None,)):
        try:
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY_STATUS or delay is None:
                raise RuntimeError(f'OpenRouter returned {e.code}: '
                                   f'{e.read().decode()[:400]}') from None
        except urllib.error.URLError as e:
            # A refused or reset connection never reached the model. (A timeout inside
            # urlopen arrives as URLError wrapping TimeoutError, and is not retried.)
            if isinstance(e.reason, TimeoutError) or delay is None:
                raise RuntimeError(f'could not reach OpenRouter: {e.reason}') from None
        except TimeoutError:
            raise RuntimeError(f'OpenRouter did not answer within {timeout} s') from None
        time.sleep(delay)


def _response(payload):
    """OpenAI-shaped reply -> an object shaped like the SDK's, for the loop to read."""
    choice = payload['choices'][0]
    msg = choice['message']
    blocks = []
    if msg.get('content'):
        blocks.append(Block(type='text', text=msg['content']))
    for call in msg.get('tool_calls') or []:
        fn = call['function']
        # A model can return an empty or malformed argument string; the loop would fail
        # deep inside a tool with no idea why, so it is named here instead.
        try:
            args = json.loads(fn['arguments'] or '{}')
        except json.JSONDecodeError:
            raise RuntimeError(f"{fn['name']} was called with arguments that are not JSON: "
                               f"{fn['arguments']!r}") from None
        blocks.append(Block(type='tool_use', id=call['id'], name=fn['name'], input=args))
    return Block(content=blocks,
                 stop_reason=_STOP.get(choice.get('finish_reason'), 'end_turn'),
                 stop_details=None, usage=payload.get('usage'))


class _Messages:
    def __init__(self, timeout):
        self.timeout = timeout

    def create(self, model, messages, max_tokens=16000, system=None, tools=None, **_):
        # OpenRouter needs a vendor prefix, and the bare ids the Anthropic SDK takes
        # ('claude-sonnet-5') have none. Without this, passing one of those here is a 400
        # from OpenRouter about an unknown model, which reads as a broken adapter rather
        # than a naming difference. An id that already names its vendor
        # ('z-ai/glm-5.3-flash', 'openai/gpt-5.4') is left alone -- which includes
        # `agent.edit`'s own default.
        if '/' not in model:
            model = 'anthropic/' + model
        body = {'model': model, 'max_tokens': max_tokens,
                'messages': ([{'role': 'system', 'content': _system(system)}] if system
                             else []) + _messages(messages)}
        if tools:
            body['tools'] = _tools(tools)
        return _response(_post(body, self.timeout))


class Client:
    """Exposes just `client.messages.create(...)` -- all `agent.edit` asks for."""

    def __init__(self, timeout=600):
        self.messages = _Messages(timeout)


def client(timeout=600):
    return Client(timeout=timeout)
