"""The MCP server exposes exactly `tools.TOOL_SCHEMAS` (plus open_image/save_image), backed
by a real `ImageWorkspace` -- so undo and verification work, and a preview tool hands back an
actual image instead of mutating the working image."""
import base64

import pytest

mcp = pytest.importorskip('mcp', reason='mcp extra not installed')

from PIL import Image

from figsurgeon import mcp_server
from figsurgeon.tools import TOOL_SCHEMAS


@pytest.fixture
def image_path(tmp_path):
    img = Image.new('RGB', (64, 64), (200, 40, 40))
    path = str(tmp_path / 'source.png')
    img.save(path)
    return path


@pytest.fixture(autouse=True)
def reset_state():
    mcp_server._state['workspace'] = None
    mcp_server._state['path'] = None
    yield
    mcp_server._state['workspace'] = None
    mcp_server._state['path'] = None


def test_all_schemas_present_by_name():
    served = {s['name'] for s in mcp_server.ALL_SCHEMAS}
    for s in TOOL_SCHEMAS:
        assert s['name'] in served
    assert 'open_image' in served
    assert 'save_image' in served


def test_schema_input_matches_tools_py():
    by_name = {s['name']: s for s in mcp_server.ALL_SCHEMAS}
    for s in TOOL_SCHEMAS:
        assert by_name[s['name']]['input_schema'] == s['input_schema']
        assert by_name[s['name']]['description'] == s['description']


def test_open_then_edit_carries_the_verdict(image_path):
    r = mcp_server._open_image({'path': image_path})
    assert 'opened' in r[0].text

    r = mcp_server._run_tool('adjust', {'saturation': 0.0})
    assert 'saturation' in r[0].text
    assert 'verified' in r[0].text or 'CHECK FAILED' in r[0].text or 'inconclusive' in r[0].text


def test_looking_tool_returns_image_content_and_does_not_mutate(image_path):
    mcp_server._open_image({'path': image_path})
    before = mcp_server._state['workspace'].image.copy()

    r = mcp_server._run_tool('show_grid', {})
    assert len(r) == 2
    assert r[1].type == 'image'
    assert r[1].mimeType == 'image/png'
    base64.b64decode(r[1].data)  # decodes cleanly

    after = mcp_server._state['workspace'].image
    assert list(before.getdata()) == list(after.getdata())


def test_save_writes_the_current_image(image_path, tmp_path):
    mcp_server._open_image({'path': image_path})
    mcp_server._run_tool('adjust', {'saturation': 0.0})
    out_path = str(tmp_path / 'out.png')
    r = mcp_server._save_image({'path': out_path})
    assert 'saved' in r[0].text
    saved = Image.open(out_path)
    assert saved.size == (64, 64)


def test_tool_before_open_image_reports_clearly():
    r = mcp_server._run_tool('adjust', {'saturation': 0.0})
    assert 'open_image' in r[0].text


def test_undo_restores_previous_state(image_path):
    mcp_server._open_image({'path': image_path})
    original = list(mcp_server._state['workspace'].image.getdata())
    mcp_server._run_tool('adjust', {'saturation': 0.0})
    mcp_server._run_tool('undo', {})
    restored = list(mcp_server._state['workspace'].image.getdata())
    assert restored == original


def test_an_edit_returns_the_edited_image(image_path):
    """An edit tool must return an image too, not only a text description."""
    mcp_server._open_image({'path': image_path})
    r = mcp_server._run_tool('stylise', {'effect': 'grayscale'})
    images = [c for c in r if c.type == 'image']
    assert len(images) == 1
    shown = Image.open(__import__('io').BytesIO(base64.b64decode(images[0].data)))
    assert shown.size == (64, 64)


def test_a_cutout_preview_shows_transparency_not_the_hidden_colours():
    """convert('RGB') on RGBA would drop alpha and show a cutout as its original rectangle."""
    img = Image.new('RGBA', (8, 8), (200, 40, 40, 0))
    c = mcp_server._image_content(img)
    shown = Image.open(__import__('io').BytesIO(base64.b64decode(c.data)))
    assert shown.mode == 'RGBA' and shown.getpixel((0, 0))[3] == 0


def test_a_slow_tool_does_not_block_the_event_loop(image_path, monkeypatch):
    """A slow tool must not run on the event loop itself, or the server cannot answer
    anything else until it finishes."""
    import asyncio
    import time
    import mcp.types as types
    mcp_server._open_image({'path': image_path})
    ran = []
    monkeypatch.setattr(mcp_server, '_run_tool',
                        lambda n, a: (time.sleep(0.5), ran.append(n),
                                      [types.TextContent(type='text', text='ok')])[2])
    server = mcp_server.create_server()
    handler = server.request_handlers[types.CallToolRequest]

    async def go():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1
        t = asyncio.create_task(ticker())
        req = types.CallToolRequest(method='tools/call',
                                    params=types.CallToolRequestParams(name='show_grid',
                                                                       arguments={}))
        await handler(req)
        t.cancel()
        return ticks
    assert asyncio.run(go()) >= 5
    assert ran == ['show_grid']
