"""An MCP server exposing figsurgeon's tools over stdio.

The package's whole premise is "an image editor a model can drive that states what actually
changed" -- `ImageWorkspace` (see `workspace.py`) already does the looking, editing, undo and
verification. This just puts it behind MCP so a model can reach it.

Images cannot travel as JSON, so everything here works on FILE PATHS: `open_image` starts a
workspace on a file, every edit tool from `tools.TOOL_SCHEMAS` acts on that workspace's
current image, and `save_image` writes it back out. Tools that only LOOK (show_grid,
preview_region, zoom, preview_object_mask, preview_colour_mask, refine_box, and anything
else that sets `ToolResult.preview`) return that preview as an MCP image so the model can
actually see it -- previews are never assigned as the new working image, exactly as
`ImageWorkspace.apply` already guarantees.

Every mutating tool's text response is `ToolResult.note` verbatim, which is where the
verification verdict lives (" | verified: ..." or " | CHECK FAILED: ..."). That verdict is
the product; this file does not reword it.

Run with: python -m figsurgeon.mcp_server
"""
import asyncio
import io

from PIL import Image
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .tools import TOOL_SCHEMAS
from .workspace import ImageWorkspace

SERVER_NAME = 'figsurgeon'

# Custom, file-path tools that sit alongside the schemas from tools.py -- images cannot be
# JSON, so opening and saving one has to be its own step.
_META_SCHEMAS = [
    {
        'name': 'open_image',
        'description': 'Open an image file and start a fresh editing session on it. Call '
                       'this before any other tool. Replaces any session already open.',
        'input_schema': {'type': 'object', 'properties': {
            'path': {'type': 'string', 'description': 'Path to the image file.'}},
            'required': ['path']},
    },
    {
        'name': 'save_image',
        'description': 'Write the current image (with every edit applied so far) to a '
                       'file path.',
        'input_schema': {'type': 'object', 'properties': {
            'path': {'type': 'string', 'description': 'Where to write the PNG.'}},
            'required': ['path']},
    },
]

ALL_SCHEMAS = _META_SCHEMAS + TOOL_SCHEMAS

_state = {'workspace': None, 'path': None}


# The edited image sent back after every change is capped at this side: a model sees it to
# judge the result, not to re-read pixel values, and a full 24-MP PNG per edit is tens of MB.
SHOWN_MAX_SIDE = 1568


def _image_content(img, max_side=None):
    buf = io.BytesIO()
    # RGBA stays RGBA: flattening a cutout to RGB shows the colours stored UNDER its
    # transparent pixels, i.e. the rectangle it was cut from.
    img = img if img.mode in ('RGB', 'RGBA') else img.convert('RGBA' if 'A' in img.mode
                                                                else 'RGB')
    if max_side and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(buf, format='PNG')
    import base64
    return types.ImageContent(type='image', data=base64.b64encode(buf.getvalue()).decode('ascii'),
                              mimeType='image/png')


def _open_image(args):
    path = args['path']
    img = Image.open(path)
    img.load()
    _state['workspace'] = ImageWorkspace(img)
    _state['path'] = path
    return [types.TextContent(type='text',
                              text=f'opened {path} ({img.size[0]}x{img.size[1]})')]


def _save_image(args):
    workspace = _state['workspace']
    if workspace is None:
        return [types.TextContent(type='text', text='no image open; call open_image first')]
    path = args['path']
    workspace.image.save(path)
    return [types.TextContent(type='text', text=f'saved {path}')]


def _run_tool(name, args):
    workspace = _state['workspace']
    if workspace is None:
        return [types.TextContent(type='text', text='no image open; call open_image first')]
    result = workspace.apply(name, args)
    content = [types.TextContent(type='text', text=result.note)]
    if result.preview is not None:
        content.append(_image_content(result.preview))
    elif result.mutates:
        # The agent loop shows the new image after every change; over MCP a model would
        # otherwise only read what an edit did, never see it, so send it explicitly here.
        content.append(types.TextContent(type='text', text='Current state of the image:'))
        content.append(_image_content(result.image, max_side=SHOWN_MAX_SIDE))
    return content


def create_server():
    server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=s['name'], description=s['description'],
                           inputSchema=s['input_schema']) for s in ALL_SCHEMAS]

    @server.call_tool()
    async def call_tool(name, arguments):
        arguments = arguments or {}
        try:
            # In a worker thread: an erase or a remap can take tens of seconds, and on the
            # event loop itself the server could answer nothing else until it finished.
            if name == 'open_image':
                return await asyncio.to_thread(_open_image, arguments)
            if name == 'save_image':
                return await asyncio.to_thread(_save_image, arguments)
            return await asyncio.to_thread(_run_tool, name, arguments)
        except Exception as e:
            return [types.TextContent(type='text', text=f'error: {e}')]

    return server


async def run():
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(run())


if __name__ == '__main__':
    main()
