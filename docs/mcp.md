# figsurgeon over MCP

`figsurgeon.mcp_server` exposes the package's tool schemas (`figsurgeon/tools.py`, ~33
operations) over the Model Context Protocol, backed by an `ImageWorkspace`. That gets you
undo and automatic verification through the protocol, not just the raw calls: every mutating
tool's response carries the same verdict text `ImageWorkspace.apply` produces (`| verified:
...` or `| CHECK FAILED: ...`), and the looking tools (`show_grid`, `preview_region`, `zoom`,
`preview_object_mask`, `preview_colour_mask`, `refine_box`) return an actual image so the
model can look before it commits to an edit.

## Install

```bash
pip install "figsurgeon[mcp]"
# add other extras as needed, e.g.:
pip install "figsurgeon[mcp,grounding,advanced]"
```

Run `figsurgeon doctor` to see what else is missing for the operations you want.

## Working model: file paths, not JSON images

Images can't travel as MCP JSON content, so the server works on paths on disk:

1. `open_image(path)` -- opens a file and starts a fresh editing session on it. Every tool
   call below acts on that session's current image.
2. Any tool from `figsurgeon/tools.py`'s schemas (`adjust`, `recolour_object`,
   `erase_object`, `remove_background`, `undo`, `reset`, ... the full list) -- same names,
   same arguments, as documented in `figsurgeon/tools.md`.
3. `save_image(path)` -- writes the current image (with every edit applied) to a file.

A tool that only looks (doesn't change the image) returns its note as text plus the picture
itself as an MCP image, and does **not** change what `save_image` will write.

## Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "figsurgeon": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "figsurgeon.mcp_server"]
    }
  }
}
```

## Claude Code

```bash
claude mcp add figsurgeon -- /path/to/venv/bin/python -m figsurgeon.mcp_server
```

## Worked example

A model driving this server through a typical session:

```
open_image({"path": "/tmp/photo.jpg"})
  -> "opened /tmp/photo.jpg (1600x1200)"

show_grid({})
  -> text: "grid overlay shown on the 1600x1200 image. Read box coordinates off the
     labelled lines; the image itself is unchanged."
  -> image: the grid overlay, to read coordinates off

preview_region({"boxes": [[420, 180, 780, 640]]})
  -> text: "(420, 180, 780, 640): 360x460 px, 10.4% of the frame, median RGB (91, 74, 58)"
  -> image: that box drawn on the photo, to check it's the right one

recolour_object({"box": [420, 180, 780, 640], "to_rgb": [30, 60, 180]})
  -> "recoloured the segmented object inside (420, 180, 780, 640) to (30, 60, 180)
     | verified: target hue reached in the masked region, 3% leaked outside"

save_image({"path": "/tmp/photo_edited.png"})
  -> "saved /tmp/photo_edited.png"
```

If a check fails, the response says so in the same text field (`CHECK FAILED: ...`) —
call `undo({})` rather than stacking another edit on a bad result.

## Notes

- `mcp` is an optional extra (`mcp>=1.0,<2` -- the 2.x line of the SDK renamed the low-level
  `Server` API this server is built on; see the SDK's own migration guide if upgrading).
- The server holds one image session per process, matching how an MCP stdio server is used
  (one client, one conversation). Calling `open_image` again replaces the current session.
