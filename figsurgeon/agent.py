"""A working LLM loop: plain-language instruction in, edited image out.

`tools.py` defines what the model can do; this drives the conversation that decides what
it should do, on a specific image it can actually see -- resolving "the sky" to a real
region and RGB, which is not a parsing problem the regex layer in `describe.py` can solve.

Three things this loop does that a plain send-tools-and-execute loop does not:
  * Look before editing: the looking tools return a preview and leave the working image
    untouched, so a wrong box costs a turn instead of the picture.
  * Verify after editing: every edit is checked against that operation's own intent
    (`workspace.verify`), since "it ran" and "it worked" are different claims.
  * Take it back: edits go through an `ImageWorkspace`, so `undo` is a tool the model
    can call.

Requires `anthropic` (pip install "figsurgeon[agent]") and an API key -- or any object
exposing `messages.create(...)`, passed as `client`.
"""
import base64
import io
import json

from .tools import TOOL_SCHEMAS
from .workspace import ImageWorkspace

SYSTEM_PROMPT = """You edit images by calling tools. You can see the image, so use that:
resolve references like "her shirt" or "the sky" by looking, not guessing.

Look before you edit. Most tools here take a pixel box, and a box estimated by eye from a
photograph is typically 10-30 px off -- enough to clip the object you meant. The looking
tools cost one turn and change nothing:
- show_grid puts labelled coordinates on the image. Use it before choosing any box.
- preview_region draws your box so you can see whether it actually contains the object.
- refine_box snaps a roughly-right box onto the object inside it.
- preview_object_mask / preview_colour_mask show exactly which pixels an edit would touch.
  Check these before recolouring or erasing: a hue that looks specific to you often matches
  most of the frame.
Skip them only for whole-image operations (brightness, grayscale, vignette, rotate).

Other rules that matter:
- Sample before you assume. To recolour something, call sample_colour on that region first
  to get its real RGB rather than assuming a generic value.
- On a PHOTOGRAPH, pass space='oklab' to isolate_colour / replace_colour / preview_colour_mask.
  The default hue window ignores lightness, so "the orange suit" also takes skin, hair and
  anything else warm: measured, 62.7% of that frame against 25.5%. For a WHITE, GREY or
  BLACK target the default cannot select anything at all (no meaningful hue) -- oklab can,
  but tighten oklab_tolerance to about 0.02, since at 0.06 a light grey reaches the white
  page as well.
- Read the result notes. They report measured effects, warnings, and a verification verdict.
  If a note says CHECK FAILED, the operation ran but did not do what it is meant to do: call
  undo and try a different approach instead of stacking another edit on top of it.
- If a note says an operation barely changed anything, do NOT repeat it with a bigger number
  without thinking about why -- an image with no colour cannot be saturated, and an
  already-vivid one clips.
- If a note says WARNING: nothing was erased/recoloured/isolated, the segmentation found no
  object in your box. The image is unchanged. Look at preview_object_mask and fix the box;
  repeating the same call will do nothing again.
- resize changes the size of the WHOLE image. To resize one thing inside it -- "make the
  star smaller" -- use scale_object instead, with a box around that object.
- generative_fill is the only tool that INVENTS content: it sends the boxed region to a
  hosted image model, so it costs real money per call and never returns the same thing
  twice. Reach for it when no deterministic tool can do the job -- plausible background
  behind something you erased, a texture continued across a gap. Try erase_object first;
  generative_fill is the fallback for when its inpainting smears. It comes back with NO
  verdict, only a seam number: zoom in on the edges before you say it worked.
  Without mask_to_object=True it may redraw the WHOLE box, background included, which is
  wrong whenever you only meant one object in it ("repaint the car" with a box that also
  has road or sky) -- set it whenever the box holds anything you want left untouched.
- Order matters for a few combinations, and the measured one is: denoise BEFORE brightening,
  never after. Brightening multiplies grain along with signal and no later denoise fully
  recovers from it.
- Boxes are [x0,y0,x1,y1] in pixels, origin top-left. You are told the image size; stay
  inside it.
- Prefer the fewest calls that achieve the request. Stop when done and say what you did.
- If a request is ambiguous or you lack the information to pick a region, say so instead of
  guessing wildly."""


JPEG_QUALITY = 88
# How much smaller JPEG has to be before its losses are worth taking. A photograph comes
# out 5x smaller and the difference is invisible to the model; a chart or a screenshot comes
# out BIGGER (measured: car_red 2.41 MB PNG against 0.44 MB JPEG; chart_bar 0.08 MB PNG
# against 0.10 MB JPEG), and JPEG ringing lands on exactly the 1 px axis lines and text this
# package exists to edit precisely. Requiring a real win means flat-colour images stay
# lossless without anything having to guess what kind of picture this is.
JPEG_WINS_BELOW = 0.6


def _encode(img, max_side=1568):
    """Base64-encode an image for the API. Returns (data, media_type).

    Downscales only when genuinely oversized: 1568 px is where the API downsamples anyway,
    so sending more costs tokens and latency for no extra detail. Sending LESS would be
    worse than either -- the model reports boxes in the coordinate frame of what it was told
    the size is, so silently shrinking the picture while quoting full-resolution dimensions
    would corrupt every box it returns.

    The format is chosen by measuring, not by a rule about content. The loop resends the
    whole conversation every turn and adds an image to it each time, so transport size is
    most of the wall-clock cost of an edit: five turns on one 1200 px photograph uploaded
    36 MB as PNG. Both encodings are tried and the smaller wins, with a margin -- see
    JPEG_WINS_BELOW.
    """
    img = img.convert('RGB')
    if max(img.size) > max_side:
        from PIL import Image as _Image
        # round(), not int(): truncation makes the long side 1567 rather than the 1568 it
        # was scaled to, so the "downscale only when oversized" boundary is off by a pixel.
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.size[0] * scale)),
                          max(1, round(img.size[1] * scale))), _Image.LANCZOS)
    png = io.BytesIO()
    img.save(png, format='PNG')
    jpg = io.BytesIO()
    img.save(jpg, format='JPEG', quality=JPEG_QUALITY)
    if jpg.tell() < JPEG_WINS_BELOW * png.tell():
        return base64.standard_b64encode(jpg.getvalue()).decode(), 'image/jpeg'
    return base64.standard_b64encode(png.getvalue()).decode(), 'image/png'


def _image_block(img):
    data, media_type = _encode(img)
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type,
                                        'data': data}}


# The model when this loop builds its own Anthropic SDK client, and when a `client` is
# passed in (the OpenRouter path, whose default matches `evals/openrouter_edit.py`). A
# shared default would send an OpenRouter id to the Anthropic API, which does not serve it.
ANTHROPIC_MODEL = 'claude-sonnet-5'
OPENROUTER_MODEL = 'z-ai/glm-5.3-flash'


def edit(image, instruction, api_key=None, model=None, max_turns=10,
         client=None, verbose=False, workspace=None):
    """Run an instruction against an image. Returns (final_image, transcript).

    `transcript` is a list of (tool_name, args, note) plus any final text, so the caller can
    see exactly what was done and why -- an image that changed for reasons you cannot
    inspect is not much use. Pass a `workspace` to keep state across several `edit` calls
    (a conversation), or read `workspace.summary()` afterwards for the verification verdicts.
    """
    if client is None:
        # Only needed when we have to build a client: requiring the SDK even when the caller
        # supplies their own would block every offline use of this loop -- a stub in a test,
        # a proxy, a different SDK.
        try:
            import anthropic
        except ImportError:
            raise ImportError('the agent loop needs the anthropic package: '
                              'pip install "figsurgeon[agent]"  '
                              '(or pass your own `client`)')
        client = anthropic.Anthropic(api_key=api_key)
        model = model or ANTHROPIC_MODEL
    model = model or OPENROUTER_MODEL
    ws = workspace or ImageWorkspace(image)
    transcript = []

    messages = [{
        'role': 'user',
        'content': [
            _image_block(ws.image),
            {'type': 'text',
             'text': f'Image size: {ws.image.size[0]}x{ws.image.size[1]} pixels.\n\n'
                     f'Instruction: {instruction}'},
        ],
    }]

    for turn in range(max_turns):
        resp = client.messages.create(
            model=model, max_tokens=16000,
            # The prefix is rendered tools -> system -> messages, and BOTH of those are
            # byte-identical on every turn of an edit, so one breakpoint at the end of the
            # system prompt caches the whole stable part. Only the growing conversation
            # after it is paid for again. A ten-turn edit resends this on every call.
            system=[{'type': 'text', 'text': SYSTEM_PROMPT,
                     'cache_control': {'type': 'ephemeral'}}],
            tools=TOOL_SCHEMAS, messages=messages)

        # `getattr` rather than `resp.stop_reason`, because this loop's contract is any
        # object exposing `messages.create` -- a stub, a proxy, a different SDK -- and only
        # the real API sets this. A refusal returns HTTP 200 with no tool calls, so without
        # this it is indistinguishable from the model deciding it is finished.
        if getattr(resp, 'stop_reason', None) == 'refusal':
            why = getattr(resp, 'stop_details', None)
            transcript.append(('refused', None,
                               f'the model declined this instruction '
                               f'({getattr(why, "category", "no category given")})'))
            break

        tool_uses = [b for b in resp.content if b.type == 'tool_use']
        for t in [b.text for b in resp.content if b.type == 'text']:
            transcript.append(('assistant', None, t))
            if verbose:
                print(f'[model] {t}')

        if not tool_uses:
            # Running out of output tokens also returns HTTP 200 with whatever was produced,
            # which can be nothing at all -- no text, no tool calls. That is the same shape as
            # the model deciding it is finished, and `evals/agent_loop/` recorded runs ending
            # this way as considered stops. Say which one it was.
            if getattr(resp, 'stop_reason', None) == 'max_tokens':
                transcript.append(('truncated', None,
                                   'the model ran out of output tokens mid-turn; it did not '
                                   'choose to stop here'))
            break

        messages.append({'role': 'assistant', 'content': resp.content})
        results, images, changed = [], [], False
        for tu in tool_uses:
            result = ws.apply(tu.name, tu.input)
            transcript.append((tu.name, tu.input, result.note))
            if verbose:
                print(f'[tool] {tu.name}({json.dumps(tu.input)[:80]}) -> {result.note[:110]}')
            results.append({'type': 'tool_result', 'tool_use_id': tu.id,
                            'content': result.note})
            if result.preview is not None:
                images.append((f'Result of {tu.name}:', result.preview))
            changed = changed or result.mutates

        # What comes back has to match what was asked for. A looking tool's answer is its
        # preview image -- returning the unchanged working image instead would make
        # show_grid useless. An edit's answer is the new working image, because several
        # operations here are ones where the measured number and the visual outcome
        # genuinely diverge, and a note saying "vignette applied" cannot report how it looks.
        for caption, preview in images:
            results.append({'type': 'text', 'text': caption})
            results.append(_image_block(preview))
        if changed:
            results.append({'type': 'text', 'text': 'Current state of the image being edited:'})
            results.append(_image_block(ws.image))
        messages.append({'role': 'user', 'content': results})
    else:
        # The loop ran out rather than the model stopping -- without this entry the two
        # looked identical in the transcript.
        transcript.append(('max_turns', None,
                           f'stopped after {max_turns} turns; the model had not finished'))

    return ws.image, transcript
