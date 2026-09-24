"""LLM tool-calling interface: drive the whole package from an AI model instead of regex.

`describe.py`'s regex parser handles only anticipated phrasings and cannot resolve a
reference to image content, pick a bounding box, chain conditional steps, or choose a
parameter sensibly -- all of which need to see the image. This module exposes the engines as
typed tool schemas instead, for a model that can see the image to call.

    from figsurgeon.tools import TOOL_SCHEMAS, dispatch
    result_img, note = dispatch(tool_name, tool_input, image)

`dispatch` is the only entry point: it validates arguments and never raises on a bad one --
the error comes back as a note the model can read and correct on its next turn.
"""
import numpy as np
from PIL import Image

from . import photo, advanced, rebrand, objects, locate
from .composite import enforce_region


class ToolResult(tuple):
    """(image, note), plus what a caller needs to drive a *loop* rather than one call.

    Subclasses tuple so `out, note = dispatch(...)` keeps working unchanged.

      `mutates`  False for a tool that inspects rather than edits, so the agent loop doesn't
                 overwrite the working image with an annotated preview (e.g. "show grid").
      `preview`  an image to show the caller that is not the working image.
      `data`     measured values (a refined box, a sampled colour); also how `workspace.apply`
                 reads `region`/`mask` to hand a localised or masked edit to `verify()`.
    """

    def __new__(cls, image, note, preview=None, mutates=True, data=None):
        self = super().__new__(cls, (image, note))
        self.image = image
        self.note = note
        self.preview = preview
        self.mutates = bool(mutates)
        self.data = data or {}
        return self


LOOKING_TOOLS = [
    # These do not change the image. Every spatial tool below takes a box, and a box guessed
    # in one shot from a photograph is routinely off by enough to ruin the edit.
    {
        'name': 'show_grid',
        'description': (
            'Overlay a labelled pixel-coordinate grid on the image and show it. Use this '
            'BEFORE any tool that takes a box, on any image where the region matters. Read '
            'the box coordinates off the grid lines instead of estimating them -- estimated '
            'boxes are routinely 10-30 px off, which is enough to clip the object. Does not '
            'modify the image.'),
        'input_schema': {'type': 'object', 'properties': {
            'step': {'type': 'integer',
                     'description': 'Grid spacing in pixels. Omit for an automatic round '
                                    'step (~10 divisions). Use a smaller step to read a '
                                    'small region precisely.'}}, 'required': []},
    },
    {
        'name': 'preview_region',
        'description': (
            'Draw one or more proposed boxes on the image and show the result, with '
            'measured facts about each region. Use this to CHECK a box before an '
            'irreversible edit (erase_object, recolour_object, remove_object, crop). Does '
            'not modify the image.'),
        'input_schema': {'type': 'object', 'properties': {
            'boxes': {'type': 'array',
                      'description': '[[x0,y0,x1,y1], ...] regions to preview',
                      'items': {'type': 'array', 'items': {'type': 'integer'}}},
            'labels': {'type': 'array', 'items': {'type': 'string'}},
            'dim_outside': {'type': 'number',
                            'description': '0..1: darken everything outside the boxes, to '
                                           'check that nothing wanted falls outside them.'}},
            'required': ['boxes']},
    },
    {
        'name': 'refine_box',
        'description': (
            'Snap a roughly-correct box onto the object actually inside it, using '
            'segmentation, and show the before/after boxes. Use after estimating a box for '
            'a distinct object. Reports honestly when it could NOT refine (the object does '
            'not contrast with its surroundings, or nothing distinct was found) and returns '
            'the original box unchanged in that case. Does not modify the image.'),
        'input_schema': {'type': 'object', 'properties': {
            'box': {'type': 'array', 'items': {'type': 'integer'}}}, 'required': ['box']},
    },
    {
        'name': 'zoom',
        'description': (
            'Show a region enlarged. Two uses: check what is inside a box at a readable '
            'size before editing, and inspect an edit afterwards at its boundary, where '
            'artifacts live (halos, hard cut-out edges, inpainting smears) -- these are '
            'invisible in the full frame. Does not modify the image.'),
        'input_schema': {'type': 'object', 'properties': {
            'box': {'type': 'array', 'items': {'type': 'integer'}}}, 'required': ['box']},
    },
    {
        'name': 'preview_colour_mask',
        'description': (
            'Show exactly which pixels a hue-based edit would touch -- they stay in '
            'colour, everything else is dimmed -- WITHOUT applying it. Use before isolate_colour or replace_colour to check the '
            'tolerance: too wide also catches the sky, too narrow leaves patches behind. '
            'Reports the fraction of the image matched. Does not modify the image.'),
        'input_schema': {'type': 'object', 'properties': {
            'target_rgb': {'type': 'array', 'items': {'type': 'integer'}},
            'hue_tolerance': {'type': 'number', 'description': 'Default 0.07.'},
            'space': {'type': 'string', 'enum': ['hsv', 'oklab'],
                      'description': "Preview the space you intend to edit in -- 'hsv' "
                                     "(default) for charts, 'oklab' for photographs. "
                                     "Measured both ways at their defaults: 'hsv' keeps a "
                                     "chart series' translucent band and 'oklab' does not; "
                                     "'oklab' follows a shaded object and 'hsv' takes "
                                     "everything of that hue."},
            'oklab_tolerance': {'type': 'number', 'description': "For space='oklab': 0.04 "
                                                                 'tight, 0.06 default.'}},
            'required': ['target_rgb']},
    },
    {
        'name': 'preview_object_mask',
        'description': (
            'Show exactly which pixels segmentation finds inside a box -- they stay in '
            'colour, everything else is dimmed -- WITHOUT applying anything. Use before recolour_object / erase_object / isolate_object '
            'to check the segmentation actually found the object rather than the whole box '
            'or a fragment of it. Does not modify the image.'),
        'input_schema': {'type': 'object', 'properties': {
                'point': {'type': 'array', 'items': {'type': 'integer'},
                          'description': 'ALTERNATIVE to box: [x, y] on the object itself (or [[x,y],[x,y]] for a large one). Use it when a rectangle around the object would be mostly background -- a rocket against sky, a lamp post. Cannot be combined with box.'},
            'box': {'type': 'array', 'items': {'type': 'integer'}},
            'boxes': {'type': 'array', 'description': 'several regions at once',
                      'items': {'type': 'array', 'items': {'type': 'integer'}}}},
            'required': []},
    },
]

STATE_TOOLS = [
    # Handled by ImageWorkspace, not dispatch -- they act on edit history, which stateless
    # dispatch has none of. Listed so a model driving a workspace knows they exist.
    {
        'name': 'undo',
        'description': (
            'Revert the last edit (or the last N). Use this the moment an edit lands in the '
            'wrong place or a result note reports CHECK FAILED -- editing on top of a bad '
            'result compounds the damage and cannot be unwound later.'),
        'input_schema': {'type': 'object', 'properties': {
            'steps': {'type': 'integer', 'description': 'How many edits to revert. Default 1.'}},
            'required': []},
    },
    {
        'name': 'reset',
        'description': 'Discard every edit and return to the original image.',
        'input_schema': {'type': 'object', 'properties': {}, 'required': []},
    },
]

TOOL_SCHEMAS = LOOKING_TOOLS + STATE_TOOLS + [
    {
        'name': 'isolate_colour',
        'description': (
            'Keep pixels near one hue in full colour and desaturate everything else '
            '("colour pop"). Use when the user wants one coloured thing to stand out. '
            'IMPORTANT: pick target_rgb by LOOKING at the image -- sample the actual '
            'colour of the object, do not assume a generic "red" is (255,0,0). Objects '
            'that share a hue with their background cannot be separated this way; use '
            'blur_background or remove_background instead in that case.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'target_rgb': {'type': 'array', 'items': {'type': 'integer'},
                               'description': 'RGB of the colour to KEEP, e.g. [220,30,30]'},
                'hue_tolerance': {'type': 'number',
                                  'description': 'How wide a hue band to keep, 0.02 (strict) '
                                                 'to 0.12 (loose). Default 0.07.'},
                'flatten': {'type': 'number',
                            'description': '0 = desaturated area keeps true luminance, '
                                           '1 = flat mid-grey (stylised). Default 0.5.'},
                'space': {'type': 'string', 'enum': ['hsv', 'oklab'],
                          'description': (
                              "How to decide which pixels count as that colour. 'hsv' "
                              "(default) is a hue window. 'oklab' is perceptual distance, "
                              "and is the better choice on almost anything photographic: "
                              "a hue window ignores lightness, so an object that is shaded "
                              "gets cut in half while skin, hair and anything else warm "
                              "gets taken. It also handles NEUTRALS, which a hue window "
                              "cannot -- white, grey and black have no meaningful hue. "
                              "Measured on the astronaut, each at its default tolerance: "
                              "HSV takes 62.7 % of the frame in 2403 fragments -- her "
                              "face, hair, the flag -- against OKLab's 25.5 % in 154. "
                              "'hsv' is still the default because on a CHART it wins at "
                              "the defaults: it keeps a series' translucent confidence "
                              "band, which perceptual distance calls a different colour. "
                              "Preview the mask if unsure."),
                          },
                'oklab_tolerance': {'type': 'number',
                                    'description': "Perceptual distance for space='oklab': "
                                                   '0.04 tight, 0.06 default, 0.12 loose '
                                                   '(0.12 takes most of a photograph). '
                                                   'Ignored for hsv.'},
            },
            'required': ['target_rgb'],
        },
    },
    {
        'name': 'replace_colour',
        'description': (
            'Change pixels of one hue to another hue, preserving each pixel\'s own '
            'brightness and shading so folds and highlights survive. Use for "make the red '
            'car blue". If the user references a colour BY REGION ("the orange of her '
            'shirt"), first call sample_colour on that region to get its RGB.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'from_rgb': {'type': 'array', 'items': {'type': 'integer'}},
                'to_rgb': {'type': 'array', 'items': {'type': 'integer'}},
                'hue_tolerance': {'type': 'number', 'description': 'Default 0.07.'},
                'space': {'type': 'string', 'enum': ['hsv', 'oklab'],
                          'description': (
                              "How to decide which pixels count as that colour. 'hsv' "
                              "(default) is a hue window -- right for a chart's flat "
                              "series colours. 'oklab' is perceptual distance -- right "
                              "for a PHOTOGRAPH, where an object is shaded and a hue "
                              "window also grabs skin, hair and anything else warm. "
                              "Measured on the astronaut: at the tightest hue window HSV "
                              "still takes 24 % of the frame in 6240 fragments, OKLab "
                              "takes 14.6 % in 221. Neither handles a NEUTRAL series well "
                              "-- on a grey bar series 'hsv' recoloured the orange one "
                              "instead and 'oklab' selected the white page; sample the "
                              "colour and preview the mask first. Preview the mask if "
                              "unsure."),
                          },
                'oklab_tolerance': {'type': 'number',
                                    'description': "Perceptual distance for space='oklab': "
                                                   '0.04 tight, 0.06 default, 0.12 loose '
                                                   '(0.12 takes most of a photograph). '
                                                   'Ignored for hsv.'},
            },
            'required': ['from_rgb', 'to_rgb'],
        },
    },
    {
        'name': 'sample_colour',
        'description': (
            'Read the median colour of a rectangular region. Use this to resolve a '
            'reference like "the orange of her shirt" into an actual RGB value before '
            'calling replace_colour. Returns the colour as a note; does not modify '
            'the image.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'box': {'type': 'array', 'items': {'type': 'integer'},
                        'description': '[x0,y0,x1,y1] region to sample'},
            },
            'required': ['box'],
        },
    },
    {
        'name': 'blur_background',
        'description': (
            'Simulated shallow depth of field. focus="subject" uses a neural segmentation '
            'mask and is the right default for a photo with a clear subject. Pass a box '
            'instead only when the subject is not what the model would segment.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'focus': {'type': 'string',
                          'description': '"subject" for the neural mask, or "box" to use '
                                         'focus_box'},
                'focus_box': {'type': 'array', 'items': {'type': 'integer'}},
                'radius': {'type': 'integer', 'description': 'Blur strength, default 14.'},
            },
            'required': [],
        },
    },
    {
        'name': 'remove_background',
        'description': 'Cut the subject out, making the background transparent (neural).',
        'input_schema': {'type': 'object', 'properties': {}, 'required': []},
    },
    {
        'name': 'replace_background',
        'description': 'Cut the subject out and composite it onto a solid colour.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'colour_rgb': {'type': 'array', 'items': {'type': 'integer'}},
            },
            'required': ['colour_rgb'],
        },
    },
    {
        'name': 'remove_object',
        'description': (
            'Paint out a region using its surroundings (inpainting). Works well removing a '
            'thin object from a smooth or repetitive background (a mast against sky, a '
            'wire, a blemish). Works BADLY at reconstructing something structurally unique '
            'that was fully covered -- it interpolates texture, it does not invent content. '
            'Requires an explicit region: there is no object detection here.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'box': {'type': 'array', 'items': {'type': 'integer'},
                        'description': '[x0,y0,x1,y1] to paint out'},
                'polyline': {'type': 'array',
                             'description': '[[x,y],[x,y],...] for a thin object like a '
                                            'wire or mast; used instead of box',
                             'items': {'type': 'array', 'items': {'type': 'integer'}}},
                'radius': {'type': 'integer', 'description': 'Default 6.'},
            },
            'required': [],
        },
    },
    {
        'name': 'adjust',
        'description': (
            'Brightness / contrast / saturation / sharpness. Each is a multiplier where '
            '1.0 = unchanged, >1 increases, <1 decreases. Combine several in one call.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'brightness': {'type': 'number'},
                'contrast': {'type': 'number'},
                'saturation': {'type': 'number'},
                'sharpness': {'type': 'number'},
            },
            'required': [],
        },
    },
    {
        'name': 'stylise',
        'description': 'Apply a named stylistic effect.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'effect': {'type': 'string',
                           'enum': ['sepia', 'grayscale', 'sketch', 'vignette']},
                'strength': {'type': 'number',
                             'description': 'For vignette: 0.35 subtle to 0.75 dramatic.'},
            },
            'required': ['effect'],
        },
    },
    {
        'name': 'denoise',
        'description': (
            'Reduce noise/grain. Strength ~3 light to ~15 heavy; high values soften real '
            'detail as well as noise. DENOISE BEFORE BRIGHTENING, not after: brightening '
            'multiplies the grain along with the signal, and no denoise strength fully '
            'recovers from that. Measured on a dark, grainy frame -- denoise(18) then '
            'brighten(2.2) ends at grain 2.45, the same two steps reversed at 5.37, from a '
            'source at 5.40. Rescuing an underexposed photo also needs a HIGHER strength '
            'than a normally-lit one; if the note says noise barely dropped, raise it.'),
        'input_schema': {
            'type': 'object',
            'properties': {'strength': {'type': 'integer'}},
            'required': [],
        },
    },
    {
        'name': 'auto_white_balance',
        'description': (
            'Grey-world colour correction. WARNING: this assumes the scene average should '
            'be neutral, which is FALSE when a strong colour genuinely belongs to the '
            'subject (an orange spacesuit, a sunset, a forest). Nothing measurable in the '
            'pixels distinguishes "colour cast" from "colourful subject" -- the returned '
            'note reports how large a shift was applied so it can be judged, but check the '
            'result rather than trusting it.'),
        'input_schema': {'type': 'object', 'properties': {}, 'required': []},
    },
    {
        'name': 'clean_transparent_box',
        'description': (
            'Clean data bleeding through a semi-transparent legend or annotation box in a '
            'chart, while keeping the legend\'s own marker swatches and text. Requires an '
            'approximate box around the legend. Draw the box TIGHT to the legend and read '
            'it off show_grid: the box interior is flat-filled, so any real plot data '
            'inside an over-large box is destroyed, and no check catches that -- the edit '
            'stayed inside the box it was given. Unreliable on legends with heavy, '
            'high-opacity data overlap -- inspect the result.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'box': {'type': 'array', 'items': {'type': 'integer'}},
            },
            'required': ['box'],
        },
    },
    {
        'name': 'remap_colormap',
        'description': (
            'Re-theme a continuous colormap to a new gradient, preserving what the colour '
            'encodes -- each pixel keeps its position along the scale. Use for matching a '
            'chart to a brand palette. Selects by COLOUR, so it covers the plot area, the '
            'colorbar and any other panel in one call: omit `box` (the default) and the '
            'whole figure is re-themed together, which is what keeps the key matching the '
            'cells. Pass a box only to PROTECT part of the figure from the change.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'box': {'type': 'array', 'items': {'type': 'integer'}},
                'source_cmap': {'type': 'string',
                                'description': 'matplotlib colormap name, e.g. "viridis"'},
                'new_colours': {'type': 'array', 'items': {'type': 'string'},
                                'description': 'Hex gradient stops, e.g. ["#00407A","#52BDEC"]'},
            },
            'required': ['new_colours'],
        },
    },
    {
        'name': 'replace_font',
        'description': (
            'Re-render every text label in a different font via OCR. Rotated labels (a '
            'vertical axis title) MUST be passed in rotated_regions with their angle -- '
            'plain OCR reads rotated text as garbage at deceptively high confidence. '
            'Returns a note listing what OCR actually read, so it can be checked.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'rotated_regions': {
                    'type': 'array',
                    'description': '[{"box":[x0,y0,x1,y1],"angle":90}, ...]',
                    'items': {'type': 'object'}},
                'min_confidence': {'type': 'number', 'description': 'Default 55.'},
            },
            'required': [],
        },
    },
    {
        'name': 'crop',
        'description': 'Crop to a rectangle. Coordinates are clamped to the image bounds; '
                       'an empty or fully out-of-bounds box returns an error note.',
        'input_schema': {'type': 'object', 'properties': {
            'box': {'type': 'array', 'items': {'type': 'integer'},
                    'description': '[x0,y0,x1,y1]'}}, 'required': ['box']},
    },
    {
        'name': 'rotate',
        'description': 'Rotate by any angle (degrees, clockwise). expand=true keeps the '
                       'whole image and grows the canvas; false keeps the original size '
                       'and clips the corners.',
        'input_schema': {'type': 'object', 'properties': {
            'angle': {'type': 'number'},
            'expand': {'type': 'boolean'}}, 'required': ['angle']},
    },
    {
        'name': 'flip',
        'description': 'Mirror the image horizontally or vertically.',
        'input_schema': {'type': 'object', 'properties': {
            'axis': {'type': 'string', 'enum': ['horizontal', 'vertical']}},
            'required': ['axis']},
    },
    {
        'name': 'resize',
        'description': 'Resize by a scale factor, or to a target width and/or height. '
                       'Giving only one of width/height preserves the aspect ratio.',
        'input_schema': {'type': 'object', 'properties': {
            'scale': {'type': 'number'},
            'width': {'type': 'integer'},
            'height': {'type': 'integer'}}, 'required': []},
    },
    {
        'name': 'recolour_object',
        'description': (
            'Recolour ONE object found inside a box, using segmentation -- not '
            'hue matching. Use this instead of replace_colour when the object shares a hue '
            'with other things elsewhere in the image that must NOT change (e.g. "make the '
            'suit blue" on a photo that also has a red-striped flag: replace_colour would '
            'hit both, this hits only the segmented object). Requires an approximate box '
            'around the object -- there is no object detection here. Pass `boxes` to '
            'recolour several objects to the same colour in one call.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'box': {'type': 'array', 'items': {'type': 'integer'},
                        'description': '[x0,y0,x1,y1] around the object'},
                'point': {'type': 'array', 'items': {'type': 'integer'},
                          'description': 'ALTERNATIVE to box: [x, y] on the object itself, or [[x,y],[x,y]] for two or three points on a large one. Use this when a rectangle around the object would be mostly background -- a rocket against sky, a lamp post, a person at the end of a corridor. Measured on the rocket photo: from a box every segmenter returns THE SKY; from two points on the body it returns the rocket. Cannot be combined with box: this model ignores points when a box is present, so passing both is an error.'},
                'boxes': {'type': 'array',
                          'description': '[[x0,y0,x1,y1], ...] for several objects at once',
                          'items': {'type': 'array', 'items': {'type': 'integer'}}},
                'to_rgb': {'type': 'array', 'items': {'type': 'integer'}},
                'preserve_shading': {'type': 'boolean',
                                      'description': 'Keep folds/highlights. Default true.'},
                'subject': {'type': 'string',
                            'description': 'What you are recolouring, in your own words ("the suit", "the red car"). It does NOT find the object -- the box does that -- it CHECKS the result. It is the only evidence here that does not come from the box, so it is the only thing that can tell a complete recolour from one that did half the object, or from one that recoloured the sky instead of the rocket. Pass it whenever you know what you are pointing at. Positional wording is fine and costs nothing -- "the middle sheep", "the chair on the right", "the second saddle from the front": the box is what says WHICH one, so those words are dropped before the check resolves the phrase and the note tells you they were.'},
            },
            'required': ['to_rgb'],
        },
    },
    {
        'name': 'isolate_object',
        'description': (
            'Keep ONE object found inside a box in colour, desaturate everything else -- '
            'the spatial counterpart to isolate_colour. Use this instead of isolate_colour '
            'when the object shares a hue with its background (isolate_colour cannot '
            'separate them by hue in that case). Requires an approximate box. For several '
            'objects at once ("the eyes", "both headlights") pass `boxes` -- do NOT call '
            'this twice, because the second call would desaturate the object the first one '
            'kept.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'point': {'type': 'array', 'items': {'type': 'integer'},
                          'description': 'ALTERNATIVE to box: [x, y] on the object itself (or [[x,y],[x,y]] for a large one). Use it when a rectangle around the object would be mostly background -- a rocket against sky, a lamp post. Cannot be combined with box.'},
                'box': {'type': 'array', 'items': {'type': 'integer'}},
                'boxes': {'type': 'array',
                          'description': '[[x0,y0,x1,y1], ...] for several objects at once, '
                                         'instead of box',
                          'items': {'type': 'array', 'items': {'type': 'integer'}}},
                'flatten': {'type': 'number',
                            'description': '0 = true luminance, 1 = flat mid-grey. Default 0.5.'},
                'subject': {'type': 'string',
                            'description': 'What you are keeping in colour, in your own words ("the chair", "the red car"). It does NOT find the object -- the box does -- it CHECKS the result: without it the verdict can only say that SOMETHING in the box kept its colour and everything outside lost it, not that the thing you meant is what kept it. Positional wording is fine ("the chair on the right"): the box is what says which one.'},
            },
            'required': [],
        },
    },
    {
        'name': 'erase_object',
        'description': (
            'Segment and inpaint out ONE object found inside a box -- more precise than '
            'remove_object\'s box-shaped paint-out when the object doesn\'t fill its box. '
            'Works well on a small object against a smooth/repetitive background; smears '
            'on a large object, since inpainting has no information about what was behind '
            'it. Returns a warning when the segmented region exceeds ~8% of the frame.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'point': {'type': 'array', 'items': {'type': 'integer'},
                          'description': 'ALTERNATIVE to box: [x, y] on the object itself (or [[x,y],[x,y]] for a large one). Use it when a rectangle around the object would be mostly background -- a rocket against sky, a lamp post. Cannot be combined with box.'},
                'box': {'type': 'array', 'items': {'type': 'integer'}},
                'radius': {'type': 'integer', 'description': 'Default 6.'},
                'subject': {'type': 'string',
                            'description': 'What you are erasing, in your own words ("the bird", "the red car"). It does NOT find the object -- the box does -- it CHECKS the result, and the check is turned around from the other object tools: the thing being graded is gone from the output if the erase worked, so this is used to check that it is no longer findable where the box was. Without it, the verdict can only say that something changed inside the box, not whether the named thing is actually gone. Positional wording is fine ("the chair on the right").'},
            },
            'required': [],
        },
    },
    {
        'name': 'scale_object',
        'description': (
            'Resize ONE object found inside a box, in place -- shrinks or grows it about '
            'its own centre and fills what a shrink leaves behind. Use this for "make the '
            'star smaller" or "make the logo bigger"; resize changes the size of the WHOLE '
            'image and cannot touch one object within it. Growing can reach past the box '
            'you drew -- the tool reports the region it actually touched.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'point': {'type': 'array', 'items': {'type': 'integer'},
                          'description': 'ALTERNATIVE to box: [x, y] on the object itself (or [[x,y],[x,y]] for a large one). Cannot be combined with box.'},
                'box': {'type': 'array', 'items': {'type': 'integer'}},
                'scale': {'type': 'number',
                          'description': '1.0 = unchanged. Below 1 shrinks the object, above 1 grows it -- e.g. 0.5 halves its linear size, 2.0 doubles it.'},
            },
            'required': ['scale'],
        },
    },
    {
        'name': 'extract_object',
        'description': 'Cut ONE object found inside a box out as a transparent-background '
                       'RGBA image, using segmentation. Requires an approximate box.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'point': {'type': 'array', 'items': {'type': 'integer'},
                          'description': 'ALTERNATIVE to box: [x, y] on the object itself (or [[x,y],[x,y]] for a large one). Use it when a rectangle around the object would be mostly background -- a rocket against sky, a lamp post. Cannot be combined with box.'},
                'box': {'type': 'array', 'items': {'type': 'integer'}},
                'subject': {'type': 'string',
                            'description': 'What you are cutting out, in your own words ("the dog", "the red car"). It does NOT find the object -- the box does -- it CHECKS the result: without it the verdict can only say the cutout is neither empty nor the whole frame, and a cutout of the BACKGROUND inside the box looks exactly like a correct one. Positional wording is fine ("the chair on the right").'},
            },
            'required': [],
        },
    },
    {
        'name': 'generative_fill',
        'description': (
            'Redraw ONE boxed region with a hosted image model, then paste only that box '
            'back -- everything outside it is unchanged. Use it for what the deterministic '
            'tools CANNOT do: inventing plausible content, such as filling a hole so it '
            'matches the background behind it, or continuing a texture across a gap. '
            'For erasing an object prefer erase_object first: it is free, instant and '
            'checked, and this is the fallback when its inpainting smears. '
            'Costs a network call and real money, and the result is not reproducible. '
            'The surrounding background is sent as context automatically. '
            'Say in `instruction` what the region should CONTAIN, not what to remove. '
            'Set `mask_to_object=True` to redraw only ONE object inside the box (found by '
            'the same segmenter erase_object/recolour_object use) rather than the whole '
            'rectangle -- use this whenever the box also contains background you want left '
            'exactly as it was, e.g. "repaint the car" with a box that also has road or sky '
            'in it.'),
        'input_schema': {'type': 'object', 'properties': {
            'box': {'type': 'array', 'items': {'type': 'integer'},
                    'description': '[x0, y0, x1, y1] -- the only pixels that may change.'},
            'instruction': {'type': 'string',
                            'description': 'what this region should look like, e.g. '
                                           '"plain brick wall continuing the wall around it"'},
            'pad': {'type': 'number',
                    'description': 'how much surrounding background to send as context, as '
                                   'a fraction of the box size per side. Default 0.6.'},
            'mask_to_object': {'type': 'boolean',
                              'description': 'redraw only the segmented object inside the '
                                             'box, not the whole rectangle. Default false.'}},
            'required': ['box', 'instruction']},
    },
    {
        'name': 'add_text',
        'description': 'Draw a text label onto the image (annotation, caption, watermark).',
        'input_schema': {'type': 'object', 'properties': {
            'text': {'type': 'string'},
            'position': {'type': 'array', 'items': {'type': 'integer'},
                         'description': '[x,y] top-left of the text'},
            'size': {'type': 'integer'},
            'colour_rgb': {'type': 'array', 'items': {'type': 'integer'}}},
            'required': ['text']},
    },
]


def _measure_effect(before, after, kind):
    """Measure whether an operation actually changed the image, and by how much.

    Several operations can silently no-op and still report success. `saturation=1.6` on a
    greyscale photo (coins, brick, clock, camera) leaves mean saturation at 0.000 -> 0.000;
    on an already-vivid image (colorwheel) it moves 0.706 -> 0.736, a 4% change when 60% was
    requested, because the channels are already clipping. A model driving these tools reads
    "success", cannot tell nothing happened, and has no basis to try something else. Every
    measurable operation reports its actual measured effect so the caller can react.
    """
    import numpy as np
    b = np.array(before.convert('RGB')).astype(float)
    a = np.array(after.convert('RGB')).astype(float)
    if kind == 'saturation':
        def sat(x):
            mx, mn = x.max(axis=2), x.min(axis=2)
            return float(((mx - mn) / np.maximum(mx, 1e-6)).mean())
        s0, s1 = sat(b), sat(a)
        if s0 < 0.02:
            return (f'WARNING: this image is essentially greyscale (saturation {s0:.3f}), '
                    f'so a saturation change has almost no effect. Use a different '
                    f'operation, or accept that the image has no colour to adjust.')
        if abs(s1 - s0) / max(s0, 1e-6) < 0.05:
            return (f'WARNING: saturation barely moved ({s0:.3f} -> {s1:.3f}) -- the image '
                    f'is likely already near maximum saturation, so the channels clip. '
                    f'A larger factor will not help much.')
        return f'saturation {s0:.3f} -> {s1:.3f}'
    if kind == 'brightness':
        return f'mean brightness {b.mean():.1f} -> {a.mean():.1f}'
    if kind == 'contrast':
        return f'contrast (std) {b.std():.1f} -> {a.std():.1f}'
    if kind == 'denoise':
        from scipy.ndimage import gaussian_filter
        g0, g1 = b.mean(axis=2), a.mean(axis=2)
        n0 = float(np.abs(g0 - gaussian_filter(g0, 2)).mean())
        n1 = float(np.abs(g1 - gaussian_filter(g1, 2)).mean())
        if n1 >= n0:
            return (f'WARNING: high-frequency energy did not drop ({n0:.2f} -> {n1:.2f}). '
                    f'On very smooth images non-local-means can introduce faint patch '
                    f'artifacts rather than remove noise -- check the result, and skip '
                    f'denoising if the image was not actually noisy.')
        if n1 > n0 * 0.85:
            return (f'noise only slightly reduced ({n0:.2f} -> {n1:.2f}); increase strength '
                    f'if more is needed, at the cost of softening real detail')
        return f'noise {n0:.2f} -> {n1:.2f}'
    diff = float((np.abs(b - a).max(axis=2) > 2).mean())
    if diff < 0.001:
        return 'WARNING: this operation changed almost nothing (<0.1% of pixels)'
    return f'{diff:.0%} of pixels changed'


def _rgb(v):
    """A colour argument as an (r, g, b) triple, or a name the package knows.

    The schema asks for three integers, and a model that sends "red" instead would reach
    `int('r')` and fail obscurely. `photo._resolve_colour` answers with the package's own
    table and names the accepted values when it cannot.
    """
    if isinstance(v, str):
        return photo._resolve_colour(v, 'colour argument')
    rgb = tuple(int(x) for x in v)
    # Checked, not clipped: an out-of-range value like [300, -20, 0] would otherwise be
    # accepted and reported as "recoloured 33 % of pixels" toward a hue computed from values
    # no pixel can hold. A model told so can send a real colour; one told "recoloured"
    # cannot know anything went wrong.
    if len(rgb) != 3 or not all(0 <= c <= 255 for c in rgb):
        raise ValueError(f'a colour is three RGB integers 0-255, like [220, 30, 30]; '
                         f'got {list(v)}')
    return rgb


def _empty_segmentation_note(coverages, regions, what, points=False, where=None):
    """Note for a segmentation that found nothing, or None if it found something.

    A box-based object tool can quietly no-op (coverage 0.0), reporting what it intended
    rather than what happened; containment checks can't catch this since an image that
    changed nowhere trivially changed nothing outside its box.
    """
    # A point prompt's coverage is of the whole frame, not a box, so its "found nothing"
    # floor must be the strictest possible: exactly zero.
    if max(coverages) > (0.0 if points else 0.005):
        return None
    # `where` is the point prompt when there is one: with points, `regions` is empty and
    # cannot supply a location on its own.
    if where is None:
        where = tuple(regions[0]) if len(regions) == 1 else tuple(regions)
    else:
        where = tuple(where[0]) if len(where) == 1 else tuple(where)
    return (f'WARNING: nothing was {what}. Segmentation found no object at {where}, '
            f'so the image is unchanged. Check the box with preview_object_mask -- the '
            f'object may not contrast with its surroundings, or the box may be off. Do not '
            f'simply retry this call.')


def _no_object_note(what, args):
    """Same no-op-reporting-success guard as `_empty_segmentation_note`, restated because
    `erase_object`/`scale_object` measure coverage of the whole frame, not the box, so they
    can't share that helper's threshold."""
    where = tuple(args['box']) if args.get('box') else _points_arg(args)
    return (f'WARNING: nothing was {what}. Segmentation found no object at {where}, so '
            f'the image is unchanged. Check it with preview_object_mask -- the object may '
            f'not contrast with its surroundings, or the box may be off. Do not simply '
            f'retry this call.')


def _followed_note(res, boxes):
    """Say it when the object turned out to be bigger than the rectangle it was named with.

    The caller cannot see this in the output otherwise: the edit reached past the box drawn,
    because the segmenter found the object continuing there.
    """
    region, outside = getattr(res, 'region', None), getattr(res, 'outside_frac', 0.0)
    if not region or outside < 0.01:
        return ''
    return (f' | the object continued past the box: {outside:.0%} of it was outside, and '
            f'those pixels were edited too. Region actually touched: {tuple(region)}. '
            f'Pass follow_object=False to cut the object at the box instead.')


def _subject_warning(fraction):
    """Note when segmentation found no plausible subject, else None.

    A 2% floor separates a real subject from a picture with none, with room to spare;
    without it, "blur the background" on a subject-less image would blur everything and
    still verify.
    """
    if fraction < 0.02:
        return (f'WARNING: no clear subject found -- segmentation kept only {fraction:.1%} '
                f'of the frame, so almost the whole image was treated as background. This '
                f'image may have no distinct subject (a texture, a landscape, a flat '
                f'graphic). Use a box-based tool, or a focus box, instead.')
    if fraction > 0.98:
        return (f'WARNING: segmentation kept {fraction:.0%} of the frame as subject, so '
                f'there is effectively no background to separate. Check the result.')
    return None


def _boxes_arg(args):
    """Normalise `box` / `boxes` into a list of boxes, whichever the caller sent."""
    if args.get('boxes'):
        return [tuple(int(v) for v in b) for b in args['boxes']]
    if args.get('box'):
        return [tuple(int(v) for v in args['box'])]
    if _points_arg(args) is not None:
        return []                     # points are the prompt; there is no box to normalise
    raise ValueError('this tool needs a box (or boxes, or points on the object)')


def _points_arg(args):
    """`points` / `point` from a tool call, or None. A point is [x, y]."""
    pts = args.get('points') if args.get('points') is not None else args.get('point')
    if pts is None:
        return None
    if len(pts) == 2 and not isinstance(pts[0], (list, tuple)):
        return [(int(pts[0]), int(pts[1]))]
    return [(int(p[0]), int(p[1])) for p in pts]


# Operations that manage transparency themselves -- their whole output is an alpha channel,
# so the wrapper must not touch it.
ALPHA_AWARE = {'remove_background', 'replace_background', 'extract_object'}

# Geometric operations. PIL carries the mode (including alpha) through these natively, and
# they change the image's size, so a saved alpha channel could not be re-attached anyway.
GEOMETRIC = {'crop', 'rotate', 'flip', 'resize'}


def _prepare(image):
    """Normalise an input mode the operations can actually work on. Returns (image, note).

    Real images arrive in modes this package never saw in testing: a palette PNG, a scanned
    page as 'L', a print asset as 'CMYK', a 16-bit scientific TIFF. Probed directly across
    eight modes: `blur_background` raised "image has wrong mode" on P/1/I;16 and `rotate`
    raised "color must be int" on L/LA/1/I;16, because those two build a PIL operation
    against the input's mode instead of converting first like everything else here does.
    Converting once, at the boundary, fixes the whole class rather than two instances of it.
    """
    if image.mode in ('RGB', 'RGBA'):
        return image, None
    return image.convert('RGB'), f'converted {image.mode} input to RGB'


def dispatch(name, args, image):
    """Execute one tool call. Returns a `ToolResult`, which unpacks as (image, note).

    Never raises on bad input -- an invalid argument comes back as a note so the calling
    model can read the problem and correct itself, instead of the session dying. Real
    programming errors still propagate.

    Transparency is preserved across operations that do not understand it. On an ordinary
    two-step request, `remove_background` then `adjust` would return a cutout silently
    flattened onto black, because `photo.adjust` (like most functions here) starts with
    `img.convert('RGB')` and drops the alpha channel -- and the verdict would still read
    "verified", since brightness had gone up, which is all that check can see. Splitting the
    alpha off before such an operation and re-attaching it after keeps the cutout a cutout.
    """
    working, mode_note = _prepare(image)

    alpha = None
    if working.mode == 'RGBA' and name not in ALPHA_AWARE and name not in GEOMETRIC:
        alpha = working.split()[-1]
        working = working.convert('RGB')

    result = _dispatch(name, args, working)
    if not isinstance(result, ToolResult):
        img, note = result
        # An operation that returned the same object it was given did not edit anything: it
        # either inspected the image (sample_colour) or reported an error. Either way the
        # caller must not record it as a new state.
        result = ToolResult(img, note, mutates=img is not working)

    out = result.image
    if result.mutates:
        # Keep the localisation promise, don't only check it afterwards. `workspace.verify`
        # reports a leak once the edit is already in the history; here it is reverted at the
        # point it happens, against the same region table, and said out loud in the note.
        out, leak = enforce_region(name, args or {}, working, out,
                                   region=(result.data or {}).get('region'))
        if leak:
            result = ToolResult(out, f'{result.note} | {leak}', preview=result.preview,
                                mutates=True, data=result.data)
    if not result.mutates:
        # Hand back the caller's own image, not our converted working copy: an inspection
        # must be a no-op even when the input needed normalising to be inspectable.
        out = image
    elif alpha is not None and out.mode == 'RGB' and out.size == alpha.size:
        out = out.copy()
        out.putalpha(alpha)

    note = result.note if not mode_note else f'{result.note} ({mode_note})'
    return ToolResult(out, note, preview=result.preview, mutates=result.mutates,
                      data=result.data)


def _dispatch(name, args, image):
    args = dict(args or {})
    try:
        # ---- looking tools: never modify the image -------------------------------
        if name == 'show_grid':
            step = args.get('step') or 'auto'
            grid = locate.grid_overlay(image, step=step)
            return ToolResult(image, (
                f'grid overlay shown on the {image.size[0]}x{image.size[1]} image. Read box '
                f'coordinates off the labelled lines; the image itself is unchanged.'),
                preview=grid, mutates=False)

        if name == 'preview_region':
            boxes = [tuple(int(v) for v in b) for b in args['boxes']]
            infos = [locate.describe_box(image, b) for b in boxes]
            preview = locate.draw_boxes(image, boxes, labels=args.get('labels'),
                                        dim_outside=args.get('dim_outside', 0.0))
            parts = []
            for b, i in zip(boxes, infos):
                if 'error' in i:
                    parts.append(f'{b}: {i["error"]}')
                else:
                    parts.append(
                        f'{b}: {i["width"]}x{i["height"]} px, {i["frame_fraction"]:.1%} of '
                        f'the frame, median RGB {i["median_rgb"]}'
                        + (' (CLIPPED to the image bounds)' if i['clipped'] else ''))
            return ToolResult(image, ' | '.join(parts), preview=preview, mutates=False,
                              data={'boxes': boxes, 'regions': infos})

        if name == 'refine_box':
            box = tuple(int(v) for v in args['box'])
            new, info = locate.refine_box(image, box)
            preview = locate.draw_boxes(
                image, [box, new] if tuple(new) != box else [box],
                labels=['your box', 'refined'] if tuple(new) != box else ['your box'])
            return ToolResult(image, info['reason'], preview=preview, mutates=False,
                              data={'box': tuple(new), **info})

        if name == 'zoom':
            crop, used = locate.zoom(image, tuple(args['box']))
            return ToolResult(image, (
                f'showing region {used} enlarged to {crop.size[0]}x{crop.size[1]} '
                f'(the image itself is unchanged)'),
                preview=crop, mutates=False, data={'box': used})

        if name == 'preview_colour_mask':
            space = args.get('space', 'hsv')
            _, mask = photo.isolate_colour(image, _rgb(args['target_rgb']),
                                           hue_tol=args.get('hue_tolerance', 0.07),
                                           space=space,
                                           oklab_tol=args.get('oklab_tolerance', 0.06))
            preview = locate.mask_overlay(image, mask)
            frac = float((mask > 0.5).mean())
            tol = (args.get('hue_tolerance', 0.07) if space == 'hsv'
                   else args.get('oklab_tolerance', 0.06))
            note = f'{frac:.1%} of the image matches that colour in {space} at tolerance ' \
                   f'{tol} (matched pixels stay in colour, everything else is dimmed)'
            if frac < 0.001:
                note += (' -- essentially nothing matched. Sample the real colour from the '
                         'region first, or widen the tolerance.')
            elif frac > 0.6:
                note += (' -- that is most of the image. Narrow the tolerance, or use the '
                         'box-based object tools instead.')
            if space == 'hsv' and frac > 0.25:
                note += (" On a photograph a wide match usually means the HUE WINDOW is "
                         "the problem, not the tolerance: try space='oklab', which "
                         "measures perceptual distance and follows the object through its "
                         "shading.")
            return ToolResult(image, note, preview=preview, mutates=False,
                              data={'matched_fraction': frac})

        if name == 'preview_object_mask':
            regions = _boxes_arg(args)
            mask, coverages = objects.segment_objects(image, regions,
                                                      points=_points_arg(args))
            preview = locate.mask_overlay(image, mask)
            coverage = min(coverages)
            unit = ('the frame' if _points_arg(args) else
                    'the box' if len(regions) == 1 else 'each box')
            note = ('segmentation found ' + ', '.join(f'{c:.1%}' for c in coverages) +
                    f' of {unit} (shown in colour, everything else dimmed)')
            # No near-100%-means-gave-up warning here: see the comment at the
            # recolour_object dispatch below for the measurement showing that no signal
            # separates a correct box-filling mask from a genuine give-up.
            if coverage < 0.05 and not _points_arg(args):
                note += (' -- almost nothing was found. The object may not contrast with '
                         'its surroundings; try a tighter box centred on it.')
            return ToolResult(image, note, preview=preview, mutates=False,
                              data={'coverage': coverage})

        if name == 'isolate_colour':
            space = args.get('space', 'hsv')
            target = _rgb(args['target_rgb'])
            out, mask = photo.isolate_colour(
                image, target,
                hue_tol=args.get('hue_tolerance', 0.07),
                flatten=args.get('flatten', 0.5), space=space,
                oklab_tol=args.get('oklab_tolerance', 0.06))
            return out, (f"isolated {(mask > 0.5).mean():.1%} of the image near "
                         f"{tuple(target)} ({space})")

        if name == 'replace_colour':
            space = args.get('space', 'hsv')
            out, mask = photo.replace_colour(
                image, _rgb(args['from_rgb']), _rgb(args['to_rgb']),
                hue_tol=args.get('hue_tolerance', 0.07), space=space,
                oklab_tol=args.get('oklab_tolerance', 0.06))
            return out, f"recoloured {mask.mean():.1%} of pixels ({space})"

        if name == 'sample_colour':
            c = photo.sample_colour(image, tuple(args['box']))
            return image, f"sampled colour is RGB{c} -- pass this as to_rgb/from_rgb"

        if name == 'blur_background':
            focus = args.get('focus', 'subject')
            if focus == 'box' and args.get('focus_box'):
                focus_arg = tuple(int(v) for v in args['focus_box'])
                out = photo.blur_background(image, focus=focus_arg,
                                            radius=args.get('radius', 14))
                return out, 'blurred background (focus=box)'
            # Segment once, inspect the coverage, and hand the same mask to the blur --
            # rather than segmenting, discarding the mask, and segmenting again inside.
            mask = advanced.subject_mask(image)
            warning = _subject_warning(float((mask > 0.5).mean()))
            if warning:
                return image, warning
            out = photo.blur_background(image, focus=mask,
                                        radius=args.get('radius', 14))
            return out, (f'blurred background (subject covers '
                         f'{float((mask > 0.5).mean()):.0%} of the frame)')

        if name == 'remove_background':
            out = advanced.remove_background(image)
            frac = float((np.asarray(out.split()[-1]) > 127).mean())
            warning = _subject_warning(frac)
            if warning:
                return image, warning
            return out, f'background removed (RGBA); subject covers {frac:.0%} of the frame'

        if name == 'replace_background':
            # Segment once, inspect the coverage, and hand the same cutout to the
            # composite -- rather than segmenting, discarding it, and segmenting again
            # inside (the same fix blur_background's own branch already has, above).
            cutout = advanced.remove_background(image)
            frac = float((np.asarray(cutout.split()[-1]) > 127).mean())
            warning = _subject_warning(frac)
            if warning:
                return image, warning
            return (advanced.replace_background(image, _rgb(args['colour_rgb']),
                                                 cutout=cutout),
                    f'background replaced; subject covers {frac:.0%} of the frame')

        if name == 'remove_object':
            region = args.get('polyline') or args.get('box')
            if region is None:
                return image, 'error: remove_object needs a box or polyline'
            if args.get('polyline'):
                region = [tuple(p) for p in args['polyline']]
            else:
                region = tuple(region)
            return (advanced.remove_object(image, region, radius=args.get('radius', 6)),
                    'object painted out')

        if name == 'adjust':
            out = photo.adjust(image,
                               brightness=args.get('brightness', 1.0),
                               contrast=args.get('contrast', 1.0),
                               saturation=args.get('saturation', 1.0),
                               sharpness=args.get('sharpness', 1.0))
            applied = {k: v for k, v in args.items() if v != 1.0}
            notes = [f'adjusted {applied}']
            for kind in ('saturation', 'brightness', 'contrast'):
                if kind in applied:
                    notes.append(_measure_effect(image, out, kind))
            return out, ' | '.join(notes)

        if name == 'stylise':
            eff = args['effect']
            if eff == 'sepia':
                return photo.sepia(image), 'sepia applied'
            if eff == 'grayscale':
                return photo.grayscale(image), 'converted to grayscale'
            if eff == 'sketch':
                return photo.sketch(image), 'sketch effect applied'
            if eff == 'vignette':
                s = args.get('strength', 0.55)
                return photo.vignette(image, strength=s), f'vignette applied (strength={s})'
            return image, f'error: unknown effect {eff!r}'

        if name == 'denoise':
            out = advanced.denoise(image, strength=args.get('strength', 8))
            return out, _measure_effect(image, out, 'denoise')

        if name == 'auto_white_balance':
            out, warning = advanced.auto_white_balance(image)
            return out, warning or 'white balance applied (shift was small)'

        if name == 'clean_transparent_box':
            # One call, not two, to avoid doubling the cost of the slowest chart operation
            # here by running the whole cleaning pass twice and discarding the first result.
            full, crop = advanced.clean_transparent_box(image, tuple(args['box']))
            return ToolResult(full, 'legend box cleaned', preview=crop,
                              data={'box': tuple(args['box'])})

        if name == 'remap_colormap':
            out, frac, alpha = rebrand.remap_colormap(
                image, tuple(args['box']) if args.get('box') else None,
                source_cmap=args.get('source_cmap', 'viridis'),
                new_colours=tuple(args['new_colours']))
            where = 'in the box' if args.get('box') else 'in the whole figure'
            return out, (f'remapped {frac:.0%} of pixels {where} '
                         f'(detected render alpha {alpha:.2f})')

        if name == 'replace_font':
            rot = [(tuple(r['box']), r.get('angle', 90))
                   for r in args.get('rotated_regions', [])]
            out, log = rebrand.replace_font(
                image, min_confidence=args.get('min_confidence', 55),
                rotated_regions=rot)
            read = ', '.join(repr(t) for t, b, s in log[:8])
            return out, f'redrew {len(log)} text elements; OCR read: {read}'

        if name == 'crop':
            box = tuple(int(v) for v in args['box'])
            W, H = image.size
            x0, y0, x1, y1 = box
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 <= x0 or y1 <= y0:
                return image, (f'error: crop box {box} is empty or outside the '
                               f'{W}x{H} image')
            return image.crop((x0, y0, x1, y1)), f'cropped to {x1-x0}x{y1-y0}'

        if name == 'rotate':
            ang = float(args.get('angle', 0))
            expand = bool(args.get('expand', True))
            # The fill has to match the image's mode. A hardcoded 3-tuple raises
            # "color must be int" on every single-channel input, and on an RGBA image it
            # would paint the exposed corners opaque white -- turning a cutout's
            # transparent surround into a white rectangle, which is the opposite of what
            # someone rotating a cutout wants.
            fill = (0, 0, 0, 0) if image.mode == 'RGBA' else (255, 255, 255)
            out = image.rotate(-ang, expand=expand, fillcolor=fill)
            return out, f'rotated {ang} degrees (size now {out.size[0]}x{out.size[1]})'

        if name == 'flip':
            axis = args.get('axis', 'horizontal')
            if axis == 'horizontal':
                return image.transpose(Image.FLIP_LEFT_RIGHT), 'flipped horizontally'
            if axis == 'vertical':
                return image.transpose(Image.FLIP_TOP_BOTTOM), 'flipped vertically'
            return image, f'error: axis must be horizontal or vertical, got {axis!r}'

        if name == 'resize':
            W, H = image.size
            if args.get('scale'):
                sc = float(args['scale'])
                nw, nh = max(1, int(W * sc)), max(1, int(H * sc))
            else:
                nw = int(args.get('width') or 0)
                nh = int(args.get('height') or 0)
                if nw and not nh:
                    nh = max(1, round(H * nw / W))     # preserve aspect ratio
                elif nh and not nw:
                    nw = max(1, round(W * nh / H))
                elif not nw and not nh:
                    return image, 'error: resize needs scale, or width and/or height'
            return (image.resize((nw, nh), Image.LANCZOS),
                    f'resized {W}x{H} -> {nw}x{nh}')

        if name == 'recolour_object':
            regions = _boxes_arg(args)
            res = objects.recolour_object(
                image, to_rgb=_rgb(args['to_rgb']), boxes=regions,
                preserve_shading=args.get('preserve_shading', True),
                points=_points_arg(args))
            out, coverage = res
            empty = _empty_segmentation_note([coverage], regions, 'recoloured',
                                            points=bool(_points_arg(args)),
                                            where=_points_arg(args))
            if empty:
                return image, empty
            pts = _points_arg(args)
            if pts:
                note = (f'recoloured the object at the point{"s" if len(pts) > 1 else ""} '
                        f'given ({coverage:.1%} of the frame)')
            else:
                note = (f'recoloured {len(regions)} segmented object'
                        f'{"s" if len(regions) > 1 else ""} '
                        f'({coverage:.0%} of box{" on average" if len(regions) > 1 else ""})')
            # No coverage > 0.95 "give-up" warning here: that signal fired on a correct
            # railing mask (0.99) and a correct door mask, while missing most of the boxes
            # that had no distinct object in them at all -- wrong in both directions, and
            # shown to be actively harmful in `evals/agent_loop/`, where a driver cannot
            # tell a bad check from a bad edit, so it obeys, undoes a correct edit, and
            # ships a worse one chasing a warning that should not have fired.
            #
            # Three independent signals were measured against the same 22 hand-labelled
            # boxes (`evals/segmentation_models/degenerate_masks.py`) and none separates
            # a degenerate mask from a correct one:
            #   * mask-derived: area fraction, IoU-with-the-box, solidity, component count.
            #   * image-derived: Canny edge density and Sobel gradient energy in the box's
            #     interior, on the original image, before segmentation runs at all.
            #   * model-derived: SlimSAM's own predicted mask-quality score, the margin
            #     between its best and second-best candidate mask, and mask stability under
            #     small jitters of the box (jittered boxes cost one shared encoder pass,
            #     `_slimsam_binaries`). Score 0.823-0.963 degenerate vs 0.822-0.973 correct,
            #     margin 0.003-0.089 vs 0.008-0.157, stability 0.756-0.931 vs 0.744-0.984 --
            #     full overlap on all three, and a real inversion: the flat white wall with
            #     nothing in it (degenerate) reads a higher score and stability than the
            #     low-contrast shoe (a correct object).
            # A segmenter with nothing to find is not measurably less confident than one
            # looking at a real box-filling object. See RESULTS.md and CONTINUE.md item 5.
            #
            # The coverage is still reported, unconditionally, as a plain number -- that is
            # what the caller actually has to go on.
            return ToolResult(out, note + _followed_note(res, regions),
                              data={'region': getattr(res, 'region', None)})

        if name == 'isolate_object':
            regions = _boxes_arg(args)
            res = objects.isolate_object(
                image, boxes=regions, flatten=args.get('flatten', 0.5),
                points=_points_arg(args))
            out, coverage = res
            empty = _empty_segmentation_note([coverage], regions, 'isolated',
                                            points=bool(_points_arg(args)),
                                            where=_points_arg(args))
            if empty:
                return image, empty
            pts = _points_arg(args)
            if pts:
                note = (f'isolated the object at the point{"s" if len(pts) > 1 else ""} '
                        f'given ({coverage:.1%} of the frame)')
            else:
                note = (f'isolated {len(regions)} segmented object'
                        f'{"s" if len(regions) > 1 else ""} ({coverage:.0%} of box'
                        f'{" on average" if len(regions) > 1 else ""})')
            return ToolResult(out, note + _followed_note(res, regions),
                              data={'region': getattr(res, 'region', None)})

        if name == 'generative_fill':
            from . import generative
            try:
                out, info = generative.fill(
                    image, tuple(args['box']), args['instruction'],
                    pad=float(args.get('pad', 0.6)),
                    mask_to_object=bool(args.get('mask_to_object', False)))
            except RuntimeError as e:
                # No key, no network, no credit, a refusal, an empty return. None of these
                # are programming errors and all of them are things the caller can act on,
                # so they come back as a note like any other bad argument does.
                return image, f'generative_fill did not run: {e}'
            if info['changed_fraction'] < 0.01:
                # The same no-op-reporting-success failure erase_object guards against: the
                # model can hand back what it was given, and a box that changed nowhere
                # trivially changed nothing outside itself.
                return image, ('WARNING: the image model returned the region essentially '
                               'unchanged, so nothing was filled. Rephrase the instruction '
                               'to say what the region should CONTAIN. Do not simply retry.')
            note = (f'region {tuple(int(v) for v in args["box"])} redrawn by '
                    f'{info["model"]} in {info["seconds"]}s'
                    + (f' (${info["cost"]})' if info.get('cost') else '')
                    + f'; {info["changed_fraction"]:.0%} of the box changed')
            if info.get('masked_fraction') is not None:
                note += (f' | masked to the segmented object ({info["masked_fraction"]:.0%} '
                         f'of the box) -- the rest of the box kept its original pixels')
            if info.get('rescaled'):
                note += (f' | the model answered at {info["returned"][0]}x'
                         f'{info["returned"][1]} and was resized back onto the box')
            if info.get('seam') is not None:
                note += (f' | seam {info["seam"]:.3f} (OKLab distance between the fill edge '
                         f'and the background it meets; ~0.02 is invisible, 0.1 is a '
                         f'clearly visible join -- LOOK at it with zoom before trusting it)')
            # The colour correction is reported, never silent: it makes a fill agree with
            # its surroundings, which is not the same as making it right, and a caller
            # judging the result should know the picture was adjusted after the model
            # returned it. See `ringfit`.
            cf = info.get('colour_fit') or {}
            if cf.get('applied'):
                err = cf.get('ring_err') or [None, None]
                note += (f' | colour matched to the surrounding image (gain {cf["gain"]}); '
                         f'on the half of the context ring the fit never saw, the distance '
                         f'to the real colours fell {err[0]} -> {err[1]} (OKLab). The model '
                         f'reproduced that ring at r={cf["r"]}. Seam '
                         f'{cf["seam_before"]:.3f} -> {cf["seam_after"]:.3f}, which barely '
                         f'moves either way -- it is an edge measure and this is a '
                         f'whole-patch correction')
            elif cf.get('refused'):
                note += f' | no colour match: {cf["refused"]}'
            elif cf.get('seam_after') is not None:
                note += (f' | colour match computed and DISCARDED: it moved the seam '
                         f'{cf["seam_before"]:.3f} -> {cf["seam_after"]:.3f}, no improvement')
            return ToolResult(out, note, mutates=True,
                              data={'region': tuple(int(v) for v in args['box']), **info})

        if name == 'erase_object':
            res = objects.erase_object(
                image, tuple(args['box']) if args.get('box') else None,
                radius=args.get('radius', 6), points=_points_arg(args))
            out, frame_frac, warning = res
            if frame_frac <= 0.0001:
                # A no-op that reports success. On the rocket photo, a box around the mast
                # that GrabCut found nothing in returned coverage 0.0, changed not one
                # pixel, and reported "object erased (0.0% of frame) | verified: 0 px
                # changed outside the target region" -- two green statements about an
                # operation that did not happen. The localisation check cannot catch this:
                # an image that changed nowhere trivially changed nothing outside the box.
                return image, _no_object_note('erased', args)
            note = f'object erased ({frame_frac:.1%} of frame)'
            note = f'{note} | {warning}' if warning else note
            return ToolResult(out, note + _followed_note(res, None),
                              data={'region': getattr(res, 'region', None)})

        if name == 'scale_object':
            scale = float(args.get('scale', 1.0))
            res = objects.scale_object(
                image, tuple(args['box']) if args.get('box') else None,
                scale=scale, points=_points_arg(args))
            out, frame_frac, warning = res
            if frame_frac <= 0.0001:
                # The same no-op-reporting-success trap `erase_object` guards against: an
                # empty mask scales nothing, and an image that changed nowhere trivially
                # changed nothing outside its box.
                return image, _no_object_note('scaled', args)
            note = f'object scaled by {scale}x (was {frame_frac:.1%} of frame)'
            note = f'{note} | {warning}' if warning else note
            return ToolResult(out, note + _followed_note(res, None),
                              data={'region': getattr(res, 'region', None)})

        if name == 'extract_object':
            points = _points_arg(args)
            box = tuple(args['box']) if args.get('box') else None
            seg = objects.segment_object(image, box, points=points)
            coverage = seg[1]
            empty = _empty_segmentation_note([coverage], [box or points], 'extracted',
                                            points=bool(points))
            if empty:
                return image, empty
            out = objects.extract_object(image, box, points=points, _seg=seg)
            where = 'the frame' if points else 'the box'
            return out, f'object extracted as an RGBA cutout ({coverage:.0%} of {where})'

        if name == 'add_text':
            from PIL import ImageDraw, ImageFont
            out = image.convert('RGB').copy()
            draw = ImageDraw.Draw(out)
            size = int(args.get('size', 24))
            try:
                fnt = ImageFont.truetype(rebrand.LIBERATION_SANS, size)
            except OSError:
                fnt = ImageFont.load_default()
            xy = tuple(int(v) for v in args.get('position', [10, 10]))
            colour = _rgb(args.get('colour_rgb', [0, 0, 0]))
            draw.text(xy, str(args['text']), font=fnt, fill=colour)
            return out, f'drew {args["text"]!r} at {xy} size {size}'

        if name in ('undo', 'reset'):
            return image, (
                f'error: {name!r} acts on the edit history, which a single dispatch call '
                f'does not have. Drive the edits through figsurgeon.ImageWorkspace, whose '
                f'apply() takes the same tool calls and handles this one.')

        return image, f'error: unknown tool {name!r}'

    except KeyError as e:
        # A KeyError here is ambiguous: it may be a missing argument, or it may come from
        # deeper in a library (matplotlib raises KeyError for an unknown colormap name).
        # Reporting the latter as "missing required argument" misleads the calling model
        # into re-sending the same arguments. Distinguish them by checking whether the key
        # is actually one of ours.
        key = str(e).strip('"\'')
        if key in args or key.replace("'", '') in args:
            return image, f'error: invalid value for argument {key!r} in {name}'
        if any(key == k for k in ('target_rgb', 'from_rgb', 'to_rgb', 'box', 'effect',
                                  'colour_rgb', 'new_colours', 'text', 'angle', 'axis')):
            return image, f'error: missing required argument {key!r} for {name}'
        return image, f'error running {name}: {e}'
    except (ValueError, TypeError) as e:
        return image, f'error running {name}: {e}'
    except (ImportError, OSError) as e:
        # A missing dependency is not a programming error, so it is reported rather than
        # left to end the session. Several tools here need something the environment may
        # not have: `rembg`'s model, OpenCV, and above all the tesseract binary, which pip
        # cannot install at all -- on the robustness sweep, `replace_font` raised
        # TesseractNotFoundError on all 13 inputs. Nothing the calling model does can fix
        # that mid-session, but it can route around it if it is told, rather than
        # disconnected. Genuine bugs (AttributeError, IndexError, anything unanticipated)
        # still propagate, which is the distinction that matters.
        return image, (f'error: {name} needs something this environment does not have '
                       f'({type(e).__name__}: {e}). This will not be fixed by different '
                       f'arguments -- use another approach, or install the dependency.')


def run_sequence(image, calls):
    """Run several tool calls in order, threading the image through. Returns (image, notes)."""
    notes = []
    img = image
    for name, args in calls:
        img, note = dispatch(name, args, img)
        notes.append(f'{name}: {note}')
    return img, notes
