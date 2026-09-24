"""The look-then-edit loop, as four panels: what the model sees at each step.

Regenerates `docs/images/looking_demo_grid.png`. The four panels are the actual outputs of the four
tools, at the sizes and settings a model would get them, not mockups.
"""
import os

from PIL import Image, ImageDraw
from skimage import data

from figsurgeon import ImageWorkspace
from figsurgeon.locate import _font

PANEL_W = 460
GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'images',
                    'looking_demo_grid.png')


def panel(img, title, caption):
    thumb = img.convert('RGB').copy()
    thumb.thumbnail((PANEL_W, PANEL_W), Image.LANCZOS)
    cell = Image.new('RGB', (PANEL_W, thumb.size[1] + 72), (250, 250, 250))
    cell.paste(thumb, ((PANEL_W - thumb.size[0]) // 2, 46))
    d = ImageDraw.Draw(cell)
    d.text((10, 8), title, font=_font(19), fill=(10, 10, 10))
    d.text((10, 28), caption[:78], font=_font(13), fill=(90, 90, 90))
    return cell


def main():
    img = Image.fromarray(data.chelsea())
    ws = ImageWorkspace(img)

    # 1. A box estimated from the plain image -- both eyes, guessed by eye.
    guessed = [150, 90, 215, 140]
    grid = ws.apply('show_grid', {})
    # 2. Snap the guess onto the object.
    refined = ws.apply('refine_box', {'box': guessed})
    box = list(refined.data['box'])
    # 3. See exactly which pixels the edit will touch.
    mask = ws.apply('preview_object_mask', {'boxes': [box, [288, 100, 352, 158]]})
    # 4. Only now, edit.
    edit = ws.apply('isolate_object', {'boxes': [box, [288, 100, 352, 158]],
                                       'flatten': 0.6})

    panels = [
        panel(grid.preview, '1. show_grid',
              'read box coordinates off the labels, do not estimate them'),
        panel(refined.preview, '2. refine_box',
              f'guessed {tuple(guessed)} snapped onto the object -> {tuple(box)}'),
        panel(mask.preview, '3. preview_object_mask',
              mask.note[:78]),
        panel(edit.image, '4. isolate_object',
              edit.note.split('|')[-1].strip()[:78]),
    ]
    h = max(p.size[1] for p in panels)
    sheet = Image.new('RGB', (PANEL_W * 2, h * 2), (250, 250, 250))
    for i, p in enumerate(panels):
        sheet.paste(p, ((i % 2) * PANEL_W, (i // 2) * h))
    sheet.save(GRID)

    print('guessed box :', guessed)
    print('refined box :', box, '|', refined.note)
    print('mask        :', mask.note)
    print('edit        :', edit.note)
    print('\nnone of steps 1-3 changed the image:',
          len(ws.history) == 1, f'({len(ws.history)} edit recorded)')
    print('wrote docs/images/looking_demo_grid.png')


if __name__ == '__main__':
    main()
