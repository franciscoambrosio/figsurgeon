"""`figsurgeon` command line: `doctor` and `edit`.

Deliberately small. Everything else this package can do is reached through the tool schemas
in `tools.py` (see the MCP server) or the Python API directly -- this is just enough to get
a first edit done and verified without writing any code.
"""
import argparse
import os
import sys

from PIL import Image


def _default_output(image_path):
    stem, _ = os.path.splitext(image_path)
    return f'{stem}_edited.png'


def _cmd_doctor(args):
    from .doctor import main as doctor_main
    return doctor_main()


def _cmd_edit(args):
    from .workspace import ImageWorkspace

    try:
        img = Image.open(args.image)
    except OSError as e:
        print(f'cannot open {args.image}: {e}', file=sys.stderr)
        return 1
    workspace = ImageWorkspace(img)
    try:
        result = workspace.ask(args.instruction)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(result.note, file=sys.stderr)
    if not result.mutates:
        print('nothing written: the instruction did not change the image', file=sys.stderr)
        return 1

    out_path = args.output or _default_output(args.image)
    out = result.image
    if out.mode in ('RGBA', 'LA', 'P') and os.path.splitext(out_path)[1].lower() in (
            '.jpg', '.jpeg'):
        # JPEG has no alpha channel. Flatten onto white rather than let the save fail after
        # the whole edit has already run.
        flat = Image.new('RGB', out.size, (255, 255, 255))
        flat.paste(out.convert('RGBA'), mask=out.convert('RGBA').split()[-1])
        out = flat
        print('JPEG has no transparency: the transparent areas were saved as white. '
              'Use a .png output to keep them.', file=sys.stderr)
    try:
        out.save(out_path)
    except (OSError, ValueError) as e:
        print(f'cannot write {out_path}: {e}', file=sys.stderr)
        return 1
    print(f'wrote {out_path}', file=sys.stderr)
    # The edit is written either way, but a script checking the exit code must be able to
    # tell a failed verification from a verified one.
    return 2 if result.data.get('verified') is False else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='figsurgeon')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('doctor', help='show which optional dependencies are present')

    p_edit = sub.add_parser('edit', help='apply one plain-language instruction to an image',
                            epilog='exit status: 0 written, 1 nothing written, '
                                   '2 written but its verification check failed')
    p_edit.add_argument('image', help='path to the input image')
    p_edit.add_argument('instruction', help='what to do, e.g. "make the sky blue"')
    p_edit.add_argument('-o', '--output', help='output path (default: <stem>_edited.png)')

    args = parser.parse_args(argv)
    if args.command == 'doctor':
        return _cmd_doctor(args)
    if args.command == 'edit':
        return _cmd_edit(args)
    parser.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
