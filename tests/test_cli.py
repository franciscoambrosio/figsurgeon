"""End-to-end tests for the `figsurgeon` command line, on a small generated image."""
import os

import pytest
from PIL import Image

from figsurgeon import cli


@pytest.fixture
def image_path(tmp_path):
    img = Image.new('RGB', (64, 64), (200, 40, 40))
    path = str(tmp_path / 'source.png')
    img.save(path)
    return path


def test_doctor_command_returns_zero(capsys):
    assert cli.main(['doctor']) == 0
    assert 'figsurgeon doctor' in capsys.readouterr().out


def test_edit_writes_default_output_and_prints_verdict(image_path, capsys):
    rc = cli.main(['edit', image_path, 'adjust the saturation to 0'])
    err = capsys.readouterr().err
    expected_out = image_path.replace('.png', '_edited.png')
    if rc == 0:
        assert os.path.exists(expected_out)
        assert 'verified' in err or 'check inconclusive' in err or 'CHECK FAILED' in err
    else:
        assert err.strip()          # must fail loudly, not silently, if unroutable


def test_edit_honours_output_flag(image_path, tmp_path):
    out_path = str(tmp_path / 'custom_out.png')
    cli.main(['edit', image_path, 'go greyscale', '-o', out_path])
    assert os.path.exists(out_path)


def test_edit_unroutable_instruction_writes_nothing(image_path, tmp_path, capsys):
    out_path = str(tmp_path / 'should_not_exist.png')
    rc = cli.main(['edit', image_path, 'zzz not a real instruction zzz', '-o', out_path])
    assert rc == 1
    assert not os.path.exists(out_path)
    assert capsys.readouterr().err.strip()


def test_a_cutout_saved_as_jpeg_is_flattened_not_a_traceback(tmp_path, capsys):
    """A cutout is RGBA, which JPEG cannot hold, so `-o out.jpg` must flatten onto white
    and say so rather than raise after the whole edit has already run."""
    from figsurgeon import workspace
    src = str(tmp_path / 'src.png')
    Image.new('RGB', (64, 64), (200, 40, 40)).save(src)
    cutout = Image.new('RGBA', (64, 64), (200, 40, 40, 0))
    fake = workspace.ToolResult(cutout, 'background removed | verified: fine', mutates=True)
    orig = workspace.ImageWorkspace.ask
    workspace.ImageWorkspace.ask = lambda self, text: fake
    try:
        out = str(tmp_path / 'out.jpg')
        assert cli.main(['edit', src, 'remove the background', '-o', out]) == 0
    finally:
        workspace.ImageWorkspace.ask = orig
    assert Image.open(out).getpixel((5, 5)) == (255, 255, 255) or \
        min(Image.open(out).getpixel((5, 5))) > 245
    assert 'white' in capsys.readouterr().err


def test_a_missing_input_file_is_a_message_not_a_traceback(tmp_path, capsys):
    assert cli.main(['edit', str(tmp_path / 'nope.png'), 'make it black and white']) == 1
    assert 'nope.png' in capsys.readouterr().err


def test_a_failed_check_is_a_nonzero_exit(tmp_path, capsys):
    """The edit is still written -- the caller may want it -- but a script checking `$?`
    must be able to tell CHECK FAILED from verified."""
    from figsurgeon import workspace
    src = str(tmp_path / 'src.png')
    Image.new('RGB', (64, 64), (200, 40, 40)).save(src)
    fake = workspace.ToolResult(Image.new('RGB', (64, 64)), 'x | CHECK FAILED: nope',
                                mutates=True, data={'verified': False})
    orig = workspace.ImageWorkspace.ask
    workspace.ImageWorkspace.ask = lambda self, text: fake
    try:
        rc = cli.main(['edit', src, 'whatever', '-o', str(tmp_path / 'o.png')])
    finally:
        workspace.ImageWorkspace.ask = orig
    assert rc == 2
    assert os.path.exists(str(tmp_path / 'o.png'))
