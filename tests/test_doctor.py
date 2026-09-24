"""`figsurgeon doctor` reports the true state of each optional piece, and the install line
when one is missing."""
import os
from unittest import mock

from figsurgeon import doctor


def test_checks_returns_a_row_per_optional_piece():
    rows = doctor.checks()
    labels = [label for label, ok, cost in rows]
    assert any('grounding' in l for l in labels)
    assert any('advanced' in l for l in labels)
    assert any('rebrand' in l for l in labels)
    assert any('LaMa' in l for l in labels)
    assert any('OPENROUTER_API_KEY' in l for l in labels)


def test_missing_piece_names_its_install_line():
    with mock.patch('figsurgeon.grounding.available', return_value=False):
        rows = doctor.checks()
    row = next(r for r in rows if 'grounding' in r[0])
    label, ok, cost = row
    assert ok is False
    assert 'pip install "figsurgeon[grounding]"' in cost


def test_present_piece_reports_ok():
    with mock.patch('figsurgeon.grounding.available', return_value=True):
        rows = doctor.checks()
    row = next(r for r in rows if 'grounding' in r[0])
    assert row[1] is True


def test_openrouter_key_check_reads_the_environment():
    with mock.patch.dict(os.environ, {'OPENROUTER_API_KEY': 'x'}):
        rows = doctor.checks()
    row = next(r for r in rows if 'OPENROUTER_API_KEY' in r[0])
    assert row[1] is True

    with mock.patch.dict(os.environ, {}, clear=True):
        rows = doctor.checks()
    row = next(r for r in rows if 'OPENROUTER_API_KEY' in r[0])
    assert row[1] is False


def test_main_runs_and_returns_zero(capsys):
    assert doctor.main() == 0
    out = capsys.readouterr().out
    assert 'figsurgeon doctor' in out
