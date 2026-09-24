"""Routing a sentence to an operation, and refusing when it should.

`photo_describe.parse` is a keyword matcher covering part of the catalogue; `intent.py`
routes the rest by meaning against `figsurgeon/tools.md`. Scored in `evals/text_coverage.py`;
these tests lock behaviours, not the score.
"""
import pytest

from figsurgeon import intent

needs_model = pytest.mark.skipif(not intent.available(),
                                 reason='needs figsurgeon[grounding] (torch)')


def test_the_catalogue_is_the_source_of_the_examples():
    """Examples belong to the operations in tools.md, not hard-coded into the router."""
    cat = intent.catalogue()
    assert len(cat) >= 25, f'only {len(cat)} operations in the catalogue'
    for name, entry in cat.items():
        assert entry['does'], f'{name} has no outcome line'
        assert len(entry['examples']) >= 2, f'{name} has too few examples'


@needs_model
def test_the_operations_the_keyword_parser_could_not_reach():
    """Each sentence raises `Could not parse instruction` in the keyword parser but names an
    operation the package performs; intent.resolve must still find it."""
    from figsurgeon.photo_describe import parse
    for sentence, expected in (('erase the bird', 'erase_object'),
                               ('cut the badge out on its own', 'extract_object'),
                               ('clean the bleed through behind the legend',
                                'clean_transparent_box')):
        with pytest.raises(ValueError):
            parse(sentence)
        got = intent.resolve(sentence)
        assert got and got[0] == expected, f'{sentence!r} -> {got}'


@needs_model
def test_an_object_edit_with_nothing_to_edit_is_refused():
    """An object operation must not be returned with no object filled in: "it" is not a
    thing `erase_object` can act on, even if the sentence routes somewhere else instead."""
    got = intent.resolve('just erase it already')
    assert got is None or got[0] not in {'erase_object', 'extract_object',
                                         'isolate_object', 'recolour_object'}, got
    assert intent.resolve('erase the lamp post')[1]['subject'] == 'the lamp post'


@needs_model
def test_what_survives_decides_between_the_object_operations():
    """Erase, extract and isolate share the same request shape; what separates them is
    whether the named thing or the rest of the scene survives."""
    cases = {'take the sign out of the shot': 'erase_object',
             'cut out just the bicycle': 'extract_object',
             'keep the cat in colour and grey the rest': 'isolate_object',
             'repaint the guitar blue': 'recolour_object'}
    for sentence, expected in cases.items():
        got = intent.resolve(sentence)
        assert got and got[0] == expected, f'{sentence!r} -> {got}'


@needs_model
def test_arguments_are_read_from_the_sentence_not_invented():
    name, args = intent.resolve('re-theme the viridis colormap to #00407A and #52BDEC')
    assert name == 'remap_colormap'
    assert args['source_cmap'] == 'viridis'
    assert args['new_colours'] == ['#00407A', '#52BDEC']

    # "our brand blues" names no colour readable from the sentence; must not be invented.
    name, args = intent.resolve('re-theme the viridis colormap to our brand blues')
    assert name == 'remap_colormap' and 'new_colours' not in args


@needs_model
def test_the_subject_keeps_the_words_the_caller_used():
    """The subject keeps its article: the grounding model is prompted with this exact
    phrase, and "the flowers" finds a different region than "flowers"."""
    got = intent.resolve('remove the person on the left')
    assert got and got[1]['subject'] == 'the person on the left', got


@needs_model
def test_out_of_scope_requests_are_refused():
    """These name things this package does not do; resolve must refuse, not misroute."""
    for sentence in ('add a hat to the dog', 'turn this into a watercolour painting',
                     'fix the perspective', 'swap the two people around'):
        assert intent.resolve(sentence) is None, sentence


def test_the_colormap_named_is_the_one_found_whatever_the_hash_seed():
    """The colormap is picked by iterating a set of words, so the result must not depend on
    PYTHONHASHSEED, and must not confuse the source colormap with a registry name used to
    describe the new colours."""
    import os
    import subprocess
    import sys
    code = ('from figsurgeon import intent; '
            'print(intent.colormap("re-theme the viridis colormap to cool ocean blues"), '
            'intent.colormap("swap the jet colormap for our brand blues"), '
            'intent.colormap("make the magma plot use our colours"))')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for seed in ('1', '2', '3', '4', '5', '6'):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=root)
        out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                             text=True, check=True).stdout.split()
        assert out == ['viridis', 'jet', 'magma'], f'seed {seed}: {out}'


def test_an_ordinary_word_is_not_read_as_a_colormap():
    """Registry names that are also ordinary words (Blues, ocean, cool...) must not be read
    as a colormap unless the sentence says so."""
    assert intent.colormap('make the chart cool blues and greens') is None
    assert intent.colormap('swap the Blues colormap for our brand colours') == 'Blues'
