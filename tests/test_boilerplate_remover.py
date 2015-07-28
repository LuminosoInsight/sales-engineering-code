from se_code.boilerplate_remover import SpaceSplittingReader
from nose.tools import eq_

# NB: This does not actually properly test the boilerplate remover; for now it
# just holds the tests for the moved SpaceSplittingReader

def test_space_splitting_reader():
    # Simple test for the SpaceSplittingReader
    text = ' a\n bc ß'
    expected = [
        ('a', None, (1, 2)),
        ('^^^', None, (2, 3)),  # newlines are normalized to this placeholder
        ('bc', None, (4, 6)),   # multi-character words stay together
        ('ss', None, (7, 8))    # we casefold instead of lowercasing
    ]
    eq_(SpaceSplittingReader().text_to_token_triples(text), expected)

