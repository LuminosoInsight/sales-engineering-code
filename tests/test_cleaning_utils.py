from se_code.cleaning_utils import RepeatShortener
from nose.tools import eq_

def test_repeat_shortener():
    # Test the default repeat count (4)
    rs = RepeatShortener()
    eq_(rs('panda'), 'panda')
    eq_(rs('looool'), 'looool')
    eq_(rs('loooooloooool heheheheheh'), 'loooolooool heheheheh')

    # Spaces are not considered
    eq_(rs('na na na na na na na na BATMAN'), 'na na na na na na na na BATMAN')

    # Test a custom repeat count
    rs = RepeatShortener(2)
    eq_(rs('nananana, hey heeeeey, goodbye'), 'nana, hey heey, goodbye')
