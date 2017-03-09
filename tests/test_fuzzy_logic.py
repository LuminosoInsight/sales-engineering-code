from nose.tools import assert_almost_equal
from se_code.fuzzy_logic import fuzzy_and, fuzzy_or


def test_fuzzy_and():
    # The identity is 1
    assert_almost_equal(fuzzy_and(), 1.0)
    assert_almost_equal(fuzzy_and(1.0, 0.5), 0.5)

    # ANDing one value leaves it alone
    assert_almost_equal(fuzzy_and(0.8), 0.8)

    # ...unless it's out of range
    assert_almost_equal(fuzzy_and(-0.2), 0.0)

    # The result of ANDing with 0 or a negative number is 0
    assert_almost_equal(fuzzy_and(0.5, 0.), 0.)
    assert_almost_equal(fuzzy_and(0.8, -0.2), 0.)

    # ANDing two middling values results in a lower value
    assert_almost_equal(fuzzy_and(0.5, 0.5), 1/3)
    assert_almost_equal(fuzzy_and(0.75, 0.75), 0.6)

    # You can take the AND of three things
    assert_almost_equal(fuzzy_and(0.5, 0.5, 0.5), 0.25)

    # Increasing one of the values results in a slightly higher number
    assert_almost_equal(fuzzy_and(0.75, 0.5, 0.5), 0.3)

    # Decreasing one of the values decreases the result a lot
    assert_almost_equal(fuzzy_and(0.25, 0.5, 0.5), 1/6)


def test_fuzzy_or():
    # The identity is 0
    assert_almost_equal(fuzzy_or(), 0.0)
    assert_almost_equal(fuzzy_or(0.0, 0.5), 0.5)

    # ORing one value leaves it alone
    assert_almost_equal(fuzzy_or(0.2), 0.2)

    # ...unless it's out of range
    assert_almost_equal(fuzzy_or(1.2), 1.0)

    # The result of ORing with 1 is 1
    assert_almost_equal(fuzzy_or(0.5, 1.), 1.)

    # ORing two middling values results in a higher value
    assert_almost_equal(fuzzy_or(0.5, 0.5), 2/3)
    assert_almost_equal(fuzzy_or(0.25, 0.25), 0.4)

    # You can take the OR of three things
    assert_almost_equal(fuzzy_or(0.5, 0.5, 0.5), 0.75)

    # Decreasing one of the values results in a slightly lower number
    assert_almost_equal(fuzzy_or(0.25, 0.5, 0.5), 0.7)

    # Increasing one of the values increases the result a lot
    assert_almost_equal(fuzzy_or(0.75, 0.5, 0.5), 5/6)

