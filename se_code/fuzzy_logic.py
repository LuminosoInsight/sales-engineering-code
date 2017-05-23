"""
Smooth functions that act like AND, OR, and NOT over values in the range [0, 1].

These functions can be used to measure topic matches with conjuctions or
disjunctions of topics, for example.

To be precise, these functions implement the Hadamard T_0 norm. The advantage
of using these functions over `max` and `min` is that they provide a better
ranking. `min(a, b)` doesn't increase when the higher value increases, but the
Hadamard product (fuzzy AND) does slightly increase.
"""
import numpy as np


def clamp(val):
    """
    Restrict a value to the range [0, 1].
    """
    return max(0., min(1., val))


def tanh_clamp(val):
    """
    Use the tanh function with a hard floor at 0 to restrict a value to the range [0, 1].
    """
    return max(0, np.tanh(val))


def _hadamard_product(v1, v2):
    """
    Compute the 'fuzzy AND' (Hadamard product) of two values in the range
    [0, 1].

    This function doesn't check the range of the values, which is why
    it's private.
    """
    try:
        return v1 * v2 / (v1 + v2 - v1 * v2)
    except ZeroDivisionError:
        return 0


def fuzzy_and(vals):
    """
    Compute the 'fuzzy AND' (Hadamard product) of any number of values,
    which will be constrained to the range [0, 1].
    """
    if len(vals) == 0:
        return 1.
    result = vals[0]
    for val in vals[1:]:
        result = _hadamard_product(result, val)
    return result


def fuzzy_not(val):
    """
    Compute the inverse of a value, which in this logic is just 1 minus the
    value.
    """
    return 1. - val


def fuzzy_or(vals):
    """
    Compute the 'fuzzy OR' (Hadamard co-product) of any number of values,
    which will be constrained to the range [0, 1].

    This is implemented using DeMorgan's law, by inverting all the values,
    taking the fuzzy AND, and inverting the result.
    """
    if len(vals) == 0:
        return 0.
    result = fuzzy_not(vals[0])
    for val in vals[1:]:
        result = _hadamard_product(result, fuzzy_not(val))
    return fuzzy_not(result)


def max_or(vals):
    """
    Compute the max of any number of values.
    """
    if len(vals) == 0:
        return 0.
    return max(vals)


def min_and(vals):
    """
    Compute the min of any number of values.
    """
    if len(vals) == 0:
        return 0
    return min(vals)