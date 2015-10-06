import numpy
from numpy.testing import assert_equal
from lvsr.error_rate import (
    _edit_distance_matrix,
    optimistic_error_matrix)


def test_edit_distance_matrix():
    matrix = _edit_distance_matrix('abdce', 'abcd')
    should_be = numpy.array(
        [[0, 1, 2, 3, 4],
         [1, 0, 1, 2, 3],
         [2, 1, 0, 1, 2],
         [3, 2, 1, 1, 1],
         [4, 3, 2, 1, 2],
         [5, 4, 3, 2, 2]])
    assert_equal(matrix, should_be)


def test_optimistic_error_matrix():
    matrix = optimistic_error_matrix('abdc', 'abcdxcxx', 'abcdx')
    should_be = numpy.array(
       [[ 0, -1, -1, -1, -1],
        [-1,  0, -1, -1, -1],
        [-1, -1, -1,  0, -1],
        [-1, -1,  0,  0, -1],
        [-1, -1,  0, -1, -1],
        [-1, -1,  0, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]])
    assert_equal(matrix, should_be)

