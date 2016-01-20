import numpy
from numpy.testing import assert_equal
import theano
from theano import tensor
from blocks_extras.bricks import FixedPermutation


def test_fixed_permutation():
    x = tensor.matrix()
    x_val = numpy.arange(15, dtype=theano.config.floatX).reshape((5, 3))
    perm = FixedPermutation([2, 0, 1])
    y = perm.apply(x)
    y_val = y.eval({x: x_val})
    assert_equal(x_val[:, [2, 0, 1]], y_val)
    perm = FixedPermutation([2, 0, 1], dot=False)
    y = perm.apply(x)
    assert_equal(x_val[:, [2, 0, 1]], y_val)
