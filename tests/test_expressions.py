import numpy
from numpy.testing import assert_allclose
from theano import tensor

from lvsr.expressions import pad_to_a_multiple

def test_pad_to_a_multiple():
    a = numpy.array([[1, 2], [3, 4], [5, 6]])
    b = numpy.vstack([a, [[0, 0]]])
    assert_allclose(
        pad_to_a_multiple(tensor.as_tensor_variable(a), 2, 0).eval(), b)
