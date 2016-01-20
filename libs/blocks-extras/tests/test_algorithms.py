from collections import OrderedDict

import theano
from numpy.testing import assert_allclose
from theano import tensor

from blocks.algorithms import BasicMomentum
from blocks_extras.algorithms import BasicNesterovMomentum, NesterovMomentum
from blocks.utils import shared_floatx


def test_basic_nesterov_momentum():
    a = shared_floatx([3, 4])
    cost = (a ** 2).sum()
    steps, updates = BasicNesterovMomentum(0.5).compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)
    steps_classic, updates_classic = BasicMomentum(0.5).compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f_classic = theano.function([], [steps_classic[a]],
                                updates=updates_classic)
    f_classic()  # One call for the "peek ahead" of the Nesterov momentum.
    assert_allclose(f()[0], f_classic()[0])
    assert_allclose(f()[0], f_classic()[0])
    assert_allclose(f()[0], f_classic()[0])


def test_nesterov_momentum():
    a = shared_floatx([3, 4])
    cost = (a ** 2).sum()
    steps, updates = NesterovMomentum(0.1, 0.5).compute_steps(
        OrderedDict([(a, tensor.grad(cost, a))]))
    f = theano.function([], [steps[a]], updates=updates)
    assert_allclose(f()[0], [0.9, 1.2])
    assert_allclose(f()[0], [1.05, 1.4])
    assert_allclose(f()[0], [1.125, 1.5])
