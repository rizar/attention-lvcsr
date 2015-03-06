import time
import numpy
import theano
from theano import tensor
from numpy.testing import assert_allclose

from blocks.bricks import Identity

from lvsr.attention import ShiftPredictor

floatX = theano.config.floatX


def test_shift_predictor():
    predictor = ShiftPredictor(
        state_names=["states"], state_dims=dict(states=100),
        attended_dim=100, predictor=Identity(),
        max_left=2, max_right=3)

    weights_var = tensor.matrix()
    shifts_var = tensor.matrix()
    func = theano.function(
        [weights_var, shifts_var],
        [predictor.convolve_weights(weights_var, shifts_var)])

    weights = numpy.array(
        [[0, 0, 1, 0],
         [0, 1, 0, 0],
         [1, 0, 0, 0]], dtype=floatX)
    shifts_probs = numpy.array(
        [[0, 0.1, 0.2, 0.5, 0.15, 0.05],
         [0, 0.1, 0.5, 0.2, 0.15, 0.05],
         [0, 0, 0, 0.8, 0.15, 0.05]], dtype=floatX)
    result, = func(weights, shifts_probs)
    assert_allclose(
        result,
        [[ 0., 0.1, 0.2, 0.5 ],
         [ 0.1, 0.5, 0.2, 0.15],
         [ 0.,  0.8, 0.15, 0.05]])



    weights_shared = theano.shared(numpy.random.rand(10, 500).astype(floatX))
    shifts_shared = theano.shared(numpy.random.rand(10, 110).astype(floatX))
    func2 = theano.function(
        [], [predictor.convolve_weights(weights_shared, shifts_shared)])
    before = time.time()
    for i in range(100):
        func2()
    print(time.time() - before)
