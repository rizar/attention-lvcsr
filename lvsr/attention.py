import theano
from theano import tensor

from blocks.bricks import Initializable
from blocks.bricks.base import lazy, application
from blocks.bricks.attention import GenericSequenceAttention
from blocks.utils import put_hook, ipdb_breakpoint


class ShiftPredictor(GenericSequenceAttention, Initializable):

    @lazy
    def __init__(self, predictor, max_left, max_right, **kwargs):
        super(GenericSequenceAttention, self).__init__(**kwargs)
        assert len(self.state_names) == 1
        self.max_left = max_left
        self.max_right = max_right
        self.predictor = predictor

        self.span = self.max_right + self.max_left + 1
        self.children = [self.predictor]

    def _push_allocation_config(self):
        self.predictor.input_dim, = self.state_dims.values()
        self.predictor.output_dim = self.span

    @application
    def convolve_weights(self, weights, shift_probs):
        # weights: (batch_size, length)
        # shift_probs: (batch_size, span)
        batch_size = weights.shape[0]
        weights = tensor.concatenate([
            tensor.zeros((batch_size, self.max_right)),
            weights,
            tensor.zeros((batch_size, self.max_left))],
            axis=1)
        image = weights.dimshuffle(0, 'x', 'x', 1)
        filters = shift_probs.dimshuffle(0, 'x', 'x', 1)
        result = tensor.nnet.conv2d(image, filters)
        # result: (batch_size, batch_size, 1, length)
        return tensor.diagonal(
            result[:, :, 0, :].dimshuffle(2, 0, 1), axis1=1, axis2=2).T

    @application
    def compute_energies(self, states, previous_weights):
        # previous_weights = put_hook(previous_weights, ipdb_breakpoint)
        state, = states.values()
        length = previous_weights.shape[1]

        shift_energies = self.predictor.apply(state)
        positions = tensor.arange(length)
        expected_positions = (
            (previous_weights * positions).sum(axis=1).astype('int64'))
        zero_row = tensor.zeros((length + self.span,))
        def fun(expected, shift_energies):
            result = tensor.set_subtensor(
                zero_row[expected:expected + self.span],
                shift_energies)
            return result
        energies, _ = theano.scan(fun,
            sequences=[expected_positions, shift_energies])
        return energies[:, self.max_left:self.max_left + length]

    @application(outputs=['weighted_averages', 'weights'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, weights=None,
                      **states):
        if not weights:
            raise ValueError
        energies = self.compute_energies(states, weights).T
        new_weights = self.compute_weights(energies, attended_mask)
        weighted_averages = self.compute_weighted_averages(new_weights,
                                                           attended)
        return weighted_averages, new_weights.T

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        return (['attended', 'preprocessed_attended',
                 'attended_mask', 'weights'] +
                self.state_names)

    @application
    def initial_glimpses(self, name, batch_size, attended):
        if name == "weighted_averages":
            return tensor.zeros((batch_size, self.attended_dim))
        elif name == "weights":
            return tensor.concatenate([
                 tensor.ones((batch_size, 1)),
                 tensor.zeros((batch_size, attended.shape[0] - 1))],
                 axis=1)
        raise ValueError("Unknown glimpse name {}".format(name))

    def get_dim(self, name):
        if name in ['weighted_averages']:
            return self.attended_dim
        if name in ['weights']:
            return 0
        return super(ShiftPredictor, self).get_dim(name)
