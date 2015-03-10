import theano
from theano import tensor

from blocks.bricks import Initializable
from blocks.bricks.base import lazy, application
from blocks.bricks.attention import (
    GenericSequenceAttention, SequenceContentAttention)
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
    def compute_energies(self, states, previous_weights):
        """Compute energies.

        Parameters
        ----------
        states : dict
        previous_weights : Theano variable
            Weights from the previous step, (batch_size, attended_len).

        """
        state, = states.values()
        batch_size = previous_weights.shape[0]
        length = previous_weights.shape[1]

        shift_energies = self.predictor.apply(state)
        positions = tensor.arange(length)
        # Positions are broadcasted along the first dimension
        expected_positions = ((previous_weights * positions)
                              .sum(axis=1).astype('int64'))
        zero_row = tensor.zeros((length + self.span,))
        def fun(expected, shift_energies, zero_row_):
            return tensor.set_subtensor(
                zero_row_[expected:expected + self.span],
                shift_energies)
        energies, _ = theano.scan(fun,
            sequences=[expected_positions, shift_energies],
            non_sequences=[zero_row],
            n_steps=batch_size)
        return energies[:, self.max_left:self.max_left + length]

    @application(outputs=['weighted_averages', 'weights'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, weights=None,
                      **states):
        if not weights:
            raise ValueError
        energies = self.compute_energies(states, weights).T
        # Energies become (attended_len, batch_size) as required
        # by inherited methods.
        new_weights = self.compute_weights(energies, attended_mask)
        weighted_averages = self.compute_weighted_averages(new_weights,
                                                           attended)
        # Weights are transposed back to (batch_size, attended_len)
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


class HybridAttention(SequenceContentAttention):

    def __init__(self, location_attention, **kwargs):
        super(HybridAttention, self).__init__(**kwargs)
        self.location_attention = location_attention
        self.children += [self.location_attention]

    @application
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, weights=None, **states):
        content_energies = self.compute_energies(
            attended, preprocessed_attended, states)
        location_energies = self.location_attention.compute_energies(
            states, weights).T
        energies = content_energies + location_energies
        new_weights = self.compute_weights(energies, attended_mask)
        weighted_averages = self.compute_weighted_averages(new_weights,
                                                           attended)
        return weighted_averages, new_weights.T

    def _push_allocation_config(self):
        super(HybridAttention, self)._push_allocation_config()
        self.location_attention.state_dims = self.state_dims
        self.location_attention.attended_dim = self.attended_dim

    @take_glimpses.delegate
    def take_glimpses_delegate(self):
        return self.location_attention.take_glimpses

    @application
    def initial_glimpses(self, *args, **kwargs):
        return self.location_attention.initial_glimpses(*args, **kwargs)
