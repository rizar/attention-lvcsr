import theano
import numpy
import scipy
from theano import tensor

from blocks.bricks import Initializable, Linear
from blocks.bricks.parallel import Parallel
from blocks.bricks.base import lazy, application
from blocks.bricks.attention import (
    GenericSequenceAttention, SequenceContentAttention,
    ShallowEnergyComputer)
from blocks.utils import put_hook, ipdb_breakpoint, shared_floatx

floatX = theano.config.floatX


class ShiftPredictor(GenericSequenceAttention, Initializable):

    @lazy
    def __init__(self, predictor, max_left, max_right, padding, **kwargs):
        super(GenericSequenceAttention, self).__init__(**kwargs)
        assert len(self.state_names) == 1
        self.max_left = max_left
        self.max_right = max_right
        self.padding = padding
        self.predictor = predictor

        self.span = self.max_right + self.max_left + 1
        self.children = [self.predictor]

    def _push_allocation_config(self):
        self.predictor.input_dim, = self.state_dims
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
        zero_row = self.padding * tensor.ones((length + self.span,))
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


class ShiftPredictor2(GenericSequenceAttention, Initializable):


    @lazy
    def __init__(self, predictor, max_left, **kwargs):
        super(GenericSequenceAttention, self).__init__(**kwargs)
        assert len(self.state_names) == 1
        self.predictor = predictor
        self.children = [self.predictor]

    def _push_allocation_config(self):
        self.predictor.input_dim = 1
        self.predictor.output_dim = 1

    @application
    def compute_energies(self, _, previous_weights):
        """Compute energies.

        Parameters
        ----------
        previous_weights : Theano variable
            Weights from the previous step, (batch_size, attended_len).

        """
        batch_size = previous_weights.shape[0]
        length = previous_weights.shape[1]
        positions = tensor.arange(length).astype(floatX)
        # Positions are broadcasted along the first dimension
        expected_positions = (previous_weights * positions).sum(axis=1)
        shifts = positions[None, :] - expected_positions[:, None]
        flat_energies = self.predictor.apply(shifts.flatten()[:, None])
        return flat_energies.reshape((batch_size, length))


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


class SequenceContentAndCumSumAttention(GenericSequenceAttention, Initializable):
    @lazy()
    def __init__(self, match_dim, state_transformer=None,
                 attended_transformer=None, energy_computer=None,
                 prior_mean=None, prior_std=None, **kwargs):
        super(SequenceContentAndCumSumAttention, self).__init__(**kwargs)
        self.match_dim = match_dim
        self.state_transformer = state_transformer

        self.state_transformers = Parallel(input_names=self.state_names,
                                           prototype=state_transformer,
                                           name="state_trans")
        if not attended_transformer:
            attended_transformer = Linear(name="preprocess")
        if not energy_computer:
            energy_computer = ShallowEnergyComputer(name="energy_comp")
        self.cumweight_handler = Linear(name="cumweight")
        self.attended_transformer = attended_transformer
        self.energy_computer = energy_computer

        self.prior = None
        if prior_mean:
            self.max_length = 5000
            values = numpy.zeros(2 * self.max_length + 1)
            values[self.max_length + 10] = 5
            self.prior = shared_floatx(values, name='prior')
            #xs = numpy.arange(-self.max_length, self.max_length)
            #self.prior = shared_floatx(
            #    scipy.stats.norm.pdf(xs, loc=prior_mean, scale=prior_std), name='prior')
            self.params = [self.prior]

        self.children = [self.state_transformers, self.attended_transformer,
                         self.energy_computer, self.cumweight_handler]

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = [self.match_dim
                                               for name in self.state_names]
        self.attended_transformer.input_dim = self.attended_dim
        self.attended_transformer.output_dim = self.match_dim
        self.energy_computer.input_dim = self.match_dim
        self.energy_computer.output_dim = 1
        self.cumweight_handler.input_dim = 1
        self.cumweight_handler.output_dim = self.match_dim

    @application
    def compute_prior(self, previous_weights):
        modes = previous_weights.argmax(axis=1)
        length = previous_weights.shape[1]
        def scan_function(mode, prior_, length_):
            # It is important to have "length_" here,
            # because otherwise `theano.scan` uses `previous_weights`
            # as input and "eats" the intermediate variable with role.
            start = self.max_length - mode
            return prior_[start:start + length_]
        result, _ = theano.scan(scan_function,
            sequences=[modes], non_sequences=[self.prior, length],
            outputs_info=[None])
        return result

    @application
    def compute_energies(self, attended, preprocessed_attended,
                         previous_weights, states):
        if not preprocessed_attended:
            preprocessed_attended = self.preprocess(attended)
        transformed_states = self.state_transformers.apply(as_dict=True,
                                                           **states)
        # Broadcasting of transformed states should be done automatically
        match_vectors = sum(transformed_states.values(),
                            preprocessed_attended)
        match_vectors += self.cumweight_handler.apply(
            tensor.cumsum(previous_weights[:, :, None], axis=1)).dimshuffle(1, 0, 2)
        energies = self.energy_computer.apply(match_vectors).reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        return energies

    @application(outputs=['weighted_averages', 'weights'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, weights=None, **states):
        energies = self.compute_energies(attended, preprocessed_attended,
                                         weights, states)
        if self.prior:
            energies += self.compute_prior(weights).T
        weights = self.compute_weights(energies, attended_mask)
        weighted_averages = self.compute_weighted_averages(weights, attended)
        return weighted_averages, weights.T

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

    @application(inputs=['attended'], outputs=['preprocessed_attended'])
    def preprocess(self, attended):
        return self.attended_transformer.apply(attended)

    def get_dim(self, name):
        if name in ['weighted_averages']:
            return self.attended_dim
        if name in ['weights']:
            return 0
        return super(SequenceContentAndCumSumAttention, self).get_dim(name)

# TODO

class WindowCumSumAttention(Initializable):
    """Symmetric window + prior + cumsum pruning.

    A good attention mechanism for speech recognition should have
    the following properties:

    * aligment should it be at least roughly sensible from the very beginning,
    otherwise we can enter very nasty local minima that are hard to leave
    (at least with good old gradient descent)

    * attention should be robust, that is in the case of confusion risky
    choices (like jumps too far away) should be avoid and safe choices (move
    10 frame forward) should be prefered

    * attention should be efficient, that is we should not search over the
    whole input utterance

    While it will probably be hard to meet all these goals, we try to do that
    by

    * using a prior probability of shift relative to the previous mode

    * using a window centered in the previous mode position

    * using cumulative sum of alignment from the previous step as an additional
    input

    """
    @lazy()
    def __init__(self, match_dim, state_transformer=None,
                 attended_transformer=None, energy_computer=None, **kwargs):
        super(SequenceContentAndCumSumAttention, self).__init__(**kwargs)
        self.match_dim = match_dim
        self.state_transformer = state_transformer

        self.state_transformers = Parallel(input_names=self.state_names,
                                           prototype=state_transformer,
                                           name="state_trans")
        if not attended_transformer:
            attended_transformer = Linear(name="preprocess")
        if not energy_computer:
            energy_computer = ShallowEnergyComputer(name="energy_comp")
        self.cumweight_handler = Linear(name="cumweight")
        self.attended_transformer = attended_transformer
        self.energy_computer = energy_computer

        self.children = [self.state_transformers, self.attended_transformer,
                         self.energy_computer, self.cumweight_handler]

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = [self.match_dim
                                               for name in self.state_names]
        self.attended_transformer.input_dim = self.attended_dim
        self.attended_transformer.output_dim = self.match_dim
        self.energy_computer.input_dim = self.match_dim
        self.energy_computer.output_dim = 1
        self.cumweight_handler.input_dim = 1
        self.cumweight_handler.output_dim = self.match_dim

    @application
    def compute_energies(self, attended, preprocessed_attended,
                         previous_weights, states):
        if not preprocessed_attended:
            preprocessed_attended = self.preprocess(attended)
        transformed_states = self.state_transformers.apply(as_dict=True,
                                                           **states)
        # Broadcasting of transformed states should be done automatically
        match_vectors = sum(transformed_states.values(),
                            preprocessed_attended)
        match_vectors += self.cumweight_handler.apply(
            tensor.cumsum(previous_weights[:, :, None], axis=1)).dimshuffle(1, 0, 2)
        energies = self.energy_computer.apply(match_vectors).reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        return energies

    @application(outputs=['weighted_averages', 'weights'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, weights=None, **states):
        energies = self.compute_energies(attended, preprocessed_attended,
                                         weights, states)
        weights = self.compute_weights(energies, attended_mask)
        weighted_averages = self.compute_weighted_averages(weights, attended)
        return weighted_averages, weights.T

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

    @application(inputs=['attended'], outputs=['preprocessed_attended'])
    def preprocess(self, attended):
        return self.attended_transformer.apply(attended)

    def get_dim(self, name):
        if name in ['weighted_averages']:
            return self.attended_dim
        if name in ['weights']:
            return 0
        return super(SequenceContentAndCumSumAttention, self).get_dim(name)
