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
from blocks.utils import (put_hook, ipdb_breakpoint, shared_floatx,
                          shared_floatx_nans)

from lvsr.expressions import conv1d


floatX = theano.config.floatX

import logging
logger = logging.getLogger(__name__)


class Conv1D(Initializable):

    def __init__(self, num_filters, filter_length, **kwargs):
        self.num_filters = num_filters
        self.filter_length = filter_length
        super(Conv1D, self).__init__(**kwargs)

    def _allocate(self):
        self.parameters = [shared_floatx_nans((self.num_filters, self.filter_length),
                                          name="filters")]

    def _initialize(self):
        self.weights_init.initialize(self.parameters[0], self.rng)

    def apply(self, input_):
        return conv1d(input_, self.parameters[0], border_mode="full")


class SequenceContentAndConvAttention(GenericSequenceAttention, Initializable):
    @lazy()
    def __init__(self, match_dim, conv_n, conv_num_filters=1,
                 state_transformer=None,
                 attended_transformer=None, energy_computer=None,
                 prior=None, energy_normalizer=None, **kwargs):
        super(SequenceContentAndConvAttention, self).__init__(**kwargs)
        if not state_transformer:
            state_transformer = Linear(use_bias=False)

        self.match_dim = match_dim
        self.state_transformer = state_transformer

        self.state_transformers = Parallel(input_names=self.state_names,
                                           prototype=state_transformer,
                                           name="state_trans")
        if not attended_transformer:
            # Only this contributor to the match vector
            # is allowed to have biases
            attended_transformer = Linear(name="preprocess")

        if not energy_normalizer:
            energy_normalizer = 'softmax'
        self.energy_normalizer = energy_normalizer

        if not energy_computer:
            energy_computer = ShallowEnergyComputer(
                name="energy_comp",
                use_bias=self.energy_normalizer != 'softmax')
        self.filter_handler = Linear(name="handler", use_bias=False)
        self.attended_transformer = attended_transformer
        self.energy_computer = energy_computer

        if not prior:
            prior = dict(type='expanding', initial_begin=0, initial_end=10000,
                         min_speed=0, max_speed=0)
        self.prior = prior

        self.conv_n = conv_n
        self.conv_num_filters = conv_num_filters
        self.conv = Conv1D(conv_num_filters, 2 * conv_n + 1)

        self.children = [self.state_transformers, self.attended_transformer,
                         self.energy_computer, self.filter_handler, self.conv]

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = [self.match_dim
                                               for name in self.state_names]
        self.attended_transformer.input_dim = self.attended_dim
        self.attended_transformer.output_dim = self.match_dim
        self.energy_computer.input_dim = self.match_dim
        self.energy_computer.output_dim = 1
        self.filter_handler.input_dim = self.conv_num_filters
        self.filter_handler.output_dim = self.match_dim

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
        conv_result = self.conv.apply(previous_weights)
        match_vectors += self.filter_handler.apply(
            conv_result[:, :, self.conv_n:-self.conv_n]
            .dimshuffle(0, 2, 1)).dimshuffle(1, 0, 2)
        energies = self.energy_computer.apply(match_vectors).reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        return energies

    @staticmethod
    def mask_row(offset, length, empty_row):
        return tensor.set_subtensor(empty_row[offset:offset+length], 1)

    @application(outputs=['weighted_averages', 'weights', 'energies', 'step'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, weights=None, step=None, **states):
        # Cut the considered window.
        p = self.prior
        length = attended.shape[0]
        prior_type = p.get('type', 'expanding')
        if prior_type=='expanding':
            begin = p['initial_begin'] + step[0] * p['min_speed']
            end = p['initial_end'] + step[0] * p['max_speed']
            begin = tensor.maximum(0, tensor.minimum(length - 1, begin))
            end = tensor.maximum(0, tensor.minimum(length, end))
            additional_mask = None
        elif prior_type.startswith('window_around'):
            #check whether we want the mean or median!
            if prior_type == 'window_around_mean':
                position_in_attended = tensor.arange(length, dtype=floatX)[None, :]
                expected_last_source_pos = (weights * position_in_attended).sum(axis=1)
            elif prior_type == 'window_around_median':
                ali_to_05 = tensor.extra_ops.cumsum(weights, axis=1) - 0.5
                ali_to_05 = (ali_to_05>=0)
                ali_median_pos = ali_to_05[:,1:] - ali_to_05[:,:-1]
                expected_last_source_pos = tensor.argmax(ali_median_pos, axis=1)
                expected_last_source_pos = theano.gradient.disconnected_grad(
                    expected_last_source_pos)
            else:
                raise ValueError
            #the window taken around each element
            begins = tensor.floor(expected_last_source_pos - p['before'])
            ends = tensor.ceil(expected_last_source_pos + p['after'])
            #the global window to optimize computations
            begin = tensor.maximum(0, begins.min()).astype('int64')
            end = tensor.minimum(length, ends.max()).astype('int64')
            #the new mask, already cut to begin:end
            position_in_attended_cut = tensor.arange(
                begin * 1., end * 1., 1., dtype=floatX)[None, :]
            additional_mask = ((position_in_attended_cut > begins[:,None]) *
                               (position_in_attended_cut < ends[:,None]))
        else:
            raise Exception("Unknown prior type: %s", prior_type)
        begin = tensor.floor(begin).astype('int64')
        end = tensor.ceil(end).astype('int64')
        attended_cut = attended[begin:end]
        preprocessed_attended_cut = (preprocessed_attended[begin:end]
                                     if preprocessed_attended else None)
        attended_mask_cut = (
            (attended_mask[begin:end] if attended_mask else None)
            * (additional_mask.T if additional_mask else 1))
        weights_cut = weights[:, begin:end]

        # Call
        energies_cut = self.compute_energies(attended_cut, preprocessed_attended_cut,
                                             weights_cut, states)
        weights_cut = self.compute_weights(energies_cut, attended_mask_cut)
        weighted_averages = self.compute_weighted_averages(weights_cut, attended_cut)

        # Paste
        new_weights = new_energies = tensor.zeros_like(weights.T)
        new_weights = tensor.set_subtensor(new_weights[begin:end],
                                           weights_cut)
        new_energies = tensor.set_subtensor(new_energies[begin:end],
                                            energies_cut)

        return weighted_averages, new_weights.T, new_energies.T, step + 1

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        return (['attended', 'preprocessed_attended',
                 'attended_mask', 'weights', 'step'] +
                self.state_names)

    @application
    def compute_weights(self, energies, attended_mask):
        if self.energy_normalizer == 'softmax':
            logger.debug("Using softmax attention weights normalization")
            energies = energies - energies.max(axis=0)
            unnormalized_weights = tensor.exp(energies)
        elif self.energy_normalizer == 'logistic':
            logger.debug("Using smoothfocus (logistic sigm) "
                        "attention weights normalization")
            unnormalized_weights = tensor.nnet.sigmoid(energies)
        elif self.energy_normalizer == 'relu':
            logger.debug("Using ReLU attention weights normalization")
            unnormalized_weights = tensor.maximum(energies/1000., 0.0)
        else:
            raise Exception("Unknown energey_normalizer: {}"
                            .format(self.energy_computer))
        if attended_mask:
            unnormalized_weights *= attended_mask

        # If mask consists of all zeros use 1 as the normalization coefficient
        normalization = (unnormalized_weights.sum(axis=0) +
                         tensor.all(1 - attended_mask, axis=0))
        return unnormalized_weights / normalization

    @application
    def initial_glimpses(self, batch_size, attended):
        return ([tensor.zeros((batch_size, self.attended_dim))]
            + 2 * [tensor.concatenate([
                       tensor.ones((batch_size, 1)),
                       tensor.zeros((batch_size, attended.shape[0] - 1))],
                       axis=1)]
            + [tensor.zeros((batch_size,), dtype='int64')])

    @initial_glimpses.property('outputs')
    def initial_glimpses_outputs(self):
        return ['weight_averages', 'weights', 'energies', 'step']

    @application(inputs=['attended'], outputs=['preprocessed_attended'])
    def preprocess(self, attended):
        return self.attended_transformer.apply(attended)

    def get_dim(self, name):
        if name in ['weighted_averages']:
            return self.attended_dim
        if name in ['weights', 'energies', 'step']:
            return 0
        return super(SequenceContentAndConvAttention, self).get_dim(name)
