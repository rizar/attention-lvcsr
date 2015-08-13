import numpy

from theano import tensor

from blocks.bricks import (
    Initializable, Linear, Random, Brick, NDimensionalSoftmax)
from blocks.bricks.base import lazy, application
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks.sequence_generators import (
    AbstractReadout, Readout, AbstractEmitter)
from blocks.bricks.wrappers import WithExtraDims
from blocks.utils import dict_union

from lvsr.ops import FSTCostsOp, FSTTransitionOp, MAX_STATES, NOT_STATE

class RecurrentWithFork(Initializable):

    @lazy(allocation=['input_dim'])
    def __init__(self, recurrent, input_dim, **kwargs):
        super(RecurrentWithFork, self).__init__(**kwargs)
        self.recurrent = recurrent
        self.input_dim = input_dim
        self.fork = Fork(
            [name for name in self.recurrent.sequences
             if name != 'mask'],
             prototype=Linear())
        self.children = [recurrent.brick, self.fork]

    def _push_allocation_config(self):
        self.fork.input_dim = self.input_dim
        self.fork.output_dims = [self.recurrent.brick.get_dim(name)
                                 for name in self.fork.output_names]

    @application(inputs=['input_', 'mask'])
    def apply(self, input_, mask=None, **kwargs):
        return self.recurrent(
            mask=mask, **dict_union(self.fork.apply(input_, as_dict=True),
                                    kwargs))

    @apply.property('outputs')
    def apply_outputs(self):
        return self.recurrent.states


class FSTTransition(BaseRecurrent, Initializable):
    def __init__(self, fst, remap_table, no_transition_cost,
                 allow_spelling_unknowns,
                 start_new_word_state, space_idx,
                 all_weights_to_zeros,
                 **kwargs):
        """Wrap FST in a recurrent brick.

        Parameters
        ----------
        fst : FST instance
        remap_table : dict
            Maps neutral network characters to FST characters.
        no_transition_cost : float
            Cost of going to the start state when no arc for an input
            symbol is available.
        allow_spelling_unknowns : bool
            do we allow to emit characters outside of th edictionary
        start_new_word_state : int
            "Main looping state" of the FST which we enter after following backoff links
        space_idx : int
            id of the space character in network coding
        all_weights_to_zero : bool
            Ignore all weights as if they all were zeros.


        """
        super(FSTTransition, self).__init__(**kwargs)
        self.fst = fst
        self.transition = FSTTransitionOp(fst, remap_table,
                                          start_new_word_state=start_new_word_state,
                                          space_idx=space_idx,
                                          allow_spelling_unknowns=allow_spelling_unknowns)
        self.probability_computer = FSTCostsOp(
            fst, remap_table, no_transition_cost, all_weights_to_zeros)

        self.out_dim = len(remap_table)

    @recurrent(sequences=['inputs', 'mask'],
               states=['states', 'weights', 'add'],
               outputs=['states', 'weights', 'add'], contexts=[])
    def apply(self, inputs, states, weights, add,
              mask=None):
        new_states, new_weights = self.transition(states, weights, inputs)
        if mask:
            # In fact I don't really understand why we do this:
            # anyway states not covered by masks should have no effect
            # on the cost...
            new_states = tensor.cast(mask * new_states +
                                     (1. - mask) * states, 'int64')
            new_weights = mask * new_weights + (1. - mask) * weights
        new_add = self.probability_computer(new_states, new_weights)
        return new_states, new_weights, new_add

    @application(outputs=['states', 'weights', 'add'])
    def initial_states(self, batch_size, *args, **kwargs):
        states = tensor.as_tensor_variable(
            self.transition.pad([self.fst.fst.start], NOT_STATE))
        states = tensor.tile(states[None, :], (batch_size, 1))
        weights = tensor.as_tensor_variable(
            self.transition.pad([0.0], 0))
        weights = tensor.tile(weights[None, :], (batch_size, 1))
        return (states, weights,
                tensor.zeros((batch_size, self.out_dim)))

    def get_dim(self, name):
        if name == 'states' or name == 'weights':
            return MAX_STATES
        if name == 'add':
            return self.out_dim
        if name == 'inputs':
            return 0
        return super(FSTTransition, self).get_dim(name)


class ShallowFusionReadout(Readout):
    def __init__(self, lm_costs_name, lm_weight,
                 normalize_am_weights=False,
                 normalize_lm_weights=False,
                 normalize_tot_weights=True,
                 am_beta=1.0,
                 **kwargs):
        super(ShallowFusionReadout, self).__init__(**kwargs)
        self.lm_costs_name = lm_costs_name
        self.lm_weight = lm_weight
        self.normalize_am_weights = normalize_am_weights
        self.normalize_lm_weights = normalize_lm_weights
        self.normalize_tot_weights = normalize_tot_weights
        self.am_beta = am_beta
        self.softmax = NDimensionalSoftmax()
        self.children += [self.softmax]

    @application
    def readout(self, **kwargs):
        lm_costs = -kwargs.pop(self.lm_costs_name)
        if self.normalize_lm_weights:
            lm_costs = self.softmax.log_probabilities(
                lm_costs, extra_ndim=lm_costs.ndim - 2)
        am_pre_softmax = self.am_beta * super(ShallowFusionReadout, self).readout(**kwargs)
        if self.normalize_am_weights:
            am_pre_softmax = self.softmax.log_probabilities(
                am_pre_softmax, extra_ndim=am_pre_softmax.ndim - 2)
        x = am_pre_softmax + self.lm_weight * lm_costs
        if self.normalize_tot_weights:
            x = self.softmax.log_probabilities(x, extra_ndim=x.ndim - 2)
        return x


class SelectInEachRow(Brick):
    @application(inputs=['matrix', 'indices'], outputs=['output_'])
    def apply(self, matrix, indices):
        return matrix[tensor.arange(matrix.shape[0]), indices]


class SelectInEachSubtensor(SelectInEachRow):
    decorators = [WithExtraDims()]


class LMEmitter(AbstractEmitter):
    """Emitter to use when language model is used.

    Since with the language model all normalization is
    done in ShallowFusionReadout, we need this no-op brick to
    interface it with BeamSearch.

    """
    @lazy(allocation=['readout_dim'])
    def __init__(self, readout_dim, **kwargs):
        super(LMEmitter, self).__init__(**kwargs)
        self.readout_dim = readout_dim
        self.select = SelectInEachSubtensor()
        self.children = [self.select]

    @application
    def emit(self, readouts):
        # Non-sense, but the returned result should never be used.
        return tensor.zeros_like(readouts[:, 0], dtype='int64')

    @application
    def cost(self, readouts, outputs):
        return -self.select.apply(
            readouts, outputs, extra_ndim=readouts.ndim - 2)

    @application
    def costs(self, readouts):
        return -readouts

    @application
    def initial_outputs(self, batch_size):
        # As long as we do not use the previous character, can be anything
        return tensor.zeros((batch_size,), dtype='int64')

    def get_dim(self, name):
        if name == 'outputs':
            return 0
        return super(LMEmitter, self).get_dim(name)
