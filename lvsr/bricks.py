from theano import tensor

from blocks.bricks import Initializable, Linear, Random, Brick
from blocks.bricks.base import lazy, application
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks.sequence_generators import AbstractReadout, Readout, SoftmaxEmitter
from blocks.utils import dict_union

from lvsr.ops import FSTProbabilitiesOp, FSTTransitionOp

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
        self.probability_computer = FSTProbabilitiesOp(
            fst, remap_table, no_transition_cost, all_weights_to_zeros)

        self.out_dim = len(remap_table)

    @recurrent(sequences=['inputs', 'mask'],
               states=['states', 'logprobs'],
               outputs=['states', 'logprobs'], contexts=[])
    def apply(self, inputs, states, logprobs,
              mask=None):
        new_states = self.transition(states, inputs)
        if mask:
            new_states = tensor.cast(mask * new_states +
                                     (1. - mask) * states, 'int64')
        logprobs = self.probability_computer(new_states)
        return new_states, logprobs

    @application(outputs=['states', 'logprobs'])
    def initial_states(self, batch_size, *args, **kwargs):
        return (tensor.ones((batch_size,), dtype='int64') * self.fst.fst.start,
                tensor.zeros((batch_size, self.out_dim)))

    def get_dim(self, name):
        if name == 'states':
            return 0
        if name == 'logprobs':
            return self.out_dim
        if name == 'inputs':
            return 0
        return super(FSTTransition, self).get_dim(name)


def normalize_log_probs(x):
    x = x - x.max(axis=(x.ndim-1), keepdims=True)
    x = x - tensor.log(tensor.exp(x).sum(axis=(x.ndim-1), keepdims=True))
    return x

class ShallowFusionReadout(Readout):
    def __init__(self, lm_logprobs_name, lm_weight,
                 normalize_am_weights=False,
                 normalize_lm_weights=False,
                 normalize_tot_weights=True,
                 am_beta=1.0,
                 **kwargs):
        super(ShallowFusionReadout, self).__init__(**kwargs)
        self.lm_logprobs_name = lm_logprobs_name
        self.lm_weight = lm_weight
        self.normalize_am_weights = normalize_am_weights
        self.normalize_lm_weights = normalize_lm_weights
        self.normalize_tot_weights = normalize_tot_weights
        self.am_beta = am_beta

    @application
    def readout(self, **kwargs):
        lm_pre_softmax = -kwargs.pop(self.lm_logprobs_name)
        if self.normalize_lm_weights:
            lm_pre_softmax = normalize_log_probs(lm_pre_softmax)
        am_pre_softmax = self.am_beta * super(ShallowFusionReadout, self).readout(**kwargs)
        if self.normalize_am_weights:
            am_pre_softmax = normalize_log_probs(am_pre_softmax)
        x = am_pre_softmax + self.lm_weight * lm_pre_softmax
        if self.normalize_tot_weights:
            x = normalize_log_probs(x)
        return x
