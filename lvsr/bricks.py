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
    def __init__(self, fst, remap_table, no_transition_cost, **kwargs):
        """Wrap FST in a recurrent brick.

        Parameters
        ----------
        fst : FST instance
        remap_table : dict
            Maps neutral network characters to FST characters.
        no_transition_cost : float
            Cost of going to the start state when no arc for an input
            symbol is available.

        """
        super(FSTTransition, self).__init__(**kwargs)
        self.fst = fst
        self.transition = FSTTransitionOp(fst, remap_table)
        self.probability_computer = FSTProbabilitiesOp(
            fst, remap_table, no_transition_cost)

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


class ShallowFusionReadout(Readout):
    def __init__(self, lm_logprobs_name, lm_weight, **kwargs):
        super(ShallowFusionReadout, self).__init__(**kwargs)
        self.lm_logprobs_name = lm_logprobs_name
        self.lm_weight = lm_weight

    @application
    def readout(self, **kwargs):
        from blocks.utils import put_hook
        def func(x):
            print x[0]
        lm_softmax = -kwargs.pop(self.lm_logprobs_name)
        lm_softmax = put_hook(lm_softmax, func)
        am_softmax = super(ShallowFusionReadout, self).readout(**kwargs)
        am_softmax = put_hook(am_softmax, func)
        result = am_softmax + self.lm_weight * lm_softmax
        result = put_hook(result, func)
        return result
