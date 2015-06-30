from theano import tensor

from blocks.bricks import Initializable, Linear
from blocks.bricks.base import lazy, application
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import BaseRecurrent, recurrent
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
    def __init__(self, fst, output_symbols, **kwargs):
        self.fst = fst
        self.transition = FSTTransitionOp(fst)
        self.probability_computer = FSTProbabilitiesOp(fst, output_symbols)
        super(FSTTransition, self).__init__(**kwargs)

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states', 'outputs', 'weights'], contexts=[])
    def apply(self, inputs=None, states=None, mask=None):
        new_states, output = self.transition(states, inputs)
        if mask:
            new_states = mask * new_states + (1. - mask) * states
        weights = self.probability_computer(states)
        return new_states, output, weights

    @application(outputs=['states'])
    def initial_states(self, batch_size, *args, **kwargs):
        return tensor.zeros((batch_size,), dtype='int64')

    def get_dim(self, name):
        if name == 'states':
            return 0
        return super(FSTTransition, self).get_dim(name)