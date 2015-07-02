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
    def __init__(self, fst, output_symbols, **kwargs):
        super(FSTTransition, self).__init__(**kwargs)
        self.fst = fst
        self.transition = FSTTransitionOp(fst)
        self.probability_computer = FSTProbabilitiesOp(fst, output_symbols)
        self.out_dim = len(output_symbols)

    @recurrent(sequences=['inputs', 'mask'],
               states=['states', 'outputs', 'weights'],
               outputs=['states', 'outputs', 'weights'], contexts=[])
    def apply(self, inputs, states=None, outputs=None, weights=None,
              mask=None):
        new_states, output = self.transition(states, inputs)
        if mask:
            new_states = tensor.cast(mask * new_states +
                                     (1. - mask) * states, 'int64')
        weights = self.probability_computer(states)
        return new_states, tensor.cast(output, 'int64'), weights

    @application(outputs=['states', 'outputs', 'weights'])
    def initial_states(self, batch_size, *args, **kwargs):
        return (tensor.zeros((batch_size,), dtype='int64'),
                tensor.zeros((batch_size,), dtype='int64'),
                tensor.zeros((batch_size, self.out_dim)))

    def get_dim(self, name):
        if name == 'states':
            return 0
        if name == 'outputs':
            return 0
        if name == 'weights':
            return self.out_dim
        if name == 'inputs':
            return 0
        return super(FSTTransition, self).get_dim(name)


class FSTReadout(AbstractReadout, Random):
    def __init__(self, source_names, readout_dim, weights_dim,
                 beta=1., **kwargs):
        super(FSTReadout, self).__init__(source_names, readout_dim, **kwargs)

        self.beta = beta
        self.weights_dim = weights_dim

    @application
    def readout(self, **kwargs):
        return kwargs['weights']

    @application
    def emit(self, readouts):
        batch_size = readouts.shape[0]
        pvals_flat = tensor.exp(-readouts.reshape((batch_size, -1)))
        generated = self.theano_rng.multinomial(pvals=pvals_flat)
        return generated.reshape(readouts.shape).argmax(axis=-1)

    @application
    def cost(self, readouts, outputs):
        max_output = readouts.shape[-1]
        flat_outputs = outputs.flatten()
        num_outputs = flat_outputs.shape[0]
        return (readouts.flatten()[max_output * tensor.arange(num_outputs) +
                                   flat_outputs].reshape(outputs.shape))

    @application
    def initial_outputs(self, batch_size):
        return tensor.zeros((batch_size,))

    @application(outputs=['feedback'])
    def feedback(self, outputs):
        return outputs

    def get_dim(self, name):
        if name == 'outputs':
            return 0
        elif name == 'feedback':
            return 0
        elif name == 'readouts':
            return self.readout_dim
        elif name == 'weights':
            return self.weights_dim
        return super(FSTReadout, self).get_dim(name)


class ShallowFusionReadout(Readout):
    def __init__(self, lm_weights, beta=1, **kwargs):
        kwargs.setdefault('emitter', ShallowFusionEmitter(beta=beta))
        super(ShallowFusionReadout, self).__init__(**kwargs)
        self.lm_weights = lm_weights
        self.beta = beta

    @application
    def readout(self, **kwargs):
        return (kwargs[self.lm_weights],
                super(ShallowFusionReadout, self).readout(**kwargs))


class ShallowFusionEmitter(SoftmaxEmitter):
    def __init__(self, beta=1., **kwargs):
        self.beta = beta
        super(ShallowFusionEmitter, self).__init__(**kwargs)

    @application
    def probs(self, readouts):
        weights, readouts = readouts
        return (super(ShallowFusionEmitter, self).probs(readouts) +
                self.beta * tensor.exp(-weights))

