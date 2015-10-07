import logging
import numpy
from theano import tensor

from blocks.bricks import (
    Initializable, Linear, Sequence, Tanh)
from blocks.bricks.base import lazy, application
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import Bidirectional
from blocks.bricks.sequence_generators import (
    AbstractFeedback, LookupFeedback, AbstractEmitter)
from blocks.utils import dict_union, check_theano_variable

logger = logging.getLogger(__name__)


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


class InitializableSequence(Sequence, Initializable):
    pass


class Encoder(Initializable):

    def __init__(self, enc_transition, dims, dim_input, subsample, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.subsample = subsample

        for layer_num, (dim_under, dim) in enumerate(
                zip([dim_input] + list(2 * numpy.array(dims)), dims)):
            bidir = Bidirectional(
                RecurrentWithFork(
                    enc_transition(dim=dim, activation=Tanh()).apply,
                    dim_under,
                    name='with_fork'),
                name='bidir{}'.format(layer_num))
            self.children.append(bidir)

    @application(outputs=['encoded', 'encoded_mask'])
    def apply(self, input_, mask=None):
        for bidir, take_each in zip(self.children, self.subsample):
            #No need to pad if all we do is subsample!
            #input_ = pad_to_a_multiple(input_, take_each, 0.)
            #if mask:
            #    mask = pad_to_a_multiple(mask, take_each, 0.)
            input_ = bidir.apply(input_, mask)
            input_ = input_[::take_each]
            if mask:
                mask = mask[::take_each]
        return input_, (mask if mask else tensor.ones_like(input_[:, :, 0]))


class OneOfNFeedback(AbstractFeedback, Initializable):
    """A feedback brick for the case when readout are integers.

    Stores and retrieves distributed representations of integers.

    """
    def __init__(self, num_outputs=None, feedback_dim=None, **kwargs):
        super(OneOfNFeedback, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.feedback_dim = num_outputs

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0
        eye = tensor.eye(self.num_outputs)
        check_theano_variable(outputs, None, "int")
        output_shape = [outputs.shape[i]
                        for i in range(outputs.ndim)] + [self.feedback_dim]
        return eye[outputs.flatten()].reshape(output_shape)

    def get_dim(self, name):
        if name == 'feedback':
            return self.feedback_dim
        return super(LookupFeedback, self).get_dim(name)


class RewardRegressionEmitter(AbstractEmitter):

    REWARD_MATRIX = 'reward_matrix'

    @application
    def cost(self, readouts, outputs):
        if readouts.ndim == 3:
            # The default reward matrix is built assuming that
            # outputs is the groundtruth sequence
            temp_shape = (readouts.shape[0] * readouts.shape[1], -1)
            reward_matrix = (-1 * tensor.ones_like(readouts)).reshape(temp_shape)
            reward_matrix[tensor.arange(reward_matrix.shape[0]),
                          outputs.flatten()] = 0
            reward_matrix.reshape(readouts.shape)
            reward_matrix.tag.name = self.REWARD_MATRIX
            # Go head and substitute the reward matrix if you
            # need a different one
            return (readouts - reward_matrix) ** 2
        return readouts[tensor.arange(readouts.shape[0]), outputs]

    @application
    def emit(self, readouts):
        # As a generator, acts greedily
        return readouts.argmax(axis=1)

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
        return super(RewardRegressionEmitter, self).get_dim(name)
