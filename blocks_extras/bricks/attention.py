from theano import tensor

from blocks.bricks.base import application
from blocks.bricks.attention import AbstractAttention


class SimpleSequenceAttention(AbstractAttention):
    """Attention mechanism for a same size sequence.

    Combines a conditioning sequence and a recurrent transition via
    attention. The conditioning sequence should have the same number
    of steps as the sequence. See :class:`AbstractAttention` for an
    explanation of use.

    Notes
    -----
    The conditioning sequence should have the shape:
    (seq_length, batch_size, features)

    """
    @application(outputs=['glimpses', 'step'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, step=None, **states):
        return attended[step, tensor.arange(attended.shape[1]), :], step + 1

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        return (['attended', 'preprocessed_attended',
                 'attended_mask', 'step'] +
                self.state_names)

    @application(outputs=['glimpses', 'step'])
    def initial_glimpses(self, batch_size, attended=None):
        return ([tensor.zeros((batch_size, self.attended_dim))] +
                [tensor.zeros((batch_size,), dtype='int64')])

    def get_dim(self, name):
        if name == 'step':
            return 0
        if name == 'glimpses':
            return self.attended_dim
        return super(SimpleSequenceAttention, self).get_dim(name)
