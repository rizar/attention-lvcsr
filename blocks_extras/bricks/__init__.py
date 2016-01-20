from theano import shared, tensor
from blocks.bricks import Feedforward
from blocks.bricks.base import application, lazy
from blocks_extras.initialization import PermutationMatrix
from blocks_extras.utils import check_valid_permutation
from blocks.utils import shared_floatx


class FixedPermutation(Feedforward):
    """Perform a fixed permutation of the input features.

    Parameters
    ----------
    order : ndarray-like
        A 1-dimensional container containing a permutation
        on the integers.
    dot : bool, optional
        Whether or not to perform the permutation by matrix
        multiplication. This may be faster in some circumstances
        but requires allocation of a permutation matrix.

    """
    @lazy(allocation=['order'])
    def __init__(self, order, dot=True, **kwargs):
        self.order = order
        self._dot = dot
        super(FixedPermutation, self).__init__(**kwargs)

    def _allocate(self):
        self.order = check_valid_permutation(self.order)
        if self.input_dim != len(self.order):
            raise ValueError("input_dim does not match length of order "
                             "vector")
        # No roles assigned here, since these are not learnable parameters.
        if self._dot:
            shape = (self.order.shape[0], self.order.shape[0])
            self._matrix = shared_floatx(
                PermutationMatrix(self.order).generate(None, shape))
        else:
            order = self.order.astype('int32')
            assert order.min() == 0  # Catch highly unlikely downcast issue.
            self._permutation = shared(order)

    @property
    def input_dim(self):
        return len(self.order)

    @application(inputs=['input_'], outputs=['output_'])
    def apply(self, input_):
        if self._dot:
            return tensor.dot(input_, self._matrix)
        else:
            return tensor.take(input_, self._permutation, axis=1)
