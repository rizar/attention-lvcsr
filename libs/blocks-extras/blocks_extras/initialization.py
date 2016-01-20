import numpy
import theano

from blocks_extras.utils import check_valid_permutation
from blocks.initialization import NdarrayInitialization


class PermutationMatrix(NdarrayInitialization):
    """Generates a 2-dimensional permutation matrix.

    Parameters
    ----------
    permutation : ndarray, 1-dimensional, optional
        A permutation on the integers in a given range. If specified,
        always generate the permutation matrix corresponding to this
        permutation, ignoring the random number generator.

    """
    def __init__(self, permutation=None):
        if permutation is not None:
            permutation = check_valid_permutation(permutation)
        self.permutation = permutation

    def generate(self, rng, shape):
        def make_matrix(size, perm):
            return numpy.eye(size, dtype=theano.config.floatX)[:, perm]

        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("requested shape is not square")
        if self.permutation is not None:
            if shape[0] != len(self.permutation):
                raise ValueError("provided permutation does not match "
                                 "requested shape")
            return make_matrix(shape[0], self.permutation)
        else:
            return make_matrix(shape[0], rng.permutation(shape[0]))
