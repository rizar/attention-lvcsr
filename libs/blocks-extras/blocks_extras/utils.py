import numpy


def check_valid_permutation(permutation):
    """Check that a given container contains a valid permutation.

    Parameters
    ----------
    permutation : array-like, 1-dimensional
        An array, list, etc.

    Returns
    -------
    permutation : ndarray, 1-dimensional
        An ndarray of integers representing a permutation.

    Raises
    ------
    ValueError
        If the given input is not a valid permutation on the
        integers.

    """
    permutation = numpy.asarray(permutation)
    if permutation.ndim != 1:
        raise ValueError("expected 1-dimensional permutation argument")
    elif permutation.dtype.kind != 'i':
        raise ValueError("expected integer dtype argument")
    elif (len(set(permutation)) != max(permutation) + 1 or
            min(permutation) < 0):
        raise ValueError("invalid permutation")
    return permutation
