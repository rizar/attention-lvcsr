from theano import tensor
from theano.tensor.nnet import conv2d

def weights_std(weights, mask_outputs=None):
    positions = tensor.arange(weights.shape[2])
    expected = (weights * positions).sum(axis=2)
    expected2 = (weights * positions ** 2).sum(axis=2)
    result = (expected2 - expected ** 2) ** 0.5
    if mask_outputs:
        result *= mask_outputs
    return result.sum() / weights.shape[0]


def monotonicity_penalty(weights, mask_x=None):
    cumsums = tensor.cumsum(weights, axis=2)
    penalties = tensor.maximum(cumsums[1:] - cumsums[:-1], 0).sum(axis=2)
    if mask_x:
        penalties *= mask_x[1:]
    return penalties.sum()


def entropy(weights, mask_x):
    entropies = (weights * tensor.log(weights + 1e-7)).sum(axis=2)
    entropies *= mask_x
    return entropies.sum()


def conv1d(sequences, masks, **kwargs):
    """Wraps Theano conv2d to perform 1D convolution.

    Parameters
    ----------
    sequence : :class:`~theano.Variable`
        (batch_size, length)
    masks : :class:`~theano.Variable`
        (num_filters, filter_length)
    **kwargs
        Will be passed to `conv2d`

    Returns
    -------
    result : :class:`~theano.Variable`
        (batch_size, num_filters, position)

    """
    # For testability
    sequences = tensor.as_tensor_variable(sequences)
    masks = tensor.as_tensor_variable(masks)
    image = sequences.dimshuffle('x', 'x', 0, 1)
    filters = masks.dimshuffle(0, 'x', 'x', 1)
    result = conv2d(image, filters, **kwargs)
    # Now number of rows is the actual batch size
    result = result.dimshuffle(2, 1, 3, 0)
    return result.reshape(result.shape[:-1], ndim=3)


def pad_to_a_multiple(tensor_, k, pad_with):
    """Pad a tensor to make its first dimension a multiple of a number.

    Parameters
    ----------
    tensor_ : :class:`~theano.Variable`
    k : int
        The number, multiple of which the length of tensor is made.
    pad_with : float or int
        The value for padding.

    """
    new_length = (
        tensor.ceil(tensor_.shape[0].astype('float32') / k) * k).astype('int64')
    new_shape = tensor.set_subtensor(tensor_.shape[:1], new_length)
    canvas = tensor.alloc(pad_with, tensor.prod(new_shape)).reshape(
        new_shape, ndim=tensor_.ndim)
    return tensor.set_subtensor(canvas[:tensor_.shape[0]], tensor_)

