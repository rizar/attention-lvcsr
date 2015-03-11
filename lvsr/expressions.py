from theano import tensor

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
