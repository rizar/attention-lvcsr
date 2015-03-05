from theano import tensor

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
