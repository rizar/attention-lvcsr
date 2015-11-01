#!/usr/bin/env/python
#
# A very temporary script that adds embeddings for one more character
# to the parameter.

import sys
import numpy

def add_one_dim(param, i):
    new_shape = list(param.shape)
    old_dim = new_shape[i]
    new_shape[i] += 1
    new_value = numpy.zeros(new_shape, dtype='float32')
    indices = len(new_shape) * [slice(None,)]
    indices[i] = slice(old_dim)
    new_value[indices] = param
    return new_value

params = numpy.load(sys.argv[1])
new_params = {}
for key, value in params.items():
    if hasattr(value, 'shape'):
        if 33 in value.shape:
            value = add_one_dim(value, value.shape.index(33))
        if 32 in value.shape:
            value = add_one_dim(value, value.shape.index(32))
    new_params[key] = value
with open(sys.argv[2], 'w') as dest:
    numpy.savez(dest, **new_params)
