from blocks.datasets.streams import DataStreamMapping

import numpy
from matplotlib.mlab import specgram

def log_spectrogram(signal):
    return numpy.log(specgram(signal)[0].T)


class Normalization(object):

    def __init__(self, data_stream, source):
        index = data_stream.sources.index(source)

        sum_features = 0
        sum_features2 = 0
        num_examples = 0

        iterator = data_stream.get_epoch_iterator()
        for data in iterator:
            features = data[index]
            sum_features += features.sum(axis=0)
            sum_features2 += (features ** 2).sum(axis=0)
            num_examples += len(features)

        mean_features = sum_features / num_examples
        std_features = (sum_features2 / num_examples - mean_features ** 2) ** 0.5

        self.mean_features = mean_features
        self.std_features = std_features
        self.index = index

    def apply(self, data):
        data = list(data)
        data[self.index] = ((data[self.index] - self.mean_features)
                            / self.std_features)
        return tuple(data)

    def wrap_stream(self, stream):
        return DataStreamMapping(stream, Invoke(self, 'apply'))


class Invoke(object):

    def __init__(self, object_, method):
        self.object_ = object_
        self.method = method

    def __call__(self, *args, **kwargs):
        return getattr(self.object_, self.method)(*args, **kwargs)


