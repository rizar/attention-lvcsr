import numpy
from matplotlib.mlab import specgram

def log_spectrogram(signal):
    return numpy.log(specgram(signal)[0].T)
