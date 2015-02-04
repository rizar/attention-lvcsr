from matplotlib.mlab import specgram

def spectrogram(signal):
    return specgram(signal)[0].T
