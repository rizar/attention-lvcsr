import sys
import numpy
import cPickle
import StringIO
import base64
import struct

from matplotlib import pyplot
from pandas import rolling_mean as rmean, DataFrame
from IPython.core.display import display, HTML

models = {}

def load_log(name):
    log = cPickle.load(open(name + "_log.zip"))
    models["log_" + name] = log
    df = DataFrame.from_dict(log, orient='index')
    models["df_" + name] = df
    print "Iterations done for {}: {}".format(name, log.status['iterations_done'])
    print "Average batch time for {} was {}".format(
        name, df.time_train_this_batch.mean())
    print "Best PER: {}".format(log.status.get('best_valid_per', '?'))


def plot(names, start=0, finish=-1, window=1, max_weight_penalty=500, max_per=1.0, max_ll=None):
    indices = slice(start, finish if finish > 0 else None)

    f, axis = pyplot.subplots(1, 2)
    f.set_size_inches((15, 5))
    costs = ("valid_sequence_log_likelihood",
             "average_sequence_log_likelihood")
    for cost, ax in zip(costs, axis):
        for name in names:
            log = models['df_' + name]
            if hasattr(log, cost):
                ax.plot(
                    rmean(getattr(log, cost).interpolate()[indices], window),
                        label=name)
                ax.set_xlim(start, finish)
                if max_ll:
                    ax.set_ylim(0, max_ll)
            ax.legend()

    f, axis = pyplot.subplots(1, 2)
    f.set_size_inches(15, 5)
    for name in names:
        log = models['df_' + name]
        axis[0].plot(
            rmean(log.average_weights_entropy_per_label.interpolate()[indices], window),
                    label=name)
        axis[0].set_xlim(start, finish)
        axis[0].legend(loc='best')
    for name in names:
        log = models['df_' + name]
        axis[1].set_xlim(start, finish)
        axis[1].set_ylim(0, max_weight_penalty)
        axis[1].plot(
            rmean(log.average_weights_penalty_per_recording.interpolate()[indices], window),
                    label=name)
        axis[1].legend(loc='best')

    f, axis = pyplot.subplots(1, 2)
    f.set_size_inches((15, 5))
    for name in names:
        log = models['df_' + name]
        axis[0].plot(log.valid_per.interpolate()[indices],
                  label=name)
        axis[1].set_xlim(start, finish)
        axis[0].set_ylim(0, max_per)
        axis[0].legend(loc='best')
    for name in names:
        log = models['df_' + name]
        axis[1].plot(
            rmean(log.average_total_gradient_norm.interpolate()[indices], window),
                    label=name)
        axis[1].set_xlim(start, finish)
        axis[1].set_ylim(0, 500)
        axis[1].legend(loc='best')


def show_alignment(weights, transcription,
                   bos_symbol=False, energies=None,
                   **kwargs):
    f = pyplot.figure(figsize=(15, 0.20 * len(transcription)))
    ax = f.gca()
    ax.matshow(weights, aspect='auto', **kwargs)
    ax.set_yticks((1 if bos_symbol else 0) + numpy.arange(len(transcription)))
    ax.set_yticklabels(transcription)
    pyplot.show()

    if energies is not None:
        pyplot.matshow(energies, **kwargs)
        pyplot.colorbar()
        pyplot.show()


def wav_player(data, rate):
    """ will display html 5 player for compatible browser

    The browser need to know how to play wav through html5.

    there is no autoplay to prevent file playing when the browser opens

    Adapted from SciPy.io.
    """

    buffer = StringIO.StringIO()
    buffer.write(b'RIFF')
    buffer.write(b'\x00\x00\x00\x00')
    buffer.write(b'WAVE')

    buffer.write(b'fmt ')
    if data.ndim == 1:
        noc = 1
    else:
        noc = data.shape[1]
    bits = data.dtype.itemsize * 8
    sbytes = rate*(bits // 8)*noc
    ba = noc * (bits // 8)
    buffer.write(struct.pack('<ihHIIHH', 16, 1, noc, rate, sbytes, ba, bits))

    # data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<i', data.nbytes))

    if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and sys.byteorder == 'big'):
        data = data.byteswap()

    buffer.write(data.tostring())
    #    return buffer.getvalue()
    # Determine file size and place it in correct
    #  position at start of the file.
    size = buffer.tell()
    buffer.seek(4)
    buffer.write(struct.pack('<i', size-8))

    val = buffer.getvalue()

    src = """
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Simple Test</title>
    </head>

    <body>
    <audio controls="controls" style="width:600px" >
      <source controls src="data:audio/wav;base64,{base64}" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>
    </body>
    """.format(base64=base64.encodestring(val))
    display(HTML(src))
