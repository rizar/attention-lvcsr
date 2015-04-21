import numpy
import cPickle

from matplotlib import pyplot
from pandas import rolling_mean as rmean

models = {}

def load_log(name):
    log = cPickle.load(open(name + "_log.pkl"))
    models["log_" + name] = log
    df = log.to_dataframe()
    models["df_" + name] = df
    print "Iterations done for {}: {}".format(name, log.status['iterations_done'])
    print "Average batch time for {} was {}".format(
        name, df.time_train_this_batch.mean())
    print "Best PER: {}".format(log.status.get('best_valid_per', '?'))


def plot(names, start=0, finish=-1, window=1, max_weight_penalty=500):
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
            ax.legend()

    f, axis = pyplot.subplots(1, 2)
    f.set_size_inches(15, 5)
    for name in names:
        log = models['df_' + name]
        axis[0].plot(
            rmean(log.average_weights_entropy_per_label.interpolate()[indices], window),
                    label=name)
        axis[0].legend(loc='best')
    for name in names:
        log = models['df_' + name]
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
        axis[0].legend(loc='best')
    for name in names:
        log = models['df_' + name]
        axis[1].plot(
            rmean(log.average_total_gradient_norm.interpolate()[indices], window),
                    label=name)
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
