from __future__ import print_function
import logging
import pprint
import math
import os
import functools
import cPickle

import numpy
import theano
from theano import tensor
from blocks.bricks import Tanh, MLP, Brick, application
from blocks.bricks.recurrent import (
    SimpleRecurrent, GatedRecurrent, LSTM, Bidirectional)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, LinearReadout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.datasets.streams import (
    DataStream, DataStreamMapping, PaddingDataStream,
    ForceFloatX, BatchDataStream)
from blocks.datasets.schemes import SequentialScheme, ConstantScheme
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule,
                               Momentum)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import SerializeMainLoop, LoadFromDump
from blocks.extensions.monitoring import (
    TrainingDataMonitoring, DataStreamMonitoring)
from blocks.extensions.plot import Plot
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.utils import named_copy, dict_union

from lvsr.datasets import TIMIT, ExampleScheme
from lvsr.preprocessing import log_spectrogram, Normalization
from lvsr.expressions import monotonicity_penalty, entropy

floatX = theano.config.floatX
logger = logging.getLogger(__name__)

def _gradient_norm_is_none(log):
    return math.isnan(log.current_row.total_gradient_norm)

def apply_preprocessing(preprocessing, example):
    recording, label = example
    return (numpy.asarray(preprocessing(recording)), label)


def switch_first_two_axes(batch):
    result = []
    for array in batch:
        if array.ndim == 2:
            result.append(array.transpose(1, 0))
        else:
            result.append(array.transpose(1, 0, 2))
    return tuple(result)


def build_stream(dataset, batch_size, normalization=None):
    if normalization:
        with open(normalization, "rb") as src:
            normalization = cPickle.load(src)

    stream = DataStream(dataset,
                        iteration_scheme=ExampleScheme(dataset.num_examples))
    stream = DataStreamMapping(
        stream, functools.partial(apply_preprocessing,
                                       log_spectrogram))
    if normalization:
        stream = normalization.wrap_stream(stream)
    if not batch_size:
        return stream

    stream = BatchDataStream(stream, iteration_scheme=ConstantScheme(10))
    stream = PaddingDataStream(stream)
    stream = DataStreamMapping(
        stream, switch_first_two_axes)
    stream = ForceFloatX(stream)
    return stream


class Config(dict):

    def __getattr__(self, name):
        return self[name]


def default_config():
    return Config(
        net=Config(
            dim_dec=100, dim_bidir=100, dims_bottom=[100],
            enc_transition='SimpleRecurrent',
            dec_transition='SimpleRecurrent',
            weights_init='IsotropicGaussian(0.1)',
            rec_weights_init='Orthogonal()',
            biases_init='Constant(0)'))


class PhonemeRecognizer(Brick):

    def __init__(self, num_features, num_phonemes,
                 dim_dec, dim_bidir, dims_bottom,
                 enc_transition, dec_transition,
                 rec_weights_init,
                 weights_init, biases_init, **kwargs):
        super(PhonemeRecognizer, self).__init__(**kwargs)

        self.rec_weights_init = eval(rec_weights_init)
        self.weights_init = eval(weights_init)
        self.biases_init = eval(biases_init)
        self.enc_transition = eval(enc_transition)
        self.dec_transition = eval(dec_transition)

        encoder = Bidirectional(self.enc_transition(
            dim=dim_bidir, activation=Tanh()))
        fork = Fork([name for name in encoder.prototype.apply.sequences
                    if name != 'mask'])
        fork.input_dim = dims_bottom[-1]
        fork.output_dims = {name: dim_bidir for name in fork.output_names}
        bottom = MLP([Tanh()] * len(dims_bottom), [num_features] + dims_bottom,
                     name="bottom")
        transition = self.dec_transition(
            dim=dim_dec, activation=Tanh(), name="transition")
        attention = SequenceContentAttention(
            state_names=transition.apply.states,
            sequence_dim=2 * dim_bidir, match_dim=dim_dec,
            name="attention")
        readout = LinearReadout(
            readout_dim=num_phonemes,
            source_names=[attention.take_glimpses.outputs[0]],
            emitter=SoftmaxEmitter(name="emitter"),
            feedbacker=LookupFeedback(num_phonemes, dim_dec),
            name="readout")
        generator = SequenceGenerator(
            readout=readout, transition=transition, attention=attention,
            name="generator")

        self.encoder = encoder
        self.fork = fork
        self.bottom = bottom
        self.generator = generator
        self.children = [encoder, fork, bottom, generator]

    def _push_initialization_config(self):
        for child in self.children:
            child.weights_init = self.weights_init
            child.biases_init = self.biases_init
        self.encoder.weights_init = self.rec_weights_init
        self.generator.push_initialization_config()
        self.generator.transition.weights_init = self.rec_weights_init

    @application
    def cost(self, recordings, recordings_mask, labels, labels_mask):
        return self.generator.cost(
            labels, labels_mask,
            attended=self.encoder.apply(
                **dict_union(
                    self.fork.apply(self.bottom.apply(recordings),
                                    return_dict=True),
                    mask=recordings_mask)),
            attended_mask=recordings_mask).sum()


def main(mode, save_path, num_batches, use_old, from_dump, config_path):
    # Experiment configuration
    config = default_config()
    if config_path:
        with open(config_path, 'rt') as config_file:
            changes = eval(config_file.read())
        def rec_update(conf, chg):
            for key in chg:
                if key in conf and isinstance(conf[key], Config):
                    rec_update(conf[key], chg[key])
                else:
                    conf[key] = chg[key]
        rec_update(config, changes)
    logging.info("Config:\n" + pprint.pformat(config))

    if mode == "init_norm":
        stream = build_stream(TIMIT("train"), None)
        normalization = Normalization(stream, "recordings")
        with open(save_path, "wb") as dst:
            cPickle.dump(normalization, dst)

    elif mode == "show_data":
        stream = build_stream(TIMIT("train"), 10, **config.data)
        pprint.pprint(next(stream.get_epoch_iterator(as_dict=True)))

    elif mode == "train":
        timit = TIMIT()
        timit_valid = TIMIT("valid")
        root_path, extension = os.path.splitext(save_path)

        # Build the bricks
        assert not use_old
        recognizer = PhonemeRecognizer(
            129, timit.num_phonemes, name="recognizer", **config["net"])
        recognizer.initialize()

        # Build the cost computation graph
        recordings = tensor.tensor3("recordings")
        recordings_mask = tensor.matrix("recordings_mask")
        labels = tensor.lmatrix("labels")
        labels_mask = tensor.matrix("labels_mask")
        batch_cost = recognizer.cost(
            recordings, recordings_mask, labels, labels_mask)
        batch_size = named_copy(recordings.shape[1], "batch_size")
        cost = aggregation.mean(batch_cost,  batch_size)
        cost.name = "sequence_log_likelihood"
        logger.info("Cost graph is built")

        # Give an idea of what's going on
        model = Model(cost)
        params = model.get_params()
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in params.items()],
                        width=120))

        cg = ComputationGraph(cost)
        r = recognizer
        # Fetch variables useful for debugging
        max_recording_length = named_copy(recordings.shape[0],
                                          "max_recording_length")
        max_num_phonemes = named_copy(labels.shape[0],
                                      "max_num_phonemes")
        cost_per_phoneme = named_copy(
            aggregation.mean(batch_cost, batch_size * max_num_phonemes),
            "phoneme_log_likelihood")
        (energies,) = VariableFilter(
            application=r.generator.readout.readout, name="output")(
                    cg.variables)
        min_energy = named_copy(energies.min(), "min_energy")
        max_energy = named_copy(energies.max(), "max_energy")
        (bottom_output,) = VariableFilter(
            application=r.bottom.apply, name="output")(cg)
        (attended,) = VariableFilter(
            application=r.generator.transition.apply, name="attended$")(cg)
        (weights,) = VariableFilter(
            application=r.generator.cost, name="weights")(cg)
        weights1 = weights[1:]
        mean_attended = named_copy(abs(attended).mean(),
                                   "mean_attended")
        mean_bottom_output = named_copy(abs(bottom_output).mean(),
                                        "mean_bottom_output")
        weights_penalty = aggregation.mean(
            named_copy(monotonicity_penalty(weights1, labels_mask),
                       "weights_penalty_per_recording"),
            batch_size)
        weights_entropy = aggregation.mean(
            named_copy(entropy(weights1, labels_mask),
                       "weights_entropy_per_phoneme"),
            labels_mask.sum())

        # Define the training algorithm.
        algorithm = GradientDescent(
            cost=cost, step_rule=CompositeRule([StepClipping(100.0),
                                                Scale(0.01)]))

        observables = [
            cost, cost_per_phoneme,
            min_energy, max_energy,
            mean_attended, mean_bottom_output,
            weights_penalty, weights_entropy,
            batch_size, max_recording_length, max_num_phonemes,
            algorithm.total_step_norm, algorithm.total_gradient_norm]
        for name, param in params.items():
            observables.append(named_copy(
                param.norm(2), name + "_norm"))
            observables.append(named_copy(
                algorithm.gradients[param].norm(2), name + "_grad_norm"))

        average = TrainingDataMonitoring(
            observables, prefix="average", every_n_batches=10)
        validation = DataStreamMonitoring(
            [cost, cost_per_phoneme],
            build_stream(timit_valid, 100, **config["data"]), prefix="valid",
            before_first_epoch=True, on_resumption=True,
            after_every_epoch=True)
        main_loop = MainLoop(
            model=model,
            data_stream=build_stream(timit, 10, **config["data"]),
            algorithm=algorithm,
            extensions=([LoadFromDump(from_dump)] if from_dump else []) +
            [Timing(),
                TrainingDataMonitoring(observables, after_every_batch=True),
                average, validation,
                FinishAfter(after_n_batches=num_batches)
                .add_condition("after_batch", _gradient_norm_is_none),
                Plot(os.path.basename(save_path),
                     [[average.record_name(cost),
                       validation.record_name(cost)],
                      [average.record_name(cost_per_phoneme)],
                      [average.record_name(algorithm.total_gradient_norm)],
                      [average.record_name(weights_entropy)]],
                     every_n_batches=10),
                SerializeMainLoop(save_path, every_n_batches=500,
                                  save_separately=["model", "log"]),
                Printing(every_n_batches=1)])
        main_loop.run()

    elif mode == "test":
        raise NotImplemented
