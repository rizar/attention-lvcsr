from __future__ import print_function
import logging
import pprint
import math
import os
import functools
import cPickle

import numpy
import theano
from numpy.testing import assert_allclose
from theano import tensor
from blocks.bricks import Tanh, MLP, Brick, application
from blocks.bricks.recurrent import (
    SimpleRecurrent, GatedRecurrent, LSTM, Bidirectional)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, LinearReadout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.dump import load_parameter_values
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
from blocks.search import BeamSearch
from fuel.transformers import Mapping, Padding, ForceFloatX, Batch
from fuel.schemes import SequentialScheme, ConstantScheme

from lvsr.datasets import TIMIT
from lvsr.preprocessing import log_spectrogram, Normalization
from lvsr.expressions import monotonicity_penalty, entropy, weights_std
from lvsr.error_rate import wer
from lvsr.attention import ShiftPredictor, HybridAttention

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

    stream = dataset.get_example_stream()
    stream = Mapping(
        stream, functools.partial(apply_preprocessing,
                                       log_spectrogram))
    if normalization:
        stream = normalization.wrap_stream(stream)
    if not batch_size:
        return stream

    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))
    stream = Padding(stream)
    stream = Mapping(
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
            biases_init='Constant(0)',
            attention_type='content',
            use_states_for_readout=False),
        data=Config())


class PhonemeRecognizerBrick(Brick):

    def __init__(self, num_features, num_phonemes,
                 dim_dec, dim_bidir, dims_bottom,
                 enc_transition, dec_transition,
                 rec_weights_init,
                 weights_init, biases_init,
                 use_states_for_readout,
                 attention_type,
                 shift_predictor_dims=None, max_left=None, max_right=None, **kwargs):
        super(PhonemeRecognizerBrick, self).__init__(**kwargs)

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

        # Choose attention mechanism according to the configuration
        content_attention = SequenceContentAttention(
            state_names=transition.apply.states,
            attended_dim=2 * dim_bidir, match_dim=dim_dec,
            name="cont_att")
        if attention_type != "content":
            predictor = MLP([Tanh(), None],
                            [None] + shift_predictor_dims + [None],
                            name="predictor")
            location_attention = ShiftPredictor(
                state_names=transition.apply.states,
                max_left=max_left, max_right=max_right,
                predictor=predictor,
                attended_dim=2 * dim_bidir,
                name="loc_att")
            hybrid_attention = HybridAttention(
                state_names=transition.apply.states,
                attended_dim=2 * dim_bidir, match_dim=dim_dec,
                location_attention=location_attention,
                name="hybrid_att")
        if attention_type == "content":
            attention = content_attention
        elif attention_type == "location":
            attention = location_attention
        elif attention_type == "hybrid":
            attention = hybrid_attention

        readout = LinearReadout(
            readout_dim=num_phonemes,
            source_names=(transition.apply.states if use_states_for_readout else [])
                + [attention.take_glimpses.outputs[0]],
            emitter=SoftmaxEmitter(name="emitter"),
            feedback_brick=LookupFeedback(num_phonemes, dim_dec),
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
                                    as_dict=True),
                    mask=recordings_mask)),
            attended_mask=recordings_mask)

    @application
    def generate(self, recordings):
        return self.generator.generate(
            n_steps=recordings.shape[0], batch_size=recordings.shape[1],
            attended=self.encoder.apply(
                **dict_union(self.fork.apply(self.bottom.apply(recordings),
                             as_dict=True))),
            attended_mask=tensor.ones_like(recordings[:, :, 0]))


class PhonemeRecognizer(object):

    def __init__(self, brick):
        self.brick = brick

        self.recordings = tensor.tensor3("recordings")
        self.recordings_mask = tensor.matrix("recordings_mask")
        self.labels = tensor.lmatrix("labels")
        self.labels_mask = tensor.matrix("labels_mask")
        self.single_recording = tensor.matrix("single_recording")
        self.single_transcription = tensor.lvector("single_transcription")

    def load_params(self, path):
        generated = self.get_generate_graph()
        Model(generated[1]).set_param_values(load_parameter_values(path))

    def get_generate_graph(self):
        return self.brick.generate(self.recordings)

    def get_cost_graph(self, batch=True):
        if batch:
            return self.brick.cost(
                       self.recordings, self.recordings_mask,
                       self.labels, self.labels_mask)
        recordings = self.single_recording[:, None, :]
        labels = self.single_transcription[:, None]
        return self.brick.cost(
            recordings, tensor.ones_like(recordings[:, :, 0]),
            labels, None)

    def analyze(self, *args, **kwargs):
        if not hasattr(self, "_analyze"):
            cost = self.get_cost_graph(batch=False)
            cg = ComputationGraph(cost)
            weights, = VariableFilter(
                bricks=[self.brick.generator], name="weights")(cg)
            self._analyze = theano.function(
                [self.single_recording, self.single_transcription],
                [cost[:, 0], weights[:, 0, :]])
        return self._analyze(*args, **kwargs)


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
        root_path, extension = os.path.splitext(save_path)

        # Build the bricks
        assert not use_old
        recognizer = PhonemeRecognizerBrick(
            129, TIMIT.num_phonemes, name="recognizer", **config["net"])
        recognizer.initialize()

        # Build the cost computation graph
        recordings = tensor.tensor3("recordings")
        recordings_mask = tensor.matrix("recordings_mask")
        labels = tensor.lmatrix("labels")
        labels_mask = tensor.matrix("labels_mask")
        batch_cost = recognizer.cost(
            recordings, recordings_mask, labels, labels_mask).sum()
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
        mean_attended = named_copy(abs(attended).mean(),
                                   "mean_attended")
        mean_bottom_output = named_copy(abs(bottom_output).mean(),
                                        "mean_bottom_output")
        weights_penalty = aggregation.mean(
            named_copy(monotonicity_penalty(weights, labels_mask),
                       "weights_penalty_per_recording"),
            batch_size)
        weights_entropy = aggregation.mean(
            named_copy(entropy(weights, labels_mask),
                       "weights_entropy_per_phoneme"),
            labels_mask.sum())
        mask_density = named_copy(labels_mask.mean(),
                                  "mask_density")

        # Define the training algorithm.
        algorithm = GradientDescent(
            cost=cost, params=cg.parameters,
            step_rule=CompositeRule([StepClipping(100.0),
                                     Scale(0.01)]))

        observables = [
            cost, cost_per_phoneme,
            min_energy, max_energy,
            mean_attended, mean_bottom_output,
            weights_penalty, weights_entropy,
            batch_size, max_recording_length, max_num_phonemes, mask_density,
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
            build_stream(TIMIT("valid"), 100, **config["data"]), prefix="valid",
            before_first_epoch=True, on_resumption=True,
            after_every_epoch=True)
        main_loop = MainLoop(
            model=model,
            data_stream=build_stream(TIMIT("train"), 10, **config["data"]),
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
    elif mode == "search":
        from matplotlib import pyplot, cm

        beam_size = 10

        recognizer_brick = PhonemeRecognizerBrick(
            129, TIMIT.num_phonemes, name="recognizer", **config["net"])
        recognizer = PhonemeRecognizer(recognizer_brick)
        recognizer.load_params(save_path)

        generated = recognizer.get_generate_graph()
        samples, = VariableFilter(
            bricks=[recognizer_brick.generator], name="outputs")(
                ComputationGraph(generated[1]))
        beam_search = BeamSearch(beam_size, samples)
        beam_search.compile()

        timit = TIMIT("valid")
        stream = build_stream(timit, None, **config["data"])
        stream = ForceFloatX(stream)
        it = stream.get_epoch_iterator()
        error_sum = 0

        weights = tensor.matrix('weights')
        weight_std_func = theano.function(
            [weights], [weights_std(weights.dimshuffle(0, 'x', 1))])

        for number, data in enumerate(it):
            print("Utterance", number)

            input_ = numpy.tile(data[0], (beam_size, 1, 1)).transpose(1, 0, 2)
            outputs, search_costs = beam_search.search(
                {recognizer.recordings: input_}, 4, input_.shape[0] / 2,
                ignore_first_eol=True)

            recognized = timit.decode(outputs[0])
            groundtruth = timit.decode(data[1])
            error = min(1, wer(groundtruth, recognized))
            error_sum += error
            print("Beam search cost:", search_costs[0])
            print(recognized)
            costs_recognized, weights_recognized = (
                recognizer.analyze(data[0], outputs[0]))
            print("Recognized cost:", costs_recognized.sum())
            print("Recognized weight std:", weight_std_func(weights_recognized)[0])
            print(groundtruth)
            costs_groundtruth, weights_groundtruth = (
                recognizer.analyze(data[0], data[1]))
            print("Groundtruth cost:", costs_groundtruth.sum())
            print("Groundtruth weight std:", weight_std_func(weights_groundtruth)[0])
            print("PER:", error)
            print("Average PER:", error_sum / (number + 1))

            #f, (ax1, ax2) = pyplot.subplots(2, 1)
            #ax1.matshow(weights_recognized, cmap=cm.gray)
            #ax2.matshow(weights_groundtruth, cmap=cm.gray)
            #pyplot.show()

            assert_allclose(search_costs[0], costs_recognized.sum(), rtol=1e-5)
