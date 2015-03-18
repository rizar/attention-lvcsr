from __future__ import print_function
import logging
import pprint
import math
import os
import functools
import cPickle
from collections import OrderedDict

import numpy
import theano
from numpy.testing import assert_allclose
from theano import tensor
from blocks.bricks import Tanh, MLP, Brick, application, Initializable
from blocks.bricks.recurrent import (
    SimpleRecurrent, GatedRecurrent, LSTM, Bidirectional)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph, apply_dropout
from blocks.dump import load_parameter_values
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule,
                               Momentum, RemoveNotFinite)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.monitoring import aggregation
from blocks.extensions import (
    FinishAfter, Printing, Timing, ProgressBar, SimpleExtension)
from blocks.extensions.saveload import Checkpoint, Dump
from blocks.extensions.monitoring import (
    TrainingDataMonitoring, DataStreamMonitoring)
from blocks.extensions.plot import Plot
from blocks.extensions.training import TrackTheBest
from blocks.extensions.predicates import OnLogRecord
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.utils import named_copy, dict_union
from blocks.search import BeamSearch
from blocks.select import Selector
from fuel.transformers import (
    SortMapping, Padding, ForceFloatX, Batch, Mapping, Unpack)
from fuel.schemes import SequentialScheme, ConstantScheme

from lvsr.datasets import TIMIT
from lvsr.preprocessing import log_spectrogram, Normalization
from lvsr.expressions import monotonicity_penalty, entropy, weights_std
from lvsr.error_rate import wer
from lvsr.attention import (
    ShiftPredictor, ShiftPredictor2, HybridAttention,
    SequenceContentAndCumSumAttention)

floatX = theano.config.floatX
logger = logging.getLogger(__name__)


def _length(example):
    return len(example[0])


def _gradient_norm_is_none(log):
    return math.isnan(log.current_row.total_gradient_norm)


def _should_compute_per(log):
    return log.status.epochs_done >= 20 and log.status.epochs_done % 3 == 0


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


def build_stream(dataset, batch_size, sort_k_batches=None, normalization=None):
    if normalization:
        with open(normalization, "rb") as src:
            normalization = cPickle.load(src)

    stream = dataset.get_example_stream()
    if sort_k_batches:
        assert batch_size
        stream = Batch(stream,
                       iteration_scheme=ConstantScheme(
                           batch_size * sort_k_batches))
        stream = Mapping(stream, SortMapping(_length))
        stream = Unpack(stream)

    stream = Mapping(
        stream, functools.partial(apply_preprocessing,
                                       log_spectrogram))
    if normalization:
        stream = normalization.wrap_stream(stream)
    stream = ForceFloatX(stream)
    if not batch_size:
        return stream

    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))
    stream = Padding(stream)
    stream = Mapping(
        stream, switch_first_two_axes)
    return stream


class Config(dict):

    def __getattr__(self, name):
        return self[name]

class InitList(list):
    pass


def default_config():
    return Config(
        net=Config(
            dim_dec=100, dim_bidir=100, dims_bottom=[100],
            enc_transition='SimpleRecurrent',
            dec_transition='SimpleRecurrent',
            attention_type='content',
            use_states_for_readout=False),
        regularization=Config(
            dropout=False),
        initialization=InitList([
            ('/recognizer', 'weights_init', 'IsotropicGaussian(0.1)'),
            ('/recognizer', 'biases_init', 'Constant(0.0)'),
            ('/recognizer', 'rec_weights_init', 'Orthogonal()')]),
        data=Config(
            batch_size=10,
            normalization=None
        ))


class SpeechRecognizer(Initializable):
    """Encapsulate all reusable logic.

    This class plays a few roles: (a) it's a top brick that knows
    how to combine bottom, bidirectional and recognizer network, (b)
    it has the inputs variables and can build whole computation graphs
    starting with them (c) it hides compilation of Theano functions
    and initialization of beam search. I find it simpler to have it all
    in one place for research code.

    Parameters
    ----------
    All defining the structure and the dimensions of the model. Typically
    receives everything from the "net" section of the config.

    """
    def __init__(self, num_features, num_phonemes,
                 dim_dec, dim_bidir, dims_bottom,
                 enc_transition, dec_transition,
                 use_states_for_readout,
                 attention_type,
                 shift_predictor_dims=None, max_left=None, max_right=None,
                 padding=None, **kwargs):
        super(SpeechRecognizer, self).__init__(**kwargs)
        self.rec_weights_init = None

        self.enc_transition = eval(enc_transition)
        self.dec_transition = eval(dec_transition)

        # Build the bricks
        encoder = Bidirectional(self.enc_transition(
            dim=dim_bidir, activation=Tanh()))
        fork = Fork([name for name in encoder.prototype.apply.sequences
                    if name != 'mask'])
        fork.input_dim = dims_bottom[-1]
        fork.output_dims = [dim_bidir for name in fork.output_names]
        bottom = MLP([Tanh()] * len(dims_bottom), [num_features] + dims_bottom,
                     name="bottom")
        transition = self.dec_transition(
            dim=dim_dec, activation=Tanh(), name="transition")
        # Choose attention mechanism according to the configuration
        if attention_type == "content":
            # Simple content-based attention from ala machine translation project.
            # The main deficiency: one wrongly generated phoneme ruins everything,
            # there is no alignment memory to recover from.
            attention = SequenceContentAttention(
                state_names=transition.apply.states,
                attended_dim=2 * dim_bidir, match_dim=dim_dec,
                name="cont_att")
        elif attention_type == "content_and_cumsum":
            # Like "content", but for each position cumulative sum of weights
            # from previous steps is an additional input for building match
            # vector. Supposedly a cumsum close to 0 is interpreted as
            # as a strong argument to give very low weight, which should protect
            # us from jumping backward. It is less clear how this can protect
            # from jumping too much forward. More qualitative analysis is needed.s
            attention = SequenceContentAndCumSumAttention(
                state_names=transition.apply.states,
                # `Dump` is a peculiar one, mostly needed now to save `.npz`
                # files in addition to pickles. There is #474, where we discuss
                # the best way to get rid of it.
                attended_dim=2 * dim_bidir, match_dim=dim_dec,
                name="cont_att")
        elif attention_type == "hybrid":
            # Like "content", but with an additional location-based attention
            # mechanism. It takes the current state as input and predicts
            # good shifts from the current position (where the current position
            # is a truncated to an integer expected position). This predictions
            # in the forms of energies for each position ares added to the energies
            # produced by the content-based attention. In fact only for shifts
            # in the range [-max_left; max_right] the energies are computed,
            # for other shifts padding is used. Setting padding to a small value
            # like -10 acts as forcing the network to operate within a certain
            # window near its current position and significantly speeds up
            # training.
            predictor = MLP([Tanh(), None],
                            [None] + shift_predictor_dims + [None],
                            name="predictor")
            location_attention = ShiftPredictor(
                state_names=transition.apply.states,
                max_left=max_left, max_right=max_right, padding=padding,
                predictor=predictor,
                attended_dim=2 * dim_bidir,
                name="loc_att")
            attention = HybridAttention(
                state_names=transition.apply.states,
                attended_dim=2 * dim_bidir, match_dim=dim_dec,
                location_attention=location_attention,
                name="hybrid_att")
        elif attention_type == "hybrid2":
            # An attempt to replicate gating mechnanism by Jan. Crashes
            # when log-likelihood is about 50, further investigations
            # are needed.
            predictor = MLP([Tanh(), None],
                            [None] + shift_predictor_dims + [None],
                            name="predictor")
            location_attention = ShiftPredictor2(
                state_names=transition.apply.states,
                predictor=predictor, attended_dim=2 * dim_bidir,
                name="loc_att")
            attention = HybridAttention(
                state_names=transition.apply.states,
                attended_dim=2 * dim_bidir, match_dim=dim_dec,
                location_attention=location_attention,
                name="hybrid_att")
        readout = Readout(
            readout_dim=num_phonemes,
            source_names=(transition.apply.states if use_states_for_readout else [])
                + [attention.take_glimpses.outputs[0]],
            emitter=SoftmaxEmitter(name="emitter"),
            feedback_brick=LookupFeedback(num_phonemes, dim_dec),
            name="readout")
        generator = SequenceGenerator(
            readout=readout, transition=transition, attention=attention,
            name="generator")

        # Remember child bricks
        self.encoder = encoder
        self.fork = fork
        self.bottom = bottom
        self.generator = generator
        self.children = [encoder, fork, bottom, generator]

        # Create input variables
        self.recordings = tensor.tensor3("recordings")
        self.recordings_mask = tensor.matrix("recordings_mask")
        self.labels = tensor.lmatrix("labels")
        self.labels_mask = tensor.matrix("labels_mask")
        self.single_recording = tensor.matrix("single_recording")
        self.single_transcription = tensor.lvector("single_transcription")

    def _push_initialization_config(self):
        super(SpeechRecognizer, self)._push_initialization_config()
        if self.rec_weights_init:
            self.encoder.weights_init = self.rec_weights_init
            self.generator.transition.transition.weights_init = self.rec_weights_init

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

    def load_params(self, path):
        generated = self.get_generate_graph()
        Model(generated[1]).set_param_values(load_parameter_values(path))

    def get_generate_graph(self):
        return self.generate(self.recordings)

    def get_cost_graph(self, batch=True):
        if batch:
            return self.cost(
                       self.recordings, self.recordings_mask,
                       self.labels, self.labels_mask)
        recordings = self.single_recording[:, None, :]
        labels = self.single_transcription[:, None]
        return self.cost(
            recordings, tensor.ones_like(recordings[:, :, 0]),
            labels, None)

    def analyze(self, recording, transcription):
        if not hasattr(self, "_analyze"):
            cost = self.get_cost_graph(batch=False)
            cg = ComputationGraph(cost)
            weights, = VariableFilter(
                bricks=[self.brick.generator], name="weights")(cg)
            self._analyze = theano.function(
                [self.single_recording, self.single_transcription],
                [cost[:, 0], weights[:, 0, :]])
        return self._analyze(recording, transcription)

    def init_beam_search(self, beam_size):
        self.beam_size = beam_size
        generated = self.get_generate_graph()
        samples, = VariableFilter(
            bricks=[self.generator], name="outputs")(
                ComputationGraph(generated[1]))
        self._beam_search = BeamSearch(beam_size, samples)
        self._beam_search.compile()

    def beam_search(self, recording):
        input_ = numpy.tile(recording, (self.beam_size, 1, 1)).transpose(1, 0, 2)
        outputs, search_costs = self._beam_search.search(
            {self.recordings: input_}, 4, input_.shape[0] / 3,
            ignore_first_eol=True)
        return outputs, search_costs


class PERExtension(SimpleExtension):

    def __init__(self, recognizer, dataset, stream, **kwargs):
        self.recognizer = recognizer
        self.dataset = dataset
        self.stream = stream
        super(PERExtension, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        logger.info("PER computing started")
        error_sum = 0.0
        num_examples = 0.0
        for data in self.stream.get_epoch_iterator():
            outputs, search_costs = self.recognizer.beam_search(data[0])
            recognized = self.dataset.decode(outputs[0])
            groundtruth = self.dataset.decode(data[1])
            error = min(1, wer(groundtruth, recognized))
            error_sum += error
            num_examples += 1
            mean_error = error_sum / num_examples
            if num_examples > 10 and mean_error > 0.8:
                mean_error = 1
                break
        self.main_loop.log.current_row.per = mean_error
        logger.info("PER computing done")


class IPDB(SimpleExtension):

    def do(self, *args, **kwargs):
        import ipdb; ipdb.set_trace()


def main(mode, save_path, num_batches, config_path):
    # Experiment configuration
    config = default_config()
    if config_path:
        with open(config_path, 'rt') as config_file:
            changes = eval(config_file.read())
        def rec_update(conf, chg):
            for key in chg:
                if isinstance(conf.get(key), Config):
                    rec_update(conf[key], chg[key])
                elif isinstance(conf.get(key), InitList):
                    conf[key].extend(chg[key])
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

        # Build the main brick and initialize all parameters.
        recognizer = SpeechRecognizer(
            129, TIMIT.num_phonemes, name="recognizer", **config["net"])
        for brick_path, attribute, value in config['initialization']:
            brick, = Selector(recognizer).select(brick_path).bricks
            setattr(brick, attribute, eval(value))
            brick.push_initialization_config()
        recognizer.initialize()

        batch_cost = recognizer.get_cost_graph().sum()
        batch_size = named_copy(recognizer.recordings.shape[1], "batch_size")
        cost = aggregation.mean(batch_cost,  batch_size)
        cost.name = "sequence_log_likelihood"
        logger.info("Cost graph is built")

        # Model is weird class, we spend lots of time arguing with Bart
        # what it should be. However it can already nice things, e.g.
        # one extract all the parameters from the computation graphs
        # and give them hierahical names. This help to notice when a
        # because of some bug a parameter is not in the computation
        # graph.
        model = Model(cost)
        params = model.get_params()
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in params.items()],
                        width=120))
        logger.info("Initialization schemes for all bricks.\n\n"
            "Works well only in my branch with __repr__ added to all them,\n",
            "there is an issue #463 in Blocks to do that properly.")
        def show_init_scheme(cur):
            result = dict()
            for attr in ['weights_init', 'biases_init']:
                if hasattr(cur, attr):
                    result[attr] = getattr(cur, attr)
            for child in cur.children:
                result[child.name] = show_init_scheme(child)
            return result
        logger.info("Initialization:" +
                    pprint.pformat(show_init_scheme(recognizer)))

        cg = ComputationGraph(cost)
        r = recognizer
        # Fetch variables useful for debugging
        max_recording_length = named_copy(r.recordings.shape[0],
                                          "max_recording_length")
        max_num_phonemes = named_copy(r.labels.shape[0],
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
            named_copy(monotonicity_penalty(weights, r.labels_mask),
                       "weights_penalty_per_recording"),
            batch_size)
        weights_entropy = aggregation.mean(
            named_copy(entropy(weights, r.labels_mask),
                       "weights_entropy_per_phoneme"),
            r.labels_mask.sum())
        mask_density = named_copy(r.labels_mask.mean(),
                                  "mask_density")

        # Regularization.
        regularized_cost = cost
        if config['regularization']['dropout']:
            logger.info('apply dropout')
            cg_dropout = apply_dropout(cg, [bottom_output], 0.5)
            regularized_cost = named_copy(cg_dropout.outputs[0],
                                          'dropout_' + cost.name)

        # Define the training algorithm.
        algorithm = GradientDescent(
            cost=regularized_cost, params=cg.parameters,
            step_rule=CompositeRule([StepClipping(100.0),
                                     Scale(0.01),
                                     RemoveNotFinite(0.0)]))

        observables = [
            cost, cost_per_phoneme,
            min_energy, max_energy,
            mean_attended, mean_bottom_output,
            weights_penalty, weights_entropy,
            batch_size, max_recording_length, max_num_phonemes, mask_density,
            algorithm.total_step_norm, algorithm.total_gradient_norm]
        if cost != regularized_cost:
            observables = [regularized_cost] + observables
        # More variables for debugging: some of them can be added only
        # after the `algorithm` object is created.
        for name, param in params.items():
            observables.append(named_copy(
                param.norm(2), name + "_norm"))
            observables.append(named_copy(
                algorithm.gradients[param].norm(2), name + "_grad_norm"))

        # Build main loop.
        timit_valid = TIMIT("valid")
        every_batch = TrainingDataMonitoring(
            [algorithm.total_gradient_norm], after_every_batch=True)
        average = TrainingDataMonitoring(
            observables, prefix="average", every_n_batches=10)
        validation = DataStreamMonitoring(
            [cost, cost_per_phoneme, weights_entropy, weights_penalty],
            build_stream(timit_valid, **config["data"]), prefix="valid",
            before_first_epoch=True, on_resumption=True,
            after_every_epoch=True)
        recognizer.init_beam_search(10)
        per = PERExtension(
            recognizer, timit_valid,
            build_stream(timit_valid, None,
                  normalization=config["data"]["normalization"]),
            before_first_epoch=True, every_n_epochs=3)
        track_the_best = TrackTheBest('per').set_conditions(
            before_first_epoch=True, after_every_epoch=True)
        main_loop = MainLoop(
            model=model,
            data_stream=build_stream(
                TIMIT("train"), **config["data"]),
            algorithm=algorithm,
            extensions=([
                Timing(),
                every_batch, average, validation, per, track_the_best,
                FinishAfter(after_n_batches=num_batches)
                .add_condition("after_batch", _gradient_norm_is_none),
                Plot(os.path.basename(save_path),
                     [[average.record_name(cost),
                       validation.record_name(cost)]
                       + ([average.record_name(regularized_cost)]
                           if cost != regularized_cost else []),
                      [average.record_name(algorithm.total_gradient_norm)],
                      ['per'],
                      [average.record_name(weights_entropy),
                       validation.record_name(weights_entropy)],
                      [average.record_name(weights_penalty),
                       validation.record_name(weights_penalty)]],
                     every_n_batches=10),
                Checkpoint(save_path,
                           before_first_epoch=True, after_every_epoch=True,
                           save_separately=["model", "log"])
                .add_condition(
                    'after_epoch',
                    OnLogRecord(track_the_best.notification_name),
                    (root_path + "_best" + extension,))
                .add_condition(
                    'before_epoch',
                    OnLogRecord(track_the_best.notification_name),
                    (root_path + "_best" + extension,)),
                ProgressBar(),
                Printing(every_n_batches=1)]))
        main_loop.run()
    elif mode == "search":
        recognizer_brick, = cPickle.load(open(save_path)).get_top_bricks()
        recognizer = SpeechRecognizer(recognizer_brick)
        recognizer.init_beam_search(10)

        timit = TIMIT("valid")
        conf = config["data"]
        conf['batch_size'] = conf['sort_k_batches'] = None
        stream = build_stream(timit, **conf)
        it = stream.get_epoch_iterator()

        weights = tensor.matrix('weights')
        weight_statistics = theano.function(
            [weights],
            [weights_std(weights.dimshuffle(0, 'x', 1)),
             monotonicity_penalty(weights.dimshuffle(0, 'x', 1))])

        error_sum = 0
        for number, data in enumerate(it):
            print("Utterance", number)

            outputs, search_costs = recognizer.beam_search(data[0])
            recognized = timit.decode(outputs[0])
            groundtruth = timit.decode(data[1])
            costs_recognized, weights_recognized = (
                recognizer.analyze(data[0], outputs[0]))
            costs_groundtruth, weights_groundtruth = (
                recognizer.analyze(data[0], data[1]))
            weight_std_recognized, mono_penalty_recognized = weight_statistics(
                weights_recognized)
            weight_std_groundtruth, mono_penalty_groundtruth = weight_statistics(
                weights_groundtruth)
            error = min(1, wer(groundtruth, recognized))
            error_sum += error

            print("Beam search cost:", search_costs[0])
            print(recognized)
            print("Recognized cost:", costs_recognized.sum())
            print("Recognized weight std:", weight_std_recognized)
            print("Recognized monotonicity penalty:", mono_penalty_recognized)
            print(groundtruth)
            print("Groundtruth cost:", costs_groundtruth.sum())
            print("Groundtruth weight std:", weight_std_groundtruth)
            print("Groundtruth monotonicity penalty:", mono_penalty_groundtruth)
            print("PER:", error)
            print("Average PER:", error_sum / (number + 1))

            assert_allclose(search_costs[0], costs_recognized.sum(), rtol=1e-5)
