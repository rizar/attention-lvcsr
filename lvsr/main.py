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
from blocks.bricks import Tanh, MLP, Brick, application, Initializable, Identity
from blocks.bricks.recurrent import (
    SimpleRecurrent, GatedRecurrent, LSTM, Bidirectional)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from blocks.dump import load_parameter_values
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule,
                               Momentum, RemoveNotFinite)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.monitoring import aggregation
from blocks.monitoring.aggregation import MonitoredQuantity
from blocks.extensions import (
    FinishAfter, Printing, Timing, ProgressBar, SimpleExtension)
from blocks.extensions.saveload import Checkpoint, Dump
from blocks.extensions.monitoring import (
    TrainingDataMonitoring, DataStreamMonitoring)
from blocks.extensions.plot import Plot
from blocks.extensions.training import TrackTheBest
from blocks.extensions.predicates import OnLogRecord
from blocks.log import TrainingLog
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.utils import named_copy, dict_union
from blocks.search import BeamSearch
from blocks.select import Selector
from fuel.schemes import (
    SequentialScheme, ConstantScheme, ShuffledExampleScheme)
from fuel.streams import DataStream
from fuel.transformers import (
    SortMapping, Padding, ForceFloatX, Batch, Mapping, Unpack,
    Filter)
from picklable_itertools.extras import equizip

from lvsr.attention import (
    ShiftPredictor, ShiftPredictor2, HybridAttention,
    SequenceContentAndConvAttention,
    SequenceContentAndCumSumAttention)
from lvsr.config import prototype, read_config
from lvsr.datasets import TIMIT2, WSJ
from lvsr.expressions import monotonicity_penalty, entropy, weights_std
from lvsr.extensions import CGStatistics, CodeVersion, AdaptiveClipping
from lvsr.error_rate import wer
from lvsr.preprocessing import log_spectrogram, Normalization


floatX = theano.config.floatX
logger = logging.getLogger(__name__)


def _length(example):
    return len(example[0])


class _LengthFilter(object):

    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, example):
        return len(example[0]) <= self.max_length


def _gradient_norm_is_none(log):
    return math.isnan(log.current_row.get('total_gradient_norm', 0))


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


class _AddEosLabel(object):

    def __init__(self, eos_label):
        self.eos_label = eos_label

    def __call__(self, example):
        return (example[0], [self.eos_label] + list(example[1]) + [self.eos_label])


class _MergeKFrames(object):

    def __init__(self, k_frames):
        self.k_frames = k_frames

    def __call__(self, example):
        features = example[0]
        assert features.ndim == 2
        new_length = features.shape[0] / self.k_frames
        new_width = features.shape[1] * self.k_frames
        remainder = features.shape[0] % self.k_frames
        if remainder:
            features = features[:-remainder]
        new_features = features.reshape((new_length, new_width))
        return (new_features, example[1])


class Data(object):

    def __init__(self, dataset, batch_size, sort_k_batches,
                 max_length, normalization, merge_k_frames=None,
                 # Need these options to handle old TIMIT models
                 add_eos=True, eos_label=None):
        if normalization:
            with open(normalization, "rb") as src:
                normalization = cPickle.load(src)

        self.dataset = dataset
        self.normalization = normalization
        self.batch_size = batch_size
        self.sort_k_batches = sort_k_batches
        self.merge_k_frames = merge_k_frames
        self.max_length = max_length
        self.add_eos = add_eos
        self._eos_label = eos_label

        self.dataset_cache = {}

    @property
    def num_labels(self):
        if self.dataset == "TIMIT":
            return self.get_dataset("train").num_phonemes
        else:
            return self.get_dataset("train").num_characters

    @property
    def num_features(self):
        return 129 * (self.merge_k_frames if self.merge_k_frames else 1)

    @property
    def eos_label(self):
        if self._eos_label:
            return self._eos_label
        if self.dataset == "TIMIT":
            return TIMIT2.eos_label
        else:
            return 124

    def get_dataset(self, part):
        if not part in self.dataset_cache:
            if self.dataset == "TIMIT":
                name_mapping = {"train": "train",
                                "valid": "dev",
                                "test": "test"}
                self.dataset_cache[part] = TIMIT2(name_mapping[part],
                                                  add_eos=self.add_eos)
            elif self.dataset == "WSJ":
                name_mapping = {"train": "train_si284", "valid": "test_dev93"}
                self.dataset_cache[part] = WSJ(name_mapping[part])
            else:
                raise ValueError
        return self.dataset_cache[part]

    def get_stream(self, part, batches=True, shuffle=True):
        dataset = self.get_dataset(part)
        stream = (DataStream(dataset,
                             iteration_scheme=ShuffledExampleScheme(dataset.num_examples))
                  if shuffle
                  else dataset.get_example_stream())
        if self.dataset == "WSJ":
            stream = Mapping(stream, _AddEosLabel(self.eos_label))
        if self.max_length:
            stream = Filter(stream, _LengthFilter(self.max_length))
        if self.sort_k_batches and batches:
            stream = Batch(stream,
                        iteration_scheme=ConstantScheme(
                            self.batch_size * self.sort_k_batches))
            stream = Mapping(stream, SortMapping(_length))
            stream = Unpack(stream)

        stream = Mapping(
            stream, functools.partial(apply_preprocessing,
                                      log_spectrogram))
        if self.normalization:
            stream = self.normalization.wrap_stream(stream)
        if self.merge_k_frames:
            stream = Mapping(
                stream, _MergeKFrames(self.merge_k_frames))
        stream = ForceFloatX(stream)
        if not batches:
            return stream

        stream = Batch(stream, iteration_scheme=ConstantScheme(self.batch_size))
        stream = Padding(stream)
        stream = Mapping(
            stream, switch_first_two_axes)
        return stream


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
    def __init__(self, eos_label,
                 num_features, num_phonemes,
                 dim_dec, dim_bidir, dims_bottom,
                 enc_transition, dec_transition,
                 use_states_for_readout,
                 attention_type,
                 dims_top=None,
                 shift_predictor_dims=None, max_left=None, max_right=None,
                 padding=None, prior=None, conv_n=None,
                 **kwargs):
        super(SpeechRecognizer, self).__init__(**kwargs)
        self.eos_label = eos_label
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
        top = (MLP([Tanh()], [2 * dim_bidir] + dims_top + [2 * dim_bidir], name="top")
               if dims_top is not None else Identity())
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
        elif attention_type == "content_and_conv":
            attention = SequenceContentAndConvAttention(
                state_names=transition.apply.states,
                conv_n=conv_n,
                attended_dim=2 * dim_bidir, match_dim=dim_dec,
                prior=prior,
                name="conv_att")
        elif attention_type == "content_and_cumsum":
            attention = SequenceContentAndCumSumAttention(
                state_names=transition.apply.states,
                attended_dim=2 * dim_bidir, match_dim=dim_dec,
                prior=prior, name="cumsum_att")
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
            emitter=SoftmaxEmitter(initial_output=num_phonemes, name="emitter"),
            feedback_brick=LookupFeedback(num_phonemes + 1, dim_dec),
            name="readout")
        generator = SequenceGenerator(
            readout=readout, transition=transition, attention=attention,
            name="generator")

        # Remember child bricks
        self.encoder = encoder
        self.fork = fork
        self.bottom = bottom
        self.top = top
        self.generator = generator
        self.children = [encoder, fork, top, bottom, generator]

        # Create input variables
        self.recordings = tensor.tensor3("recordings")
        self.recordings_mask = tensor.matrix("recordings_mask")
        self.labels = tensor.lmatrix("labels")
        self.labels_mask = tensor.matrix("labels_mask")
        self.batch_inputs = [self.recordings, self.recordings_mask,
                             self.labels, self.labels_mask]
        self.single_recording = tensor.matrix("recordings")
        self.single_transcription = tensor.lvector("labels")

    def push_initialization_config(self):
        super(SpeechRecognizer, self).push_initialization_config()
        if self.rec_weights_init:
            self.encoder.weights_init = self.rec_weights_init
            self.encoder.push_initialization_config()
            self.generator.transition.transition.weights_init = self.rec_weights_init
            self.generator.transition.transition.push_initialization_config()

    @application
    def cost(self, recordings, recordings_mask, labels, labels_mask):
        return self.generator.cost_matrix(
            labels, labels_mask,
            attended=self.top.apply(
                self.encoder.apply(
                    **dict_union(self.fork.apply(self.bottom.apply(recordings),
                                                 as_dict=True),
                    mask=recordings_mask))),
            attended_mask=recordings_mask)

    @application
    def generate(self, recordings):
        return self.generator.generate(
            n_steps=recordings.shape[0], batch_size=recordings.shape[1],
            attended=self.top.apply(
                self.encoder.apply(
                    **dict_union(self.fork.apply(self.bottom.apply(recordings),
                                                 as_dict=True)))),
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
        """Compute cost and aligment for a recording/transcription pair."""
        if not hasattr(self, "_analyze"):
            cost = self.get_cost_graph(batch=False)
            cg = ComputationGraph(cost)
            energies = VariableFilter(
                bricks=[self.generator], name="energies")(cg)
            energies_output = [energies[0][:, 0, :] if energies
                               else tensor.zeros((self.single_transcription.shape[0],
                                                  self.single_recording.shape[0]))]
            weights, = VariableFilter(
                bricks=[self.generator], name="weights")(cg)
            self._analyze = theano.function(
                [self.single_recording, self.single_transcription],
                [cost[:, 0], weights[:, 0, :]] + energies_output)
        return self._analyze(recording, transcription)

    def init_beam_search(self, beam_size):
        """Compile beam search and set the beam size.

        See Blocks issue #500.

        """
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
            {self.recordings: input_}, self.eos_label, input_.shape[0] / 3,
            ignore_first_eol=True)
        return outputs, search_costs

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ['_analyze', '_beam_search']:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # To use bricks used on a GPU first on a CPU later
        del self.generator.readout.emitter._theano_rng


class PhonemeErrorRate(MonitoredQuantity):

    def __init__(self, recognizer, dataset, **kwargs):
        self.recognizer = recognizer
        # Will only be used to decode generated outputs,
        # which is necessary for correct scoring.
        self.dataset = dataset
        kwargs.setdefault('name', 'per')
        kwargs.setdefault('requires', [self.recognizer.single_recording,
                                       self.recognizer.single_transcription])
        super(PhonemeErrorRate, self).__init__(**kwargs)

    def initialize(self):
        self.error_sum = 0
        self.num_examples = 0

    def accumulate(self, recording, transcription):
        # Hack to avoid hopeless decoding of an untrained model
        if self.num_examples > 10 and self.mean_error > 0.8:
            self.mean_error = 1
            return
        outputs, search_costs = self.recognizer.beam_search(recording)
        recognized = self.dataset.decode(outputs[0])
        groundtruth = self.dataset.decode(transcription)
        error = min(1, wer(groundtruth, recognized))
        self.error_sum += error
        self.num_examples += 1
        self.mean_error = self.error_sum / self.num_examples

    def readout(self):
        return self.mean_error


def main(cmd_args):
    # Experiment configuration
    config = prototype
    if cmd_args.config_path:
        with open(cmd_args.config_path, 'rt') as src:
            config = read_config(src)
    config['cmd_args'] = cmd_args.__dict__
    for path, value in equizip(
            cmd_args.config_changes[::2],
            cmd_args.config_changes[1::2]):
        parts = path.split('.')
        assign_to = config
        for part in parts[:-1]:
            assign_to = assign_to[part]
        assign_to[parts[-1]] = eval(value)
    logging.info("Config:\n" + pprint.pformat(config, width=120))

    data = Data(**config['data'])

    if cmd_args.mode == "init_norm":
        stream = data.get_stream("train", batches=False, shuffle=False)
        normalization = Normalization(stream, "recordings")
        with open(cmd_args.save_path, "wb") as dst:
            cPickle.dump(normalization, dst)

    elif cmd_args.mode == "show_data":
        stream = data.get_stream("train")
        data = next(stream.get_epoch_iterator(as_dict=True))
        import IPython; IPython.embed()

    elif cmd_args.mode == "train":
        root_path, extension = os.path.splitext(cmd_args.save_path)

        # Build the main brick and initialize all parameters.
        recognizer = SpeechRecognizer(
            data.eos_label,
            data.num_features, data.num_labels,
            name="recognizer", **config["net"])
        for brick_path, attribute, value in config['initialization']:
            brick, = Selector(recognizer).select(brick_path).bricks
            setattr(brick, attribute, eval(value))
            brick.push_initialization_config()
        recognizer.initialize()

        logger.info("Initialization schemes for all bricks.\n"
            "Works well only in my branch with __repr__ added to all them,\n"
            "there is an issue #463 in Blocks to do that properly.")
        def show_init_scheme(cur):
            result = dict()
            for attr in ['weights_init', 'biases_init']:
                if hasattr(cur, attr):
                    result[attr] = getattr(cur, attr)
            for child in cur.children:
                result[child.name] = show_init_scheme(child)
            return result
        logger.info(pprint.pformat(show_init_scheme(recognizer)))

        if cmd_args.params:
            logger.info("Load parameters from " + cmd_args.params)
            recognizer.load_params(cmd_args.params)

        batch_cost = recognizer.get_cost_graph().sum()
        batch_size = named_copy(recognizer.recordings.shape[1], "batch_size")
        # Assumes constant batch size. `aggregation.mean` is not used because
        # of Blocks #514.
        cost = batch_cost / batch_size
        cost.name = "sequence_log_likelihood"
        logger.info("Cost graph is built")

        # Fetch variables useful for debugging.
        # It is important not to use any aggregation schemes here,
        # as it's currently impossible to spread the effect of
        # regularization on their variables, see Blocks #514.
        cost_cg = ComputationGraph(cost)
        r = recognizer
        (energies,) = VariableFilter(
            applications=[r.generator.readout.readout], name="output_0")(
                    cost_cg)
        (bottom_output,) = VariableFilter(
            applications=[r.bottom.apply], name="output")(
                    cost_cg)
        (attended,) = VariableFilter(
            applications=[r.generator.transition.apply], name="attended")(
                    cost_cg)
        (weights,) = VariableFilter(
            applications=[r.generator.cost_matrix], name="weights")(
                    cost_cg)
        max_recording_length = named_copy(r.recordings.shape[0],
                                         "max_recording_length")
        max_num_phonemes = named_copy(r.labels.shape[0],
                                      "max_num_phonemes")
        min_energy = named_copy(energies.min(), "min_energy")
        max_energy = named_copy(energies.max(), "max_energy")
        mean_attended = named_copy(abs(attended).mean(),
                                   "mean_attended")
        mean_bottom_output = named_copy(abs(bottom_output).mean(),
                                        "mean_bottom_output")
        weights_penalty = named_copy(monotonicity_penalty(weights, r.labels_mask),
                                     "weights_penalty")
        weights_entropy = named_copy(entropy(weights, r.labels_mask),
                                     "weights_entropy")
        mask_density = named_copy(r.labels_mask.mean(),
                                  "mask_density")
        cg = ComputationGraph([
            cost, weights_penalty, weights_entropy,
            min_energy, max_energy,
            mean_attended, mean_bottom_output,
            batch_size, max_recording_length, max_num_phonemes,
            mask_density])

        # Regularization. It is applied explicitly to all variables
        # of interest, it could not be applied to the cost only as it
        # would not have effect on auxiliary variables, see Blocks #514.
        reg_config = config['regularization']
        regularized_cg = cg
        if reg_config['dropout']:
            logger.info('apply dropout')
            regularized_cg = apply_dropout(cg, [bottom_output], 0.5)
        if reg_config['noise'] is not None:
            logger.info('apply noise')
            regularized_cg = apply_noise(cg, cg.parameters, reg_config['noise'])
        regularized_cost = regularized_cg.outputs[0]

        # Model is weird class, we spend lots of time arguing with Bart
        # what it should be. However it can already nice things, e.g.
        # one extract all the parameters from the computation graphs
        # and give them hierahical names. This help to notice when a
        # because of some bug a parameter is not in the computation
        # graph.
        model = Model(regularized_cost)
        params = model.get_params()
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in params.items()],
                        width=120))

        # Define the training algorithm.
        train_conf = config['training']
        clipping = StepClipping(train_conf['gradient_threshold'])
        clipping.threshold.name = "gradient_norm_threshold"
        algorithm = GradientDescent(
            cost=regularized_cost, params=params.values(),
            step_rule=CompositeRule([
                clipping,
                Momentum(train_conf['scale'], train_conf['momentum']),
                # Parameters are not changed at all
                # when nans are encountered.
                RemoveNotFinite(0.0)]))

        # More variables for debugging: some of them can be added only
        # after the `algorithm` object is created.
        observables = regularized_cg.outputs
        observables += [
            algorithm.total_step_norm, algorithm.total_gradient_norm,
            clipping.threshold]
        for name, param in params.items():
            observables.append(named_copy(
                param.norm(2), name + "_norm"))
            observables.append(named_copy(
                algorithm.gradients[param].norm(2), name + "_grad_norm"))

        def attach_aggregation_schemes(variables):
            # Aggregation specification has to be factored out as a separate
            # function as it has to be applied at the very last stage
            # separately to training and validation observables.
            result = []
            for var in variables:
                if var.name == 'weights_penalty':
                    result.append(named_copy(aggregation.mean(var, batch_size),
                                             'weights_penalty_per_recording'))
                elif var.name == 'weights_entropy':
                    result.append(named_copy(aggregation.mean(
                        var, recognizer.labels_mask.sum()), 'weights_entropy_per_label'))
                else:
                    result.append(var)
            return result

        # Build main loop.
        every_batch_monitoring = TrainingDataMonitoring(
            [observables[0], algorithm.total_gradient_norm,
             algorithm.total_step_norm, clipping.threshold], after_batch=True)
        average_monitoring = TrainingDataMonitoring(
            attach_aggregation_schemes(observables),
            prefix="average", every_n_batches=10)
        validation = DataStreamMonitoring(
            attach_aggregation_schemes([cost, weights_entropy, weights_penalty]),
            data.get_stream("valid"), prefix="valid",
            before_first_epoch=not cmd_args.fast_start,
            after_epoch=True)
        recognizer.init_beam_search(10)
        per = PhonemeErrorRate(recognizer, data.get_dataset("valid"))
        per_monitoring = DataStreamMonitoring(
            [per], data.get_stream("valid", batches=False, shuffle=False),
            prefix="valid").set_conditions(
                before_first_epoch=not cmd_args.fast_start, every_n_epochs=3)
        track_the_best = TrackTheBest(
            per_monitoring.record_name(per)).set_conditions(
                before_first_epoch=True, after_epoch=True)
        adaptive_clipping = AdaptiveClipping(
            algorithm.total_gradient_norm.name,
            clipping, train_conf['gradient_threshold'])

        # Save the config into the status
        log = TrainingLog()
        log.status['_config'] = config
        main_loop = MainLoop(
            model=model, log=log, algorithm=algorithm,
            data_stream=data.get_stream("train"),
            extensions=([
                Timing(after_batch=True),
                CGStatistics(),
                CodeVersion(['lvsr']),
                every_batch_monitoring, average_monitoring,
                validation, per_monitoring,
                track_the_best,
                adaptive_clipping,
                FinishAfter(after_n_batches=cmd_args.num_batches)
                .add_condition("after_batch", _gradient_norm_is_none),
                # Live plotting: requires launching `bokeh-server`
                # and allows to see what happens online.
                Plot(os.path.basename(cmd_args.save_path),
                     [# Plot 1: training and validation costs
                      [average_monitoring.record_name(regularized_cost),
                       validation.record_name(cost)],
                      # Plot 2: gradient norm,
                      [average_monitoring.record_name(algorithm.total_gradient_norm),
                       average_monitoring.record_name(clipping.threshold)],
                      # Plot 3: phoneme error rate
                      [per_monitoring.record_name(per)],
                      # Plot 4: training and validation mean weight entropy
                      [average_monitoring._record_name('weights_entropy_per_label'),
                       validation._record_name('weights_entropy_per_label')],
                      # Plot 5: training and validation monotonicity penalty
                      [average_monitoring._record_name('weights_penalty_per_recording'),
                       validation._record_name('weights_penalty_per_recording')]],
                     every_n_batches=10),
                Checkpoint(cmd_args.save_path,
                           before_first_epoch=not cmd_args.fast_start, after_epoch=True,
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
    elif cmd_args.mode == "search":
        # Try to guess if just parameters or the whole model was given.
        if cmd_args.save_path.endswith('.pkl'):
            recognizer, = cPickle.load(
                open(cmd_args.save_path)).get_top_bricks()
        elif cmd_args.save_path.endswith('.npz'):
            recognizer = SpeechRecognizer(
                data.eos_label, 29, WSJ.num_characters, name="recognizer", **config["net"])
            recognizer.load_params(cmd_args.save_path)
        recognizer.init_beam_search(cmd_args.beam_size)

        dataset = data.get_dataset(cmd_args.part)
        stream = data.get_stream(cmd_args.part, batches=False, shuffle=False)
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
            recognized = dataset.decode(
                outputs[0], **({'old_labels': True} if cmd_args.old_labels else {}))
            groundtruth = dataset.decode(data[1])
            costs_recognized, weights_recognized, _ = (
                recognizer.analyze(data[0], outputs[0]))
            costs_groundtruth, weights_groundtruth, _ = (
                recognizer.analyze(data[0], data[1]))
            weight_std_recognized, mono_penalty_recognized = weight_statistics(
                weights_recognized)
            weight_std_groundtruth, mono_penalty_groundtruth = weight_statistics(
                weights_groundtruth)
            error = min(1, wer(groundtruth, recognized))
            error_sum += error

            print("Beam search cost:", search_costs[0])
            print("Recognizer:", recognized)
            print("Recognized cost:", costs_recognized.sum())
            print("Recognized weight std:", weight_std_recognized)
            print("Recognized monotonicity penalty:", mono_penalty_recognized)
            print("Groundtruth:", groundtruth)
            print("Groundtruth cost:", costs_groundtruth.sum())
            print("Groundtruth weight std:", weight_std_groundtruth)
            print("Groundtruth monotonicity penalty:", mono_penalty_groundtruth)
            print("PER:", error)
            print("Average PER:", error_sum / (number + 1))

            # assert_allclose(search_costs[0], costs_recognized.sum(), rtol=1e-5)
