from __future__ import print_function
import logging
import pprint
import math
import os
import functools
import cPickle
import cPickle as pickle
import sys
import copy
from collections import OrderedDict

import numpy
import theano
import fuel
from numpy.testing import assert_allclose
from theano import tensor
from blocks.bricks import (
    Tanh, MLP, Brick, application,
    Initializable, Identity, Rectifier, Maxout,
    Sequence, Bias, Linear)
from blocks.bricks.recurrent import (
    SimpleRecurrent, GatedRecurrent, LSTM, Bidirectional, BaseRecurrent,
    RecurrentStack)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback,
    AbstractFeedback)
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule,
                               Momentum, RemoveNotFinite, AdaDelta,
                               Restrict, VariableClipping)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.monitoring import aggregation
from blocks.monitoring.aggregation import MonitoredQuantity
from blocks.theano_expressions import l2_norm
from blocks.extensions import (
    FinishAfter, Printing, Timing, ProgressBar, SimpleExtension,
    TrainingExtension, saveload, PrintingFilterList)
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.monitoring import (
    TrainingDataMonitoring, DataStreamMonitoring)
from blocks.extras.extensions.plot import Plot
from blocks.extensions.training import TrackTheBest
from blocks.extensions.predicates import OnLogRecord
from blocks.log import TrainingLog
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.roles import OUTPUT, WEIGHT
from blocks.utils import named_copy, dict_union, check_theano_variable,\
    reraise_as
from blocks.search import BeamSearch
from blocks.select import Selector
from blocks.serialization import load_parameter_values
from fuel.schemes import (
    SequentialScheme, ConstantScheme, ShuffledExampleScheme)
from fuel.streams import DataStream
from fuel.transformers import (
    SortMapping, Padding, ForceFloatX, Batch, Mapping, Unpack,
    Filter, FilterSources)
from picklable_itertools.extras import equizip

import lvsr.datasets.wsj
from lvsr.datasets.h5py import H5PYAudioDataset
from lvsr.attention import (
    SequenceContentAndConvAttention)
from lvsr.bricks import RecurrentWithFork
from lvsr.config import prototype, read_config
from lvsr.datasets import TIMIT2, WSJ
from lvsr.expressions import (
    monotonicity_penalty, entropy, weights_std, pad_to_a_multiple)
from lvsr.extensions import CGStatistics, CodeVersion, AdaptiveClipping
from lvsr.error_rate import wer
from lvsr.preprocessing import log_spectrogram, Normalization
from blocks import serialization

floatX = theano.config.floatX
logger = logging.getLogger(__name__)

def _length(example):
    return len(example[0])


class _LengthFilter(object):

    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, example):
        if self.max_length:
            return len(example[0]) <= self.max_length
        return True


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


class _AddEosLabelEnd(object):

    def __init__(self, eos_label):
        self.eos_label = eos_label

    def __call__(self, example):
        return (example[0], list(example[1]) + [self.eos_label])


class _AddEosLabelBeginEnd(object):

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


class _SilentPadding(object):

    def __init__(self, k_frames):
        self.k_frames = k_frames

    def __call__(self, example):
        features = example[0]
        features = numpy.vstack([features, numpy.zeros_like(features[[0]])])
        return (features, example[1])


class Data(object):

    def __init__(self, dataset, recordings_source, labels_source,
                 batch_size, sort_k_batches,
                 max_length, normalization,
                 merge_k_frames=None,
                 pad_k_frames=None,
                 feature_name='wav', preprocess_features=None,
                 # Need these options to handle old TIMIT models
                 add_eos=True, eos_label=None,
                 prepend_eos=True,
                 # For WSJ
                 preprocess_text=False):
        if not dataset in ('TIMIT', 'WSJ', 'WSJnew'):
            raise ValueError()

        if normalization:
            with open(normalization, "rb") as src:
                normalization = cPickle.load(src)

        self.dataset = dataset
        self.recordings_source = recordings_source
        self.labels_source = labels_source
        self.normalization = normalization
        self.batch_size = batch_size
        self.sort_k_batches = sort_k_batches
        self.merge_k_frames = merge_k_frames
        self.pad_k_frames = pad_k_frames
        self.feature_name = feature_name
        self.max_length = max_length
        self.add_eos = add_eos
        self._eos_label = eos_label
        self.preprocess_text = preprocess_text
        self.preprocess_features = preprocess_features
        self.prepend_eos = prepend_eos
        self.dataset_cache = {}

        self.length_filter =_LengthFilter(self.max_length)

    @property
    def num_labels(self):
        if self.dataset == "TIMIT":
            return self.get_dataset("train").num_phonemes
        return self.get_dataset("train").num_characters

    @property
    def num_features(self):
        merge_multiplier = self.merge_k_frames if self.merge_k_frames else 1
        # For old datasets
        if self.dataset in ['TIMIT', 'WSJ']:
            if self.feature_name == 'wav':
                return 129 * merge_multiplier
            elif self.feature_name == 'fbank_and_delta_delta':
                return 123 * merge_multiplier
        return self.get_dataset("train").num_features * merge_multiplier

    @property
    def eos_label(self):
        if self._eos_label:
            return self._eos_label
        if self.dataset == "TIMIT":
            return TIMIT2.eos_label
        elif self.dataset == "WSJ":
            return 124
        return self.get_dataset("train").eos_label

    def get_dataset(self, part):
        timit_name_mapping = {"train": "train", "valid": "dev", "test": "test"}
        wsj_name_mapping = {"train": "train_si284", "valid": "test_dev93", "test": "test_eval92"}

        if not part in self.dataset_cache:
            if self.dataset == "TIMIT":
                self.dataset_cache[part] = TIMIT2(
                    timit_name_mapping[part], feature_name=self.feature_name)
            elif self.dataset == "WSJ":
                self.dataset_cache[part] = WSJ(
                    wsj_name_mapping[part], feature_name=self.feature_name)
            elif self.dataset == "WSJnew":
                self.dataset_cache[part] = H5PYAudioDataset(
                    os.path.join(fuel.config.data_path, "WSJ/wsj_new.h5"),
                    which_sets=(wsj_name_mapping[part],),
                    sources=(self.recordings_source,
                             self.labels_source))
        return self.dataset_cache[part]

    def get_stream(self, part, batches=True, shuffle=True):
        dataset = self.get_dataset(part)
        stream = (DataStream(dataset,
                             iteration_scheme=ShuffledExampleScheme(dataset.num_examples))
                  if shuffle
                  else dataset.get_example_stream())
        stream = FilterSources(stream, (self.recordings_source,
                                        self.labels_source))
        if self.add_eos:
            if self.prepend_eos:
                stream = Mapping(stream, _AddEosLabelBeginEnd(self.eos_label))
            else:
                stream = Mapping(stream, _AddEosLabelEnd(self.eos_label))
        if self.preprocess_text:
            if not self.dataset == "WSJ":
                raise ValueError("text preprocessing only for WSJ")
            stream = Mapping(stream, lvsr.datasets.wsj.preprocess_text)
        stream = Filter(stream, self.length_filter)
        if self.sort_k_batches and batches:
            stream = Batch(stream,
                        iteration_scheme=ConstantScheme(
                            self.batch_size * self.sort_k_batches))
            stream = Mapping(stream, SortMapping(_length))
            stream = Unpack(stream)

        if self.preprocess_features == 'log_spectrogram':
            stream = Mapping(
                stream, functools.partial(apply_preprocessing,
                                        log_spectrogram))
        if self.pad_k_frames:
            stream = Mapping(
                stream, _SilentPadding(self.pad_k_frames))
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


class InitializableSequence(Sequence, Initializable):
    pass


class Encoder(Initializable):

    def __init__(self, enc_transition, dims, dim_input, subsample, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.subsample = subsample

        for layer_num, (dim_under, dim) in enumerate(
                zip([dim_input] + list(2 * numpy.array(dims)), dims)):
            bidir = Bidirectional(
                RecurrentWithFork(
                    enc_transition(dim=dim, activation=Tanh()).apply,
                    dim_under,
                    name='with_fork'),
                name='bidir{}'.format(layer_num))
            self.children.append(bidir)

    @application(outputs=['encoded', 'encoded_mask'])
    def apply(self, input_, mask=None):
        for bidir, take_each in zip(self.children, self.subsample):
            #No need to pad if all we do is subsample!
            #input_ = pad_to_a_multiple(input_, take_each, 0.)
            #if mask:
            #    mask = pad_to_a_multiple(mask, take_each, 0.)
            input_ = bidir.apply(input_, mask)
            input_ = input_[::take_each]
            if mask:
                mask = mask[::take_each]
        return input_, (mask if mask else tensor.ones_like(input_[:, :, 0]))


def global_push_initialization_config(brick, initialization_config,
                                      filter_type=object):
    #TODO: this needs proper selectors! NOW!
    if not brick.initialization_config_pushed:
        raise Exception("Please push_initializatio_config first to prevent it "
                        "form overriding the changes made by "
                        "global_push_initialization_config")
    if isinstance(brick, filter_type):
        for k,v in initialization_config.items():
            if hasattr(brick, k):
                setattr(brick, k, v)
    for c in brick.children:
        global_push_initialization_config(c, initialization_config, filter_type)


class OneOfNFeedback(AbstractFeedback, Initializable):
    """A feedback brick for the case when readout are integers.

    Stores and retrieves distributed representations of integers.

    """
    def __init__(self, num_outputs=None, feedback_dim=None, **kwargs):
        super(OneOfNFeedback, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.feedback_dim = num_outputs

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0
        eye = tensor.eye(self.num_outputs)
        check_theano_variable(outputs, None, "int")
        output_shape = [outputs.shape[i]
                        for i in range(outputs.ndim)] + [self.feedback_dim]
        return eye[outputs.flatten()].reshape(output_shape)

    def get_dim(self, name):
        if name == 'feedback':
            return self.feedback_dim
        return super(LookupFeedback, self).get_dim(name)


class SpeechModel(Model):
    def set_param_values(self, param_values):
        filtered_param_values = {
            key: value for key, value in param_values.items()
            # Shared variables are now saved separately, thanks to the
            # recent PRs by Dmitry Serdyuk and Bart. Unfortunately,
            # that applies to all shared variables, and not only to the
            # parameters. That's why temporarily we have to filter the
            # unnecessary ones. The filter deliberately does not take into
            # account for a few exotic ones, there will be a warning
            # with the list of the variables that were not matched with
            # model parameters.
            if not ('shared' in key
                    or 'None' in key)}
        super(SpeechModel,self).set_param_values(filtered_param_values)


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
    def __init__(self, recordings_source, labels_source, eos_label,
                 num_features, num_phonemes,
                 dim_dec, dims_bidir, dims_bottom,
                 enc_transition, dec_transition,
                 use_states_for_readout,
                 attention_type,
                 subsample=None,
                 dims_top=None,
                 shift_predictor_dims=None, max_left=None, max_right=None,
                 padding=None, prior=None, conv_n=None,
                 bottom_activation='Tanh()',
                 post_merge_activation='Tanh()',
                 post_merge_dims=None,
                 dim_matcher=None,
                 embed_outputs=True,
                 dec_stack=1,
                 conv_num_filters=1,
                 data_prepend_eos=True,
                 **kwargs):
        super(SpeechRecognizer, self).__init__(**kwargs)
        self.recordings_source = recordings_source
        self.labels_source = labels_source
        self.eos_label = eos_label
        self.data_prepend_eos = data_prepend_eos

        self.rec_weights_init = None
        self.initial_states_init = None

        self.enc_transition = eval(enc_transition)
        self.dec_transition = eval(dec_transition)
        self.dec_stack = dec_stack

        bottom_activation = eval(bottom_activation)
        post_merge_activation = eval(post_merge_activation)

        if dim_matcher is None:
            dim_matcher = dim_dec

        # The bottom part, before BiRNN
        if dims_bottom:
            bottom = MLP([bottom_activation] * len(dims_bottom),
                        [num_features] + dims_bottom,
                        name="bottom")
        else:
            bottom = Identity(name='bottom')

        # BiRNN
        if not subsample:
            subsample = [1] * len(dims_bidir)
        encoder = Encoder(self.enc_transition, dims_bidir,
                          dims_bottom[-1] if len(dims_bottom) else num_features,
                          subsample)

        # The top part, on top of BiRNN but before the attention
        if dims_top:
            top = MLP([Tanh()],
                      [2 * dims_bidir[-1]] + dims_top + [2 * dims_bidir[-1]], name="top")
        else:
            top = Identity(name='top')

        if dec_stack == 1:
            transition = self.dec_transition(
                dim=dim_dec, activation=Tanh(), name="transition")
        else:
            transitions = [self.dec_transition(dim=dim_dec,
                                               activation=Tanh(),
                                               name="transition_{}".format(trans_level))
                           for trans_level in xrange(dec_stack)]
            transition = RecurrentStack(transitions=transitions,
                                        skip_connections=True)
        # Choose attention mechanism according to the configuration
        if attention_type == "content":
            attention = SequenceContentAttention(
                state_names=transition.apply.states,
                attended_dim=2 * dims_bidir[-1], match_dim=dim_matcher,
                name="cont_att")
        elif attention_type == "content_and_conv":
            attention = SequenceContentAndConvAttention(
                state_names=transition.apply.states,
                conv_n=conv_n,
                conv_num_filters=conv_num_filters,
                attended_dim=2 * dims_bidir[-1], match_dim=dim_matcher,
                prior=prior,
                name="conv_att")
        else:
            raise ValueError("Unknown attention type {}"
                             .format(attention_type))
        if embed_outputs:
            feedback = LookupFeedback(num_phonemes + 1, dim_dec)
        else:
            feedback = OneOfNFeedback(num_phonemes + 1)
        readout_config = dict(
            readout_dim=num_phonemes,
            source_names=(transition.apply.states if use_states_for_readout else [])
                + [attention.take_glimpses.outputs[0]],
            emitter=SoftmaxEmitter(initial_output=num_phonemes, name="emitter"),
            feedback_brick=feedback,
            name="readout")
        if post_merge_dims:
            readout_config['merged_dim'] = post_merge_dims[0]
            readout_config['post_merge'] = InitializableSequence([
                    Bias(post_merge_dims[0]).apply,
                    post_merge_activation.apply,
                    MLP([post_merge_activation] * (len(post_merge_dims) - 1) + [Identity()],
                        # MLP was designed to support Maxout is activation
                        # (because Maxout in a way is not one). However
                        # a single layer Maxout network works with the trick below.
                        # For deeper Maxout network one has to use the
                        # Sequence brick.
                        [d//getattr(post_merge_activation, 'num_pieces', 1)
                         for d in post_merge_dims] + [num_phonemes]).apply,
                ],
                name='post_merge')
        readout = Readout(**readout_config)
        generator = SequenceGenerator(
            readout=readout, transition=transition, attention=attention,
            name="generator")

        # Remember child bricks
        self.encoder = encoder
        self.bottom = bottom
        self.top = top
        self.generator = generator
        self.children = [encoder, top, bottom, generator]

        # Create input variables
        self.recordings = tensor.tensor3(self.recordings_source)
        self.recordings_mask = tensor.matrix(self.recordings_source + "_mask")
        self.labels = tensor.lmatrix(self.labels_source)
        self.labels_mask = tensor.matrix(self.labels_source + "_mask")
        self.batch_inputs = [self.recordings, self.recordings_source,
                             self.labels, self.labels_mask]
        self.single_recording = tensor.matrix(self.recordings_source)
        self.single_transcription = tensor.lvector(self.labels_source)

    def push_initialization_config(self):
        super(SpeechRecognizer, self).push_initialization_config()
        if self.rec_weights_init:
            rec_weights_config = {'weights_init': self.rec_weights_init,
                                  'recurrent_weights_init': self.rec_weights_init}
            global_push_initialization_config(self,
                                              rec_weights_config,
                                              BaseRecurrent)
        if self.initial_states_init:
            global_push_initialization_config(self,
                                              {'initial_states_init': self.initial_states_init})

    @application
    def cost(self, recordings, recordings_mask, labels, labels_mask):
        bottom_processed = self.bottom.apply(recordings)
        encoded, encoded_mask = self.encoder.apply(
            input_=bottom_processed,
            mask=recordings_mask)
        encoded = self.top.apply(encoded)
        return self.generator.cost_matrix(
            labels, labels_mask,
            attended=encoded, attended_mask=encoded_mask)

    @application
    def generate(self, recordings):
        encoded, encoded_mask = self.encoder.apply(
            input_=self.bottom.apply(recordings))
        encoded = self.top.apply(encoded)
        return self.generator.generate(
            n_steps=recordings.shape[0], batch_size=recordings.shape[1],
            attended=encoded,
            attended_mask=encoded_mask,
            as_dict=True)

    def load_params(self, path):
        generated = self.get_generate_graph()
        param_values = load_parameter_values(path)
        SpeechModel(generated['outputs']).set_param_values(param_values)

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
            states, = VariableFilter(
                applications=[self.encoder.apply], roles=[OUTPUT],
                name="encoded")(cg)
            ctc_matrix_output = []
            if len(self.generator.readout.source_names) == 1:
                ctc_matrix_output = [
                    self.generator.readout.readout(weighted_averages=states)[:, 0, :]]
            weights, = VariableFilter(
                bricks=[self.generator], name="weights")(cg)
            self._analyze = theano.function(
                [self.single_recording, self.single_transcription],
                [cost[:, 0], weights[:, 0, :]] + energies_output + ctc_matrix_output)
        return self._analyze(recording, transcription)

    def init_beam_search(self, beam_size):
        """Compile beam search and set the beam size.

        See Blocks issue #500.

        """
        self.beam_size = beam_size
        generated = self.get_generate_graph()
        samples, = VariableFilter(
            applications=[self.generator.generate], name="outputs")(
                ComputationGraph(generated['outputs']))
        self._beam_search = BeamSearch(beam_size, samples)
        self._beam_search.compile()

    def beam_search(self, recording):
        if not hasattr(self, '_beam_search'):
            self.init_beam_search(self.beam_size)
        input_ = numpy.tile(recording, (self.beam_size, 1, 1)).transpose(1, 0, 2)
        outputs, search_costs = self._beam_search.search(
            {self.recordings: input_}, self.eos_label, input_.shape[0] / 3,
            ignore_first_eol=self.data_prepend_eos)
        return outputs, search_costs

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ['_analyze', '_beam_search']:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # To use bricks used on a GPU first on a CPU later
        emitter = self.generator.readout.emitter
        if hasattr(emitter, '_theano_rng'):
            del emitter._theano_rng


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
        self.total_errors = 0.
        self.total_length = 0.
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
        self.total_errors += error * len(groundtruth)
        self.total_length += len(groundtruth)
        self.num_examples += 1
        self.mean_error = self.total_errors / self.total_length

    def readout(self):
        return self.mean_error


class SwitchOffLengthFilter(SimpleExtension):

    def __init__(self, length_filter, **kwargs):
        self.length_filter = length_filter
        super(SwitchOffLengthFilter, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        self.length_filter.max_length = None
        self.main_loop.log.current_row['length_filter_switched'] = True

class LoadLog(TrainingExtension):
    """Loads a the log from the checkoint.

    Makes a `LOADED_FROM` record in the log with the dump path.

    Parameters
    ----------
    path : str
        The path to the folder with dump.

    """
    def __init__(self, path, **kwargs):
        super(LoadLog, self).__init__(**kwargs)
        self.path = path[:-4] + '_log.zip'

    def load_to(self, main_loop):

        with open(self.path, "rb") as source:
            loaded_log = pickle.load(source)
            #TODO: remove and fix the printing issue!
            loaded_log.status['resumed_from'] = None
        main_loop.log = loaded_log

    def before_training(self):
        if not os.path.exists(self.path):
            logger.warning("No log dump found")
            return
        logger.info("loading log from {}".format(self.path))
        try:
            self.load_to(self.main_loop)
            #self.main_loop.log.current_row[saveload.LOADED_FROM] = self.path
        except Exception:
            reraise_as("Failed to load the state")



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
        normalization = Normalization(stream, data.recordings_source)
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
            data.recordings_source, data.labels_source,
            data.eos_label,
            data.num_features, data.num_labels,
            name="recognizer",
            data_prepend_eos=data.prepend_eos,
            **config["net"])
        for brick_path, attribute, value in config['initialization']:
            brick, = Selector(recognizer).select(brick_path).bricks
            setattr(brick, attribute, eval(value))
            brick.push_initialization_config()
        recognizer.initialize()

        # Separate attention_params to be handled differently
        # when regularization is applied
        attention = recognizer.generator.transition.attention
        attention_params = Selector(attention).get_params().values()

        logger.info("Initialization schemes for all bricks.\n"
            "Works well only in my branch with __repr__ added to all them,\n"
            "there is an issue #463 in Blocks to do that properly.")
        def show_init_scheme(cur):
            result = dict()
            for attr in dir(cur):
                if attr.endswith('_init'):
                    result[attr] = getattr(cur, attr)
            for child in cur.children:
                result[child.name] = show_init_scheme(child)
            return result
        logger.info(pprint.pformat(show_init_scheme(recognizer)))

        if cmd_args.params:
            logger.info("Load parameters from " + cmd_args.params)
            recognizer.load_params(cmd_args.params)

        if cmd_args.test_tag:
            tensor.TensorVariable.__str__ = tensor.TensorVariable.__repr__
            __stream = data.get_stream("train")
            __data = next(__stream.get_epoch_iterator(as_dict=True))
            recognizer.recordings.tag.test_value = __data[data.recordings_source]
            recognizer.recordings_mask.tag.test_value = __data[data.recordings_source + '_mask']
            recognizer.labels.tag.test_value = __data[data.labels_source]
            recognizer.labels_mask.tag.test_value = __data[data.labels_source + '_mask']
            theano.config.compute_test_value = 'warn'

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
        energies, = VariableFilter(
            applications=[r.generator.readout.readout], name="output_0")(
                    cost_cg)
        bottom_output, = VariableFilter(
            applications=[r.bottom.apply], name="output")(
                    cost_cg)
        attended, = VariableFilter(
            applications=[r.generator.transition.apply], name="attended")(
                    cost_cg)
        attended_mask, = VariableFilter(
            applications=[r.generator.transition.apply], name="attended_mask")(
                    cost_cg)
        weights, = VariableFilter(
            applications=[r.generator.evaluate], name="weights")(
                    cost_cg)
        max_recording_length = named_copy(r.recordings.shape[0],
                                         "max_recording_length")
        # To exclude subsampling related bugs
        max_attended_mask_length = named_copy(attended_mask.shape[0],
                                              "max_attended_mask_length")
        max_attended_length = named_copy(attended.shape[0],
                                         "max_attended_length")
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
            batch_size, max_num_phonemes,
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
            noise_subjects = [p for p in cg.parameters if p not in attention_params]
            regularized_cg = apply_noise(cg, noise_subjects, reg_config['noise'])
        regularized_cost = regularized_cg.outputs[0]
        regularized_weights_penalty = regularized_cg.outputs[1]

        # Model is weird class, we spend lots of time arguing with Bart
        # what it should be. However it can already nice things, e.g.
        # one extract all the parameters from the computation graphs
        # and give them hierahical names. This help to notice when a
        # because of some bug a parameter is not in the computation
        # graph.
        model = SpeechModel(regularized_cost)
        params = model.get_params()
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, params[key].get_value().shape) for key
                         in sorted(params.keys())],
                        width=120))

        # Define the training algorithm.
        train_conf = config['training']
        clipping = StepClipping(train_conf['gradient_threshold'])
        clipping.threshold.name = "gradient_norm_threshold"
        rule_names = train_conf.get('rules', ['momentum'])
        core_rules = []
        if 'momentum' in rule_names:
            logger.info("Using scaling and momentum for training")
            core_rules.append(Momentum(train_conf['scale'], train_conf['momentum']))
        if 'adadelta' in rule_names:
            logger.info("Using AdaDelta for training")
            core_rules.append(AdaDelta(train_conf['decay_rate'], train_conf['epsilon']))
        max_norm_rules = []
        if reg_config.get('max_norm', False):
            logger.info("Apply MaxNorm")
            maxnorm_subjects = VariableFilter(roles=[WEIGHT])(cg.parameters)
            logger.info("Parameters covered by MaxNorm:\n"
                        + pprint.pformat([name for name, p in params.items()
                                          if p in maxnorm_subjects]))
            logger.info("Parameters NOT covered by MaxNorm:\n"
                        + pprint.pformat([name for name, p in params.items()
                                          if not p in maxnorm_subjects]))
            max_norm_rules = [
                Restrict(VariableClipping(reg_config['max_norm'], axis=0),
                         maxnorm_subjects)]
        algorithm = GradientDescent(
            cost=regularized_cost +
                reg_config.get("penalty_coof", .0) * regularized_weights_penalty / batch_size +
                reg_config.get("decay", .0) *
                l2_norm(VariableFilter(roles=[WEIGHT])(cg.parameters)) ** 2,
            params=params.values(),
            step_rule=CompositeRule(
                [clipping] + core_rules + max_norm_rules +
                # Parameters are not changed at all
                # when nans are encountered.
                [RemoveNotFinite(0.0)]))

        # More variables for debugging: some of them can be added only
        # after the `algorithm` object is created.
        observables = regularized_cg.outputs
        observables += [
            algorithm.total_step_norm, algorithm.total_gradient_norm,
            clipping.threshold]
        for name, param in params.items():
            num_elements = numpy.product(param.get_value().shape)
            norm = param.norm(2) / num_elements ** 0.5
            grad_norm = algorithm.gradients[param].norm(2) / num_elements ** 0.5
            step_norm = algorithm.steps[param].norm(2) / num_elements ** 0.5
            stats = tensor.stack(norm, grad_norm, step_norm, step_norm / grad_norm)
            stats.name = name + '_stats'
            observables.append(stats)

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
        logger.info("Initialize extensions")
        extensions = []
        if cmd_args.use_load_ext and cmd_args.params:
            extensions.append(Load(cmd_args.params, load_iteration_state=True, load_log=True))
        if cmd_args.load_log and cmd_args.params:
            extensions.append(LoadLog(cmd_args.params))
        extensions += [
            Timing(after_batch=True),
            CGStatistics(),
            #CodeVersion(['lvsr']),
            ]
        extensions.append(TrainingDataMonitoring(
            [observables[0], algorithm.total_gradient_norm,
             algorithm.total_step_norm, clipping.threshold,
             max_recording_length,
             max_attended_length, max_attended_mask_length], after_batch=True))
        average_monitoring = TrainingDataMonitoring(
            attach_aggregation_schemes(observables),
            prefix="average", every_n_batches=10)
        extensions.append(average_monitoring)
        validation = DataStreamMonitoring(
            attach_aggregation_schemes([cost, weights_entropy, weights_penalty]),
            data.get_stream("valid"), prefix="valid",
            before_first_epoch=not cmd_args.fast_start,
            after_epoch=True, after_training=False)
        extensions.append(validation)
        recognizer.init_beam_search(10)
        per = PhonemeErrorRate(recognizer, data.get_dataset("valid"))
        per_monitoring = DataStreamMonitoring(
            [per], data.get_stream("valid", batches=False, shuffle=False),
            prefix="valid").set_conditions(
                before_first_epoch=not cmd_args.fast_start, every_n_epochs=2,
                after_training=False)
        extensions.append(per_monitoring)
        track_the_best_per = TrackTheBest(
            per_monitoring.record_name(per)).set_conditions(
                before_first_epoch=True, after_epoch=True)
        track_the_best_likelihood = TrackTheBest(
            validation.record_name(cost)).set_conditions(
                before_first_epoch=True, after_epoch=True)
        extensions += [track_the_best_likelihood, track_the_best_per]
        extensions.append(AdaptiveClipping(
            algorithm.total_gradient_norm.name,
            clipping, train_conf['gradient_threshold'],
            decay_rate=0.998, burnin_period=500))
        extensions += [
            SwitchOffLengthFilter(data.length_filter,
                after_n_batches=train_conf.get('stop_filtering', 1)),
            FinishAfter(after_n_batches=cmd_args.num_batches,
                        after_n_epochs=cmd_args.num_epochs)
            .add_condition(["after_batch"], _gradient_norm_is_none),
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
                       save_separately=["model", "log"],
                       use_cpickle=True)
            .add_condition(
                ['after_epoch'],
                OnLogRecord(track_the_best_per.notification_name),
                (root_path + "_best" + extension,))
            .add_condition(
                ['after_epoch'],
                OnLogRecord(track_the_best_likelihood.notification_name),
                (root_path + "_best_ll" + extension,)),
            ProgressBar(),
            Printing(every_n_batches=1,
                     attribute_filter=PrintingFilterList()
                     )]

        # Save the config into the status
        log = TrainingLog()
        log.status['_config'] = config
        main_loop = MainLoop(
            model=model, log=log, algorithm=algorithm,
            data_stream=data.get_stream("train"),
            extensions=extensions)
        main_loop.run()


    elif cmd_args.mode == "search":
        from matplotlib import pyplot
        from lvsr.notebook import show_alignment

        # Try to guess if just parameters or the whole model was given.
        if cmd_args.params is not None:
            recognizer = SpeechRecognizer(
                data.recordings_source, data.labels_source,
                data.eos_label, data.num_features, data.num_labels,
                name='recognizer', **config["net"])
            recognizer.load_params(cmd_args.save_path)
        else:
            recognizer, = cPickle.load(
                open(cmd_args.save_path)).get_top_bricks()
        recognizer.init_beam_search(cmd_args.beam_size)

        dataset = data.get_dataset(cmd_args.part)
        stream = data.get_stream(cmd_args.part, batches=False, shuffle=False)
        it = stream.get_epoch_iterator()

        weights = tensor.matrix('weights')
        weight_statistics = theano.function(
            [weights],
            [weights_std(weights.dimshuffle(0, 'x', 1)),
             monotonicity_penalty(weights.dimshuffle(0, 'x', 1))])

        print_to = sys.stdout
        if cmd_args.report:
            alignments_path = os.path.join(cmd_args.report, "alignments")
            if not os.path.exists(cmd_args.report):
                os.mkdir(cmd_args.report)
                os.mkdir(alignments_path)
            print_to = open(os.path.join(cmd_args.report, "report.txt"), 'w')

        total_errors = .0
        total_length = .0
        for number, data in enumerate(it):
            print("Utterance", number, file=print_to)

            outputs, search_costs = recognizer.beam_search(data[0])
            recognized = dataset.decode(
                outputs[0], **({'old_labels': True} if cmd_args.old_labels else {}))
            groundtruth = dataset.decode(data[1])
            costs_recognized, weights_recognized = (
                recognizer.analyze(data[0], outputs[0])[:2])
            costs_groundtruth, weights_groundtruth = (
                recognizer.analyze(data[0], data[1])[:2])
            weight_std_recognized, mono_penalty_recognized = weight_statistics(
                weights_recognized)
            weight_std_groundtruth, mono_penalty_groundtruth = weight_statistics(
                weights_groundtruth)
            error = min(1, wer(groundtruth, recognized))
            total_errors += len(groundtruth) * error
            total_length += len(groundtruth)

            if cmd_args.report:
                show_alignment(weights_groundtruth, groundtruth, bos_symbol=True)
                pyplot.savefig(os.path.join(
                    alignments_path, "{}.groundtruth.png".format(number)))
                show_alignment(weights_recognized, recognized, bos_symbol=True)
                pyplot.savefig(os.path.join(
                    alignments_path, "{}.recognized.png".format(number)))

            print("Beam search cost:", search_costs[0], file=print_to)
            print("Recognizer:", recognized, file=print_to)
            print("Recognized cost:", costs_recognized.sum(), file=print_to)
            print("Recognized weight std:", weight_std_recognized, file=print_to)
            print("Recognized monotonicity penalty:", mono_penalty_recognized, file=print_to)
            print("Groundtruth:", groundtruth, file=print_to)
            print("Groundtruth cost:", costs_groundtruth.sum(), file=print_to)
            print("Groundtruth weight std:", weight_std_groundtruth, file=print_to)
            print("Groundtruth monotonicity penalty:", mono_penalty_groundtruth, file=print_to)
            print("PER:", error, file=print_to)
            print("Average PER:", total_errors / total_length, file=print_to)

            # assert_allclose(search_costs[0], costs_recognized.sum(), rtol=1e-5)
