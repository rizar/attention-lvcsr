from __future__ import print_function
import time
import logging
import pprint
import math
import os
import cPickle
import cPickle as pickle
import sys

import numpy
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from blocks.bricks.lookup import LookupTable
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from blocks.algorithms import (GradientDescent,
                               StepClipping, CompositeRule,
                               Momentum, RemoveNotFinite, AdaDelta,
                               Restrict, VariableClipping)
from blocks.monitoring import aggregation
from blocks.monitoring.aggregation import MonitoredQuantity
from blocks.theano_expressions import l2_norm
from blocks.extensions import (
    FinishAfter, Printing, Timing, ProgressBar, SimpleExtension,
    TrainingExtension, PrintingFilterList)
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.monitoring import (
    TrainingDataMonitoring, DataStreamMonitoring)
from blocks.extras.extensions.plot import Plot
from blocks.extensions.training import TrackTheBest
from blocks.extensions.predicates import OnLogRecord
from blocks.log import TrainingLog
from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter, get_brick
from blocks.roles import WEIGHT
from blocks.utils import named_copy, reraise_as
from blocks.search import CandidateNotFoundError
from blocks.select import Selector

from lvsr.bricks import RewardRegressionEmitter
from lvsr.bricks.recognizer import SpeechRecognizer
from lvsr.datasets import Data
from lvsr.expressions import (
    monotonicity_penalty, entropy, weights_std)
from lvsr.extensions import (
    CGStatistics, AdaptiveClipping, LogInputsGains)
from lvsr.error_rate import wer
from lvsr.preprocessing import Normalization
from lvsr.utils import SpeechModel

floatX = theano.config.floatX
logger = logging.getLogger(__name__)


def _gradient_norm_is_none(log):
    return math.isnan(log.current_row.get('total_gradient_norm', 0))


class PhonemeErrorRate(MonitoredQuantity):

    def __init__(self, recognizer, data, **kwargs):
        self.recognizer = recognizer
        # Will only be used to decode generated outputs,
        # which is necessary for correct scoring.
        self.data = data
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
        groundtruth = self.data.decode(transcription)
        try:
            outputs, search_costs = self.recognizer.beam_search(
                recording, char_discount=0.1)
            recognized = self.data.decode(outputs[0])
            error = min(1, wer(groundtruth, recognized))
        except CandidateNotFoundError:
            error = 1.0
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


def train(config, save_path, bokeh_name,
          params, bokeh_server, test_tag, use_load_ext,
          load_log, fast_start, validation_epochs, validation_batches,
          per_epochs, per_batches):
    root_path, extension = os.path.splitext(save_path)

    train_conf = config['training']

    data = Data(**config['data'])

    # Build the main brick and initialize all parameters.
    recognizer = SpeechRecognizer(
        data.recordings_source, data.labels_source,
        data.eos_label,
        data.num_features, data.num_labels,
        name="recognizer",
        data_prepend_eos=data.prepend_eos,
        character_map=data.character_map,
        **config["net"])
    for brick_path, attribute_dict in sorted(
            config['initialization'].items(),
            key=lambda (k, v): k.count('/')):
        for attribute, value in attribute_dict.items():
            brick, = Selector(recognizer).select(brick_path).bricks
            setattr(brick, attribute, value)
            brick.push_initialization_config()
    recognizer.initialize()

    if test_tag:
        tensor.TensorVariable.__str__ = tensor.TensorVariable.__repr__
        __stream = data.get_stream("train")
        __data = next(__stream.get_epoch_iterator(as_dict=True))
        recognizer.recordings.tag.test_value = __data[data.recordings_source]
        recognizer.recordings_mask.tag.test_value = __data[data.recordings_source + '_mask']
        recognizer.labels.tag.test_value = __data[data.labels_source]
        recognizer.labels_mask.tag.test_value = __data[data.labels_source + '_mask']
        theano.config.compute_test_value = 'warn'

    # Separate attention_params to be handled differently
    # when regularization is applied
    attention = recognizer.generator.transition.attention
    attention_params = Selector(attention).get_parameters().values()

    logger.info(
        "Initialization schemes for all bricks.\n"
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

    prediction = None
    prediction_mask = None
    explore_conf = train_conf.get('exploration')
    if explore_conf in ['greedy', 'mixed']:
        length_expand = 10
        prediction = recognizer.get_generate_graph(
            n_steps=recognizer.labels.shape[0] + length_expand)['outputs']
        prediction_mask = tensor.lt(
            tensor.cumsum(tensor.eq(prediction, data.eos_label), axis=0),
            1).astype(floatX)
        prediction_mask = tensor.roll(prediction_mask, 1, 0)
        prediction_mask = tensor.set_subtensor(
            prediction_mask[0, :], tensor.ones_like(prediction_mask[0, :]))

        if explore_conf == 'mixed':
            batch_size = recognizer.labels.shape[1]
            targets = tensor.concatenate([
                recognizer.labels,
                tensor.zeros((length_expand, batch_size), dtype='int64')])

            targets_mask = tensor.concatenate([
                recognizer.labels_mask,
                tensor.zeros((length_expand, batch_size), dtype=floatX)])
            rng = MRG_RandomStreams()
            generate = rng.binomial((batch_size,), p=0.5, dtype='int64')
            prediction = (generate[None, :] * prediction +
                          (1 - generate[None, :]) * targets)
            prediction_mask = (tensor.cast(generate[None, :] *
                                           prediction_mask, floatX) +
                               tensor.cast((1 - generate[None, :]) *
                                           targets_mask, floatX))

        prediction_mask = theano.gradient.disconnected_grad(prediction_mask)
    elif explore_conf != 'imitative':
        raise ValueError

    cg = recognizer.get_cost_graph(
        batch=True, prediction=prediction, prediction_mask=prediction_mask)
    labels, = VariableFilter(
        applications=[recognizer.cost], name='labels')(cg)
    labels_mask, = VariableFilter(
        applications=[recognizer.cost], name='labels_mask')(cg)
    criterion_related_observables = []
    gain_matrix = VariableFilter(
        theano_name=RewardRegressionEmitter.GAIN_MATRIX)(cg)
    if len(gain_matrix):
        gain_matrix, = gain_matrix
        criterion_related_observables.append(
            named_copy(gain_matrix.min(), 'min_gain'))
        criterion_related_observables.append(
            named_copy(gain_matrix.max(), 'max_gain'))

    batch_cost = cg.outputs[0].sum().sum()
    batch_size = named_copy(recognizer.recordings.shape[1], "batch_size")
    # Assumes constant batch size. `aggregation.mean` is not used because
    # of Blocks #514.
    cost = batch_cost / batch_size
    cost.name = "sequence_total_cost"
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
    bottom_output = VariableFilter(
        applications=[r.bottom.apply], name="output")(
                cost_cg)[-1]
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
    max_num_phonemes = named_copy(labels.shape[0],
                                  "max_num_phonemes")
    min_energy = named_copy(energies.min(), "min_energy")
    max_energy = named_copy(energies.max(), "max_energy")
    mean_attended = named_copy(abs(attended).mean(),
                               "mean_attended")
    mean_bottom_output = named_copy(abs(bottom_output).mean(),
                                    "mean_bottom_output")
    weights_penalty = named_copy(monotonicity_penalty(weights, labels_mask),
                                 "weights_penalty")
    weights_entropy = named_copy(entropy(weights, labels_mask),
                                 "weights_entropy")
    mask_density = named_copy(labels_mask.mean(),
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
    if reg_config.get('dropout'):
        logger.info('apply dropout')
        regularized_cg = apply_dropout(cg, [bottom_output], 0.5)
    if reg_config.get('noise'):
        logger.info('apply noise')
        noise_subjects = [p for p in cg.parameters if p not in attention_params]
        regularized_cg = apply_noise(cg, noise_subjects, reg_config['noise'])
    regularized_cost = regularized_cg.outputs[0]

    # Model is weird class, we spend lots of time arguing with Bart
    # what it should be. However it can already nice things, e.g.
    # one extract all the parameters from the computation graphs
    # and give them hierahical names. This help to notice when a
    # because of some bug a parameter is not in the computation
    # graph.
    model = SpeechModel(regularized_cost)
    parameters = model.get_parameter_dict()
    logger.info("Parameters:\n" +
                pprint.pformat(
                    [(key, parameters[key].get_value().shape) for key
                        in sorted(parameters.keys())],
                    width=120))

    # Define the training algorithm.
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
    if reg_config.get('max_norm', False) > 0:
        logger.info("Apply MaxNorm")
        maxnorm_subjects = VariableFilter(roles=[WEIGHT])(cg.parameters)
        if reg_config.get('max_norm_exclude_lookup', False):
            maxnorm_subjects = [v for v in maxnorm_subjects
                                if not isinstance(get_brick(v), LookupTable)]
        logger.info("Parameters covered by MaxNorm:\n"
                    + pprint.pformat([name for name, p in parameters.items()
                                        if p in maxnorm_subjects]))
        logger.info("Parameters NOT covered by MaxNorm:\n"
                    + pprint.pformat([name for name, p in parameters.items()
                                        if not p in maxnorm_subjects]))
        max_norm_rules = [
            Restrict(VariableClipping(reg_config['max_norm'], axis=0),
                        maxnorm_subjects)]
    algorithm = GradientDescent(
        cost=regularized_cost +
            reg_config.get("decay", .0) *
            l2_norm(VariableFilter(roles=[WEIGHT])(cg.parameters)) ** 2,
        parameters=parameters.values(),
        step_rule=CompositeRule(
            [clipping] + core_rules + max_norm_rules +
            # Parameters are not changed at all
            # when nans are encountered.
            [RemoveNotFinite(0.0)]),
        on_unused_sources='warn')

    logger.debug("Scan Ops in the gradients")
    gradient_cg = ComputationGraph(algorithm.gradients.values())
    for op in ComputationGraph(gradient_cg).scans:
        logger.debug(op)

    if params:
        logger.info("Load parameters from " + params)
        recognizer.load_params(params)

    # Sometimes there are a few competing losses
    # other_losses = VariableFilter(roles=[OTHER_LOSS])(cg)
    other_losses = []

    # More variables for debugging: some of them can be added only
    # after the `algorithm` object is created.
    observables = regularized_cg.outputs
    observables += [
        algorithm.total_step_norm, algorithm.total_gradient_norm,
        clipping.threshold]
    for name, param in parameters.items():
        num_elements = numpy.product(param.get_value().shape)
        norm = param.norm(2) / num_elements ** 0.5
        grad_norm = algorithm.gradients[param].norm(2) / num_elements ** 0.5
        step_norm = algorithm.steps[param].norm(2) / num_elements ** 0.5
        stats = tensor.stack(norm, grad_norm, step_norm, step_norm / grad_norm)
        stats.name = name + '_stats'
        observables.append(stats)
    observables.extend(other_losses)
    primary_observables = [
        regularized_cost, algorithm.total_gradient_norm,
        algorithm.total_step_norm, clipping.threshold,
        max_recording_length,
        max_attended_length, max_attended_mask_length]
    primary_observables += criterion_related_observables

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
                    var, labels_mask.sum()), 'weights_entropy_per_label'))
            else:
                result.append(var)
        return result

    # Build main loop.
    logger.info("Initialize extensions")
    extensions = []
    if use_load_ext and params:
        extensions.append(Load(params, load_iteration_state=True, load_log=True))
    if load_log and params:
        extensions.append(LoadLog(params))
    extensions += [
        Timing(after_batch=True),
        CGStatistics(),
        #CodeVersion(['lvsr']),
        ]
    extensions.append(TrainingDataMonitoring(
        primary_observables, after_batch=True))
    average_monitoring = TrainingDataMonitoring(
        attach_aggregation_schemes(observables),
        prefix="average", every_n_batches=10)
    extensions.append(average_monitoring)
    validation = DataStreamMonitoring(
        attach_aggregation_schemes([cost, weights_entropy, weights_penalty]),
        data.get_stream("valid"), prefix="valid").set_conditions(
            before_first_epoch=not fast_start,
            every_n_epochs=validation_epochs,
            every_n_batches=validation_batches,
            after_training=False)
    extensions.append(validation)
    recognizer.init_beam_search(10)
    per = PhonemeErrorRate(recognizer, data)
    per_monitoring = DataStreamMonitoring(
        [per], data.get_stream("valid", batches=False, shuffle=False),
        prefix="valid").set_conditions(
            before_first_epoch=not fast_start,
            every_n_epochs=per_epochs,
            every_n_batches=per_batches,
            after_training=False)
    extensions.append(per_monitoring)
    track_the_best_per = TrackTheBest(
        per_monitoring.record_name(per)).set_conditions(
            before_first_epoch=True, after_epoch=True)
    track_the_best_cost = TrackTheBest(
        validation.record_name(cost)).set_conditions(
            before_first_epoch=True, after_epoch=True)
    extensions += [track_the_best_cost, track_the_best_per]
    extensions.append(AdaptiveClipping(
        algorithm.total_gradient_norm.name,
        clipping, train_conf['gradient_threshold'],
        decay_rate=0.998, burnin_period=500))
    extensions += [
        SwitchOffLengthFilter(
            data.length_filter,
            after_n_batches=train_conf.get('stop_filtering')),
        FinishAfter(after_n_batches=train_conf['num_batches'],
                    after_n_epochs=train_conf['num_epochs'])
        .add_condition(["after_batch"], _gradient_norm_is_none),
        ]
    channels = [
        # Plot 1: training and validation costs
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
        validation._record_name('weights_penalty_per_recording')]]
    for loss in other_losses:
        channels[0].append(average_monitoring.record_name(loss))
    extensions += [
        Plot(bokeh_name if bokeh_name
             else os.path.basename(save_path),
             channels,
             every_n_batches=10,
             server_url=bokeh_server),
        Checkpoint(save_path,
                   before_first_epoch=not fast_start, after_epoch=True,
                   every_n_batches=train_conf.get('save_every_n_batches'),
                   save_separately=["model", "log"],
                   use_cpickle=True)
        .add_condition(
            ['after_epoch'],
            OnLogRecord(track_the_best_per.notification_name),
            (root_path + "_best" + extension,))
        .add_condition(
            ['after_epoch'],
            OnLogRecord(track_the_best_cost.notification_name),
            (root_path + "_best_ll" + extension,)),
        ProgressBar()]
    if config['net']['criterion']['name'].startswith('mse'):
        extensions.append(
            LogInputsGains(
                labels, cg, recognizer.generator.readout.emitter, data))

    extensions.append(Printing(every_n_batches=1,
                               attribute_filter=PrintingFilterList()))

    # Save the config into the status
    log = TrainingLog()
    log.status['_config'] = repr(config)
    main_loop = MainLoop(
        model=model, log=log, algorithm=algorithm,
        data_stream=data.get_stream("train"),
        extensions=extensions)
    main_loop.run()


def search(config, params, load_path, beam_size, part, decode_only, report,
           decoded_save, nll_only, char_discount):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot
    from lvsr.notebook import show_alignment

    data = Data(**config['data'])

    logger.info("Recognizer initialization started")
    recognizer = SpeechRecognizer(
        data.recordings_source, data.labels_source,
        data.eos_label, data.num_features, data.num_labels,
        character_map=data.character_map,
        name='recognizer', **config["net"])
    recognizer.load_params(load_path)
    recognizer.init_beam_search(beam_size)
    logger.info("Recognizer is initialized")

    stream = data.get_stream(part, batches=False, shuffle=False,
                                add_sources=(data.uttid_source,))
    it = stream.get_epoch_iterator()
    if decode_only is not None:
        decode_only = eval(decode_only)

    weights = tensor.matrix('weights')
    weight_statistics = theano.function(
        [weights],
        [weights_std(weights.dimshuffle(0, 'x', 1)),
            monotonicity_penalty(weights.dimshuffle(0, 'x', 1))])

    print_to = sys.stdout
    if report:
        alignments_path = os.path.join(report, "alignments")
        if not os.path.exists(report):
            os.mkdir(report)
            os.mkdir(alignments_path)
        print_to = open(os.path.join(report, "report.txt"), 'w')

    decoded_file = None
    if decoded_save:
        decoded_file = open(decoded_save, 'w')

    num_examples = .0
    total_nll = .0
    total_errors = .0
    total_length = .0
    total_wer_errors = .0
    total_word_length = 0.
    with open(os.path.expandvars(config['vocabulary'])) as f:
        vocabulary = dict(line.split() for line in f.readlines())

    def to_words(chars):
        words = chars.split()
        words = [vocabulary[word] if word in vocabulary
                    else vocabulary['<UNK>'] for word in words]
        return words

    for number, example in enumerate(it):
        if decode_only and number not in decode_only:
            continue
        print("Utterance {} ({})".format(number, example[2]), file=print_to)
        groundtruth = data.decode(example[1])
        groundtruth_text = data.pretty_print(example[1])
        costs_groundtruth, weights_groundtruth = (
            recognizer.analyze(example[0], example[1], example[1])[:2])
        weight_std_groundtruth, mono_penalty_groundtruth = weight_statistics(
            weights_groundtruth)
        total_nll += costs_groundtruth.sum()
        num_examples += 1
        print("Groundtruth:", groundtruth_text, file=print_to)
        print("Groundtruth cost:", costs_groundtruth.sum(), file=print_to)
        print("Groundtruth weight std:", weight_std_groundtruth, file=print_to)
        print("Groundtruth monotonicity penalty:", mono_penalty_groundtruth, file=print_to)
        print("Average groundtruth cost: {}".format(total_nll / num_examples),
                file=print_to)
        if nll_only:
            continue

        before = time.time()
        stop_on = ('patience'
                   if config['net']['criterion']['name'].startswith('mse')
                   else 'nll')
        outputs, search_costs = recognizer.beam_search(
            example[0], char_discount=char_discount, round_to_inf=4.5,
            stop_on=stop_on)
        took = time.time() - before
        recognized = data.decode(outputs[0])
        recognized_text = data.pretty_print(outputs[0])
        costs_recognized, weights_recognized = (
            recognizer.analyze(example[0], example[1], outputs[0])[:2])
        weight_std_recognized, mono_penalty_recognized = weight_statistics(
            weights_recognized)
        error = min(1, wer(groundtruth, recognized))
        total_errors += len(groundtruth) * error
        total_length += len(groundtruth)

        wer_error = min(1, wer(to_words(groundtruth_text),
                                to_words(recognized_text)))
        total_wer_errors += len(groundtruth) * wer_error
        total_word_length += len(groundtruth)

        if report and recognized:
            show_alignment(weights_groundtruth, groundtruth, bos_symbol=True)
            pyplot.savefig(os.path.join(
                alignments_path, "{}.groundtruth.png".format(number)))
            show_alignment(weights_recognized, recognized, bos_symbol=True)
            pyplot.savefig(os.path.join(
                alignments_path, "{}.recognized.png".format(number)))

        if decoded_file is not None:
            print("{} {}".format(example[2], ' '.join(recognized)), file=decoded_file)

        print("Decoding took:", took, file=print_to)
        print("Beam search cost:", search_costs[0], file=print_to)
        print("Recognized:", recognized_text, file=print_to)
        print("Recognized cost:", costs_recognized.sum(), file=print_to)
        print("Recognized weight std:", weight_std_recognized, file=print_to)
        print("Recognized monotonicity penalty:", mono_penalty_recognized, file=print_to)
        print("CER:", error, file=print_to)
        print("Average CER:", total_errors / total_length, file=print_to)
        print("WER:", wer_error, file=print_to)
        print("Average WER:", total_wer_errors / total_word_length, file=print_to)

        #assert_allclose(search_costs[0], costs_recognized.sum(), rtol=1e-5)

def sample(config, params, load_path, part):
    data = Data(**config['data'])

    recognizer = SpeechRecognizer(
        data.recordings_source, data.labels_source,
        data.eos_label, data.num_features, data.num_labels,
        character_map=data.character_map,
        name='recognizer', **config["net"])
    recognizer.load_params(load_path)

    dataset = data.get_dataset(part, add_sources=(data.uttid_source,))
    stream = data.get_stream(part, batches=False, shuffle=False,
                                add_sources=(data.uttid_source,))
    it = stream.get_epoch_iterator()

    print_to = sys.stdout
    for number, data in enumerate(it):
        print("Utterance {} ({})".format(number, data[2]), file=print_to)
        groundtruth_text = dataset.pretty_print(data[1])
        print("Groundtruth:", groundtruth_text, file=print_to)
        sample = recognizer.sample(data[0])['outputs'][:, 0]
        recognized_text = dataset.pretty_print(sample)
        print("Recognized:", recognized_text, file=print_to)


def init_norm(config, save_path):
    config['data']['normalization'] = None
    data = Data(**config['data'])
    stream = data.get_stream("train", batches=False, shuffle=False)
    normalization = Normalization(stream, data.recordings_source)
    with open(save_path, "wb") as dst:
        cPickle.dump(normalization, dst)


def show_data(config):
    data = Data(**config['data'])
    stream = data.get_stream("train")
    batch = next(stream.get_epoch_iterator(as_dict=True))
    import IPython; IPython.embed()


def train_multistage(config, save_path, bokeh_name, params, start_stage, **kwargs):
    """Run multiple stages of the training procedure."""
    if config.multi_stage:
        if not start_stage:
            os.mkdir(save_path)
        start_stage = (list(config.ordered_stages).index(start_stage)
                       if start_stage else 0)
        stages = list(config.ordered_stages.items())
        for number in range(start_stage, len(stages)):
            stage_name, stage_config = stages[number]
            logging.info("Stage \"{}\" config:\n".format(stage_name)
                         + pprint.pformat(stage_config, width=120))
            stage_save_path = '{}/{}.zip'.format(save_path, stage_name)
            stage_bokeh_name = '{}_{}'.format(save_path, stage_name)
            if number and not params:
                stage_params = '{}/{}{}.zip'.format(
                    save_path, stages[number - 1][0],
                    stage_config['training'].get('restart_from', ''))
            else:
                stage_params = params
                # Avoid loading the params twice
                params = None

            train(stage_config, stage_save_path, stage_bokeh_name,
                  stage_params, **kwargs)
    else:
        train(config, save_path, bokeh_name, params, **kwargs)


def test(config, **kwargs):
    raise NotImplementedError()
