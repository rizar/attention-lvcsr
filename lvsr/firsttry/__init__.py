from __future__ import print_function
import logging
import pprint
import math
import os
import functools
import dill

import numpy
import theano
from theano import tensor
from blocks.bricks import Tanh, application, MLP
from blocks.bricks.recurrent import (
    BaseRecurrent, SimpleRecurrent, GatedRecurrent, Bidirectional)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, LinearReadout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.datasets.streams import (
    DataStream, DataStreamMapping, PaddingDataStream,
    ForceFloatX)
from blocks.datasets.schemes import SequentialScheme
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
from blocks.select import Selector
from blocks.filter import VariableFilter
from blocks.utils import named_copy, unpack, dict_union

from lvsr.datasets import TIMIT
from lvsr.preprocessing import log_spectrogram
from lvsr.expressions import monotonicity_penalty, entropy

floatX = theano.config.floatX
logger = logging.getLogger(__name__)


class Transition(SimpleRecurrent):
    def __init__(self, attended_dim, **kwargs):
        super(Transition, self).__init__(**kwargs)
        self.attended_dim = attended_dim

    @application(contexts=['attended', 'attended_mask'])
    def apply(self, *args, **kwargs):
        for context in Transition.apply.contexts:
            kwargs.pop(context)
        return super(Transition, self).apply(*args, **kwargs)

    @apply.delegate
    def apply_delegate(self):
        return super(Transition, self).apply

    def get_dim(self, name):
        if name == 'attended':
            return self.attended_dim
        if name == 'attended_mask':
            return 0
        return super(Transition, self).get_dim(name)


def apply_preprocessing(preprocessing, batch):
    recordings, labels = batch
    recordings = [preprocessing(r) for r in recordings]
    return (numpy.asarray(recordings), labels)


def switch_first_two_axes(batch):
    result = []
    for array in batch:
        if array.ndim == 2:
            result.append(array.transpose(1, 0))
        else:
            result.append(array.transpose(1, 0, 2))
    return tuple(result)


def build_stream(dataset, batch_size):
    data_stream = DataStream(
        dataset,
        iteration_scheme = SequentialScheme(dataset.num_examples, 10))
    data_stream = DataStreamMapping(
        data_stream, functools.partial(apply_preprocessing,
                                       log_spectrogram))
    data_stream = PaddingDataStream(data_stream)
    data_stream = DataStreamMapping(
        data_stream, switch_first_two_axes)
    data_stream = ForceFloatX(data_stream)
    return data_stream


def main(mode, save_path, num_batches, use_old, from_dump, config_path):
    with open(config_path, 'rt') as config_file:
        config = eval(config_file.read())

    if mode == "train":
        # Experiment configuration
        dimension = config['dim']
        dim_bidir = config['dim_bidir']
        dims_bottom = config['dims_bottom']

        timit = TIMIT()
        timit_valid = TIMIT("valid")
        root_path, extension = os.path.splitext(save_path)
        model_path = root_path + "_model" + extension

        # Build the bricks
        if not use_old:
            encoder = Bidirectional(
                SimpleRecurrent(dim=dim_bidir, activation=Tanh()),
                weights_init=Orthogonal())
            encoder.initialize()
            fork = Fork([name for name in encoder.prototype.apply.sequences
                        if name != 'mask'],
                        weights_init=IsotropicGaussian(0.1),
                        biases_init=Constant(0))
            fork.input_dim = dims_bottom[-1]
            fork.output_dims = {name: dim_bidir for name in fork.output_names}
            mlp = MLP([Tanh()] * len(config['dims_bottom']),
                    [129] + config['dims_bottom'],
                    name="bottom",
                    weights_init=IsotropicGaussian(0.1),
                    biases_init=Constant(0))
            transition = Transition(
                activation=Tanh(),
                dim=dimension, attended_dim=2 * dim_bidir, name="transition")
            attention = SequenceContentAttention(
                state_names=transition.apply.states,
                match_dim=dimension, name="attention")
            readout = LinearReadout(
                readout_dim=timit.num_phonemes,
                source_names=transition.apply.states,
                emitter=SoftmaxEmitter(name="emitter"),
                feedbacker=LookupFeedback(timit.num_phonemes, dimension),
                name="readout")
            generator = SequenceGenerator(
                readout=readout, transition=transition, attention=attention,
                weights_init=IsotropicGaussian(0.1), biases_init=Constant(0),
                name="generator")
            generator.push_initialization_config()
            transition.weights_init = Orthogonal()
            bricks = [encoder, fork, mlp, generator]
            for brick in bricks:
                brick.initialize()
        else:
            with open(model_path, "rb") as source:
                bricks = dill.load(source)
            encoder, fork, mlp, generator = bricks
            readout = generator.readout
            logging.info("Loaded an old model")

        # Give an idea of what's going on
        params = Selector(bricks).get_params()
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in params.items()],
                        width=120))

        # Build the cost computation graph
        recordings = tensor.tensor3("recordings")
        recordings_mask = tensor.matrix("recordings_mask")
        labels = tensor.lmatrix("labels")
        labels_mask = tensor.matrix("labels_mask")
        batch_cost = generator.cost(
            labels, labels_mask,
            attended=encoder.apply(
                **dict_union(
                    fork.apply(mlp.apply(recordings), return_dict=True),
                    mask=recordings_mask)),
            attended_mask=recordings_mask).sum()
        batch_size = named_copy(recordings.shape[1], "batch_size")
        cost = aggregation.mean(batch_cost,  batch_size)
        cost.name = "sequence_log_likelihood"
        logger.info("Cost graph is built")

        # Fetch variables useful for debugging
        max_recording_length = named_copy(recordings.shape[0],
                                          "max_recording_length")
        max_num_phonemes = named_copy(labels.shape[0],
                                      "max_num_phonemes")
        cost_per_phoneme = named_copy(
            aggregation.mean(batch_cost, batch_size * max_num_phonemes),
            "phoneme_log_likelihood")
        cg = ComputationGraph(cost)
        energies = unpack(
            VariableFilter(application=readout.readout, name="output")(
                cg.variables),
            singleton=True)
        min_energy = named_copy(energies.min(), "min_energy")
        max_energy = named_copy(energies.max(), "max_energy")
        (bottom_output,) = VariableFilter(
            application=mlp.apply, name="output")(cg)
        (attended,) = VariableFilter(
            application=generator.transition.apply, name="attended$")(cg)
        (activations,) = VariableFilter(
            application=generator.transition.apply,
            name=transition.apply.states[0])(cg)
        (weights,) = VariableFilter(
            application=generator.cost, name="weights")(cg)
        weights1, activations1 = weights[1:], activations[1:]
        mean_activation = named_copy(abs(activations1).mean(),
                                     "mean_activation")
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
                                                Scale(0.01),
                                                Momentum(0.00)]))

        observables = [
            cost, cost_per_phoneme,
            min_energy, max_energy,
            mean_activation, mean_attended, mean_bottom_output,
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
            build_stream(timit_valid, 100), prefix="valid",
            before_first_epoch=True, on_resumption=True,
            after_every_epoch=True)
        main_loop = MainLoop(
            model=bricks,
            data_stream=build_stream(timit, 10),
            algorithm=algorithm,
            extensions=([LoadFromDump(from_dump)] if from_dump else []) +
            [Timing(),
                TrainingDataMonitoring(observables, after_every_batch=True),
                average, validation,
                FinishAfter(after_n_batches=num_batches)
                .add_condition(
                    "after_batch",
                    lambda log:
                        math.isnan(log.current_row.total_gradient_norm)),
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
