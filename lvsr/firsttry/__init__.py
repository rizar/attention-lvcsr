from __future__ import print_function
import logging
import pprint
import sys
import math
import os
import functools

import numpy
import theano
from theano import tensor
from blocks.bricks import Tanh, application, MLP
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, LinearReadout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.datasets import (
    DataStream,
    DataStreamMapping, BatchDataStream, PaddingDataStream)
from blocks.datasets.schemes import ConstantScheme, SequentialScheme
from blocks.algorithms import (GradientDescent, SteepestDescent,
                               GradientClipping, CompositeRule)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import SerializeMainLoop, LoadFromDump
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.plot import Plot
from blocks.main_loop import MainLoop
from blocks.select import Selector
from blocks.filter import VariableFilter
from blocks.utils import named_copy, unpack, dict_union

from lvsr.datasets import TIMIT
from lvsr.preprocessing import spectrogram

sys.setrecursionlimit(100000)
floatX = theano.config.floatX
logger = logging.getLogger(__name__)


class Transition(GatedRecurrent):
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


def main(mode, save_path, num_batches, from_dump):
    if mode == "train":
        # Experiment configuration
        dimension = 100

        # Data processing pipeline
        timit = TIMIT()
        data_stream=DataStream(
            timit, iteration_scheme=SequentialScheme(timit.num_examples, 10))
        data_stream=DataStreamMapping(
            data_stream, functools.partial(apply_preprocessing, spectrogram))
        data_stream=PaddingDataStream(data_stream)
        data_stream = DataStreamMapping(
            data_stream, switch_first_two_axes)

        # Build the model
        recordings = tensor.tensor3("recordings")
        recordings_mask = tensor.matrix("recordings_mask")
        labels = tensor.lmatrix("labels")
        labels_mask = tensor.matrix("labels_mask")

        encoder = Bidirectional(
            GatedRecurrent(dim=dimension, activation=Tanh()),
            weights_init=Orthogonal())
        encoder.initialize()
        fork = Fork([name for name in encoder.prototype.apply.sequences
                     if name != 'mask'],
                    weights_init=IsotropicGaussian(0.1),
                    biases_init=Constant(0))
        fork.input_dim = dimension
        fork.fork_dims = {name: dimension for name in fork.fork_names}
        mlp = MLP([Tanh()], [129, dimension], name="bottom",
                  weights_init=IsotropicGaussian(0.1),
                  biases_init=Constant(0))
        transition = Transition(
            activation=Tanh(),
            dim=dimension, attended_dim=2 * dimension, name="transition")
        attention = SequenceContentAttention(
            state_names=transition.apply.states,
            match_dim=dimension, name="attention")
        readout = LinearReadout(
            readout_dim=timit.num_phonemes, source_names=["states"],
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

        # Give an idea of what's going on
        params = Selector(bricks).get_params()
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in params.items()],
                        width=120))

        # Build the cost computation graph
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
        (activations,) = VariableFilter(
            application=generator.transition.apply,
            name="states")(cg.variables)
        mean_activation = named_copy(activations.mean(), "mean_activation")

        # Define the training algorithm.
        algorithm = GradientDescent(
            cost=cost, step_rule=CompositeRule([GradientClipping(10.0),
                                                SteepestDescent(0.01)]))

        observables = [
            cost, cost_per_phoneme,
            min_energy, max_energy, mean_activation,
            batch_size, max_recording_length, max_num_phonemes,
            algorithm.total_step_norm, algorithm.total_gradient_norm]
        for name, param in params.items():
            observables.append(named_copy(
                param.norm(2), name + "_norm"))
            observables.append(named_copy(
                algorithm.gradients[param].norm(2), name + "_grad_norm"))

        main_loop = MainLoop(
            model=bricks,
            data_stream=data_stream,
            algorithm=algorithm,
            extensions=([LoadFromDump(from_dump)] if from_dump else []) +
            [Timing(),
                TrainingDataMonitoring(observables, after_every_batch=True),
                TrainingDataMonitoring(observables, prefix="average",
                                       every_n_batches=10),
                FinishAfter(after_n_batches=num_batches)
                .add_condition(
                    "after_batch",
                    lambda log:
                        math.isnan(log.current_row.total_gradient_norm)),
                Plot(os.path.basename(save_path),
                     [["average_" + cost.name],
                      ["average_" + cost_per_phoneme.name]],
                     every_n_batches=10),
                SerializeMainLoop(save_path, every_n_batches=500,
                                  save_separately=["model", "log"]),
                Printing(every_n_batches=1)])
        main_loop.run()
    elif mode == "test":
        raise NotImplemented
