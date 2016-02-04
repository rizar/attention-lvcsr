"""The beam search module."""
from collections import OrderedDict
from six.moves import range

import numpy
from picklable_itertools.extras import equizip
from theano import config, function, tensor

from blocks.bricks.sequence_generators import BaseSequenceGenerator
from blocks.filter import VariableFilter, get_application_call, get_brick
from blocks.graph import ComputationGraph
from blocks.roles import INPUT, OUTPUT


class CandidateNotFoundError(Exception):
    pass


class BeamSearch(object):
    """Approximate search for the most likely sequence.

    Beam search is an approximate algorithm for finding :math:`y^* =
    argmax_y P(y|c)`, where :math:`y` is an output sequence, :math:`c` are
    the contexts, :math:`P` is the output distribution of a
    :class:`.SequenceGenerator`. At each step it considers :math:`k`
    candidate sequence prefixes. :math:`k` is called the beam size, and the
    sequence are called the beam. The sequences are replaced with their
    :math:`k` most probable continuations, and this is repeated until
    end-of-line symbol is met.

    The beam search compiles quite a few Theano functions under the hood.
    Normally those are compiled at the first :meth:`search` call, but
    you can also explicitly call :meth:`compile`.

    Parameters
    ----------
    beam_size : int
        The beam size.
    samples : :class:`~theano.Variable`
        An output of a sampling computation graph built by
        :meth:`~blocks.brick.SequenceGenerator.generate`, the one
        corresponding to sampled sequences.

    See Also
    --------
    :class:`.SequenceGenerator`

    Notes
    -----
    Sequence generator should use an emitter which has `probs` method
    e.g. :class:`SoftmaxEmitter`.

    Does not support dummy contexts so far (all the contexts must be used
    in the `generate` method of the sequence generator for the current code
    to work).

    """
    def __init__(self, beam_size, samples):
        self.beam_size = beam_size

        # Extracting information from the sampling computation graph
        cg = ComputationGraph(samples)
        self.inputs = cg.inputs
        self.generator = get_brick(samples)
        if not isinstance(self.generator, BaseSequenceGenerator):
            raise ValueError
        self.generate_call = get_application_call(samples)
        if (not self.generate_call.application ==
                self.generator.generate):
            raise ValueError
        self.inner_cg = ComputationGraph(self.generate_call.inner_outputs)

        # Fetching names from the sequence generator
        self.context_names = self.generator.generate.contexts
        self.state_names = self.generator.generate.states

        # Parsing the inner computation graph of sampling scan
        self.contexts = [
            VariableFilter(bricks=[self.generator],
                           name=name,
                           roles=[INPUT])(self.inner_cg)[0]
            for name in self.context_names]
        self.input_states = []
        # Includes only those state names that were actually used
        # in 'generate'
        self.input_state_names = []
        for name in self.generator.generate.states:
            var = VariableFilter(
                bricks=[self.generator], name=name,
                roles=[INPUT])(self.inner_cg)
            if var:
                self.input_state_names.append(name)
                self.input_states.append(var[0])

        self.compiled = False

    def _compile_context_computer(self):
        self.context_computer = function(
            self.inputs, self.contexts, on_unused_input='ignore')

    def _compile_initial_state_computer(self):
        # TODO: should be now extractable from the computation graph
        initial_states = self.generator.initial_states(
                1, as_dict=True,
                **dict(equizip(self.context_names, self.contexts)))
        self.initial_state_computer = function(
            self.contexts, initial_states, on_unused_input='ignore')

    def _compile_next_state_computer(self):
        next_states = [VariableFilter(bricks=[self.generator],
                                      name=name,
                                      roles=[OUTPUT])(self.inner_cg)[-1]
                       for name in self.state_names]
        next_outputs = VariableFilter(
            applications=[self.generator.readout.emit], roles=[OUTPUT])(
                self.inner_cg.variables)
        self.next_state_computer = function(
            self.contexts + self.input_states + next_outputs, next_states,
            # This is temporarily required because `lm_logprobs` is a weird
            # state which is not used to compute next state, but used to
            # compute the next output.
            on_unused_input='ignore')

    def _compile_logprobs_computer(self):
        # This filtering should return identical variables
        # (in terms of computations) variables, and we do not care
        # which to use.
        readouts = VariableFilter(
            applications=[self.generator.readout.readout],
            roles=[OUTPUT])(self.inner_cg)[0]
        costs = self.generator.readout.costs(readouts)
        self.logprobs_computer = function(
            self.contexts + self.input_states, costs,
            on_unused_input='ignore')

    def compile(self):
        """Compile all Theano functions used."""
        self._compile_context_computer()
        self._compile_initial_state_computer()
        self._compile_next_state_computer()
        self._compile_logprobs_computer()
        self.compiled = True

    def compute_contexts(self, inputs):
        """Computes contexts from inputs.

        Parameters
        ----------
        inputs : dict
            Dictionary of input arrays.

        Returns
        -------
        A {name: :class:`numpy.ndarray`} dictionary of contexts ordered
        like `self.context_names`.

        """
        contexts = self.context_computer(*[inputs[var]
                                           for var in self.inputs])
        return OrderedDict(equizip(self.context_names, contexts))

    def compute_initial_states(self, contexts):
        """Computes initial states.

        Parameters
        ----------
        contexts : dict
            A {name: :class:`numpy.ndarray`} dictionary of contexts.

        Returns
        -------
        A {name: :class:`numpy.ndarray`} dictionary of states ordered like
        `self.state_names`.

        """
        return self.initial_state_computer(*list(contexts.values()))

    def compute_logprobs(self, contexts, states):
        """Compute log probabilities of all possible outputs.

        Parameters
        ----------
        contexts : dict
            A {name: :class:`numpy.ndarray`} dictionary of contexts.
        states : dict
            A {name: :class:`numpy.ndarray`} dictionary of states.

        Returns
        -------
        A :class:`numpy.ndarray` of the (beam size, number of possible
        outputs) shape.

        """
        input_states = [states[name] for name in self.input_state_names]
        return self.logprobs_computer(*(list(contexts.values()) +
                                      input_states))

    def compute_next_states(self, contexts, states, outputs):
        """Computes next states.

        Parameters
        ----------
        contexts : dict
            A {name: :class:`numpy.ndarray`} dictionary of contexts.
        states : dict
            A {name: :class:`numpy.ndarray`} dictionary of states.
        outputs : :class:`numpy.ndarray`
            A :class:`numpy.ndarray` of this step outputs.

        Returns
        -------
        A {name: numpy.array} dictionary of next states.

        """
        input_states = [states[name] for name in self.input_state_names]
        next_values = self.next_state_computer(*(list(contexts.values()) +
                                                 input_states + [outputs]))
        return OrderedDict(equizip(self.state_names, next_values))

    @staticmethod
    def _smallest(matrix, k):
        """Find k smallest elements of a matrix.

        Parameters
        ----------
        matrix : :class:`numpy.ndarray`
            The matrix.
        k : int
            The number of smallest elements required.

        Returns
        -------
        Tuple of ((row numbers, column numbers), values).

        """
        flatten = matrix.flatten()
        if flatten.shape[0] > k:
            args = numpy.argpartition(flatten, k)[:k]
        else:
            args = numpy.arange(flatten.shape[0])
        args = args[numpy.argsort(flatten[args])]
        return numpy.unravel_index(args, matrix.shape), flatten[args]

    def search(self, input_values, eol_symbol, max_length,
               ignore_first_eol=False, as_arrays=False,
               char_discount=0, round_to_inf=1e9,
               stop_on='patience',
               validate_solution_function=None):
        """Performs beam search.

        If the beam search was not compiled, it also compiles it.

        Parameters
        ----------
        input_values : dict
            A {:class:`~theano.Variable`: :class:`~numpy.ndarray`}
            dictionary of input values. The shapes should be
            the same as if you ran sampling with batch size equal to
            `beam_size`. Put it differently, the user is responsible
            for duplicaling inputs necessary number of times, because
            this class has insufficient information to do it properly.
        eol_symbol : int
            End of sequence symbol, the search stops when the symbol is
            generated.
        max_length : int
            Maximum sequence length, the search stops when it is reached.
        ignore_first_eol : bool, optional
            When ``True``, the end if sequence symbol generated at the
            first iteration are ignored. This useful when the sequence
            generator was trained on data with identical symbols for
            sequence start and sequence end.
        as_arrays : bool, optional
            If ``True``, the internal representation of search results
            is returned, that is a (matrix of outputs, mask,
            costs of all generated outputs) tuple.

        Returns
        -------
        outputs : list of lists of ints
            A list of the `beam_size` best sequences found in the order
            of decreasing likelihood.
        costs : list of floats
            A list of the costs for the `outputs`, where cost is the
            negative log-likelihood.

        """
        if not self.compiled:
            self.compile()

        contexts = self.compute_contexts(input_values)
        large_contexts = OrderedDict(contexts)
        states = self.compute_initial_states(contexts)

        # This array will store all generated outputs, including those from
        # previous step and those from already finished sequences.
        all_outputs = states['outputs'][None, :]
        all_costs = numpy.zeros_like(all_outputs, dtype=config.floatX)

        done = []
        min_cost = 1000

        for i in range(max_length):
            if len(states.values()[0].flatten()) == 0:
                break

            if stop_on == 'patience':
                done = sorted(done, key=lambda x: x[1][-1] - char_discount * len(x[1]))
                done = done[:self.beam_size]
                if done:
                    current_best_cost = done[0][1][-1] - char_discount * len(done[0][1])
                    if current_best_cost < min_cost:
                        min_cost = current_best_cost
                        patience = 30
                    else:
                        patience -= 1
                        if patience == 0:
                            break
            elif stop_on == 'optimistic_future_cost':
                # stop only when we have at least self.beam_size sequences,
                # that are all cheaper than we can possibly obtain by extending
                # other ones
                if (len(done) >= self.beam_size):
                    optimistic_future_cost = (all_costs[-1, :].min() -
                                              char_discount * max_length)
                    last_in_done = done[self.beam_size - 1][1]
                    # note: done is sorted by the cost with char discount subtracted
                    last_in_done_cost = (last_in_done[-1] -
                                         char_discount * len(last_in_done))
                    if last_in_done_cost < optimistic_future_cost:
                        break
            else:
                raise ValueError('Unknown stopping criterion {}'.format(stop_on))

            # We carefully hack values of the `logprobs` array to ensure
            # that all finished sequences are continued with `eos_symbol`.
            if large_contexts.values()[0].shape[1] != states.values()[0].shape[0]:
                for name, ctx in contexts.items():
                    large_contexts[name] = numpy.take(ctx, [0]*states.values()[0].shape[0], axis=1)
            logprobs = self.compute_logprobs(large_contexts, states)
            assert numpy.isfinite(logprobs).all()
            next_costs = (all_costs[-1, :, None] + logprobs)

            (indexes, outputs), chosen_costs = self._smallest(
                next_costs, self.beam_size)

            # Rearrange everything
            for name in states:
                states[name] = numpy.take(states[name], indexes, axis=0)
            all_outputs = numpy.take(all_outputs, indexes, axis=1)
            all_costs = numpy.take(all_costs, indexes, axis=1)

            # Record chosen output and compute new states
            if large_contexts.values()[0].shape[1] != states.values()[0].shape[0]:
                for name, ctx in contexts.items():
                    large_contexts[name] = numpy.take(ctx, [0]*states.values()[0].shape[0], axis=1)
            states = self.compute_next_states(large_contexts, states, outputs)

            all_outputs = numpy.vstack([all_outputs, outputs[None, :]])
            all_costs = numpy.vstack([all_costs, chosen_costs[None, :]])

            mask = outputs != eol_symbol
            if ignore_first_eol and i == 0:
                mask[:] = 1

            for idx in numpy.where(
                    (all_outputs[-1] == eol_symbol) &
                    (all_costs[-1] - all_costs[-2] < round_to_inf))[0]:
                if (validate_solution_function is None or
                        validate_solution_function(input_values,
                                                   all_outputs[:, idx])):
                    done.append((all_outputs[:, idx], all_costs[:, idx]))

            unfinished = numpy.where(mask == 1)[0]
            for name in states:
                states[name] = numpy.take(states[name], unfinished, axis=0)
            all_outputs = numpy.take(all_outputs, unfinished, axis=1)
            all_costs = numpy.take(all_costs, unfinished, axis=1)

        if not done:
            raise CandidateNotFoundError()

        done = sorted(done, key=lambda x: x[1][-1] - char_discount * len(x[1]))

        max_len = max((seq[0].shape[0] for seq in done))
        all_outputs = numpy.zeros((max_len, len(done)))
        all_masks = numpy.zeros((max_len, len(done)))
        all_costs = numpy.zeros((max_len, len(done)))
        for i, (seq, cost) in enumerate(done):
            all_outputs[:len(seq), i] = seq
            all_masks[:len(seq), i] = 1
            all_costs[:len(cost), i] = cost
            all_costs[len(cost):, i] = cost[-1]
        all_outputs = all_outputs[1:]
        all_masks = all_masks[1:]
        all_costs = all_costs[1:] - all_costs[:-1]
        result = all_outputs, all_masks, all_costs
        if as_arrays:
            return result
        return self.result_to_lists(result)

    @staticmethod
    def result_to_lists(result):
        outputs, masks, costs = [array.T for array in result]
        outputs = [list(output[:mask.sum()])
                   for output, mask in equizip(outputs, masks)]
        costs = list(costs.T.sum(axis=0))
        return outputs, costs
