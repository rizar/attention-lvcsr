from __future__ import print_function
import math
try:
    import fst
except ImportError:
    print("No PyFST module, trying to work without it. If you want to run the "
          "language model, please install openfst and PyFST")
import numpy
import theano
import itertools
from theano import tensor, Op
from theano.gradient import disconnected_type
from fuel.utils import do_not_pickle_attributes
from picklable_itertools.extras import equizip
from collections import defaultdict, deque

from toposort import toposort_flatten

from lvsr.error_rate import reward_matrix, gain_matrix


EPSILON = 0
MAX_STATES = 7
NOT_STATE = -1


def read_symbols(fname):
    syms = fst.SymbolTable('eps')
    with open(fname) as sf:
        for line in sf:
            s,i = line.strip().split()
            syms[s] = int(i)
    return syms


@do_not_pickle_attributes('fst')
class FST(object):

    """Picklable wrapper around FST."""
    def __init__(self, path):
        self.path = path

    def load(self):
        self.fst = fst.read(self.path)
        self.isyms = dict(self.fst.isyms.items())

    def __getitem__(self, state):
        """Returns all arcs of the state i"""
        return self.fst[state]

    def combine_weights(self, *args):
        # Protection from underflow when -x is too small
        m = max(args)
        return m - math.log(sum(math.exp(m - x) for x in args if x is not None))

    def get_arcs(self, state, character):
        return [(state, arc.nextstate, arc.ilabel, float(arc.weight))
                for arc in self[state] if arc.ilabel == character]

    def transition(self, states, character):
        arcs = list(itertools.chain(
            *[self.get_arcs(state, character) for state in states]))
        next_states = {}
        for next_state in {arc[1] for arc in arcs}:
            next_states[next_state] = self.combine_weights(
                *[states[arc[0]] + arc[3] for arc in arcs
                  if arc[1] == next_state])
        return next_states

    def expand(self, states):
        seen = set()
        depends = defaultdict(list)
        queue = deque()
        for state in states:
            queue.append(state)
            seen.add(state)
        while len(queue):
            state = queue.popleft()
            for arc in self.get_arcs(state, EPSILON):
                depends[arc[1]].append((arc[0], arc[3]))
                if arc[1] in seen:
                    continue
                queue.append(arc[1])
                seen.add(arc[1])

        depends_for_toposort = {key: {state for state, weight in value}
                                for key, value in depends.items()}
        order = toposort_flatten(depends_for_toposort)

        next_states = states
        for next_state in order:
            next_states[next_state] = self.combine_weights(
                *([next_states.get(next_state)] +
                  [next_states[prev_state] + weight
                   for prev_state, weight in depends[next_state]]))

        return next_states

    def explain(self, input_):
        input_ = list(input_)
        states = {self.fst.start: 0}
        print("Initial states: {}".format(states))
        states = self.expand(states)
        print("Expanded states: {}".format(states))

        for char, ilabel in zip(input_, [self.isyms[char] for char in input_]):
            states = self.transition(states, ilabel)
            print("{} consumed: {}".format(char, states))
            states = self.expand(states)
            print("Expanded states: {}".format(states))

        result = None
        for state, weight in states.items():
            if numpy.isfinite(weight + float(self.fst[state].final)):
                print("Finite state {} with path weight {} and its own weight {}".format(
                    state, weight, self.fst[state].final))
                result = self.combine_weights(
                    result, weight + float(self.fst[state].final))

        print("Total weight: {}".format(result))
        return result


class FSTTransitionOp(Op):
    """Performs transition in an FST.

    Given a state and an input symbol (character) returns the next state.

    Parameters
    ----------
    fst : FST instance
    remap_table : dict
        Maps neutral network characters to FST characters.

    """
    __props__ = ()


    def __init__(self, fst, remap_table):
        self.fst = fst
        self.remap_table = remap_table

    def pad(self, arr, value):
        return numpy.pad(arr, (0, MAX_STATES - len(arr)),
                         mode='constant', constant_values=value)

    def perform(self, node, inputs, output_storage):
        all_states, all_weights, all_inputs = inputs
        # Each row of all_states contains a set of states
        # padded with NOT_STATE.

        all_next_states = []
        all_next_weights = []
        for states, weights, input_ in equizip(all_states, all_weights, all_inputs):
            states_dict = dict(zip(states, weights))
            del states_dict[NOT_STATE]
            next_states_dict = self.fst.transition(
                states_dict, self.remap_table[input_])
            next_states_dict = self.fst.expand(next_states_dict)
            if next_states_dict:
                next_states, next_weights = zip(*next_states_dict.items())
            else:
                # No adequate state when no arc exists for now
                next_states, next_weights = [], []
            all_next_states.append(self.pad(next_states, NOT_STATE))
            all_next_weights.append(self.pad(next_weights, 0))

        output_storage[0][0] = numpy.array(all_next_states, dtype='int64')
        output_storage[1][0] = numpy.array(all_next_weights)

    def make_node(self, states, weights, input_):
        # check that the theano version has support for __props__
        assert hasattr(self, '_props')
        states = theano.tensor.as_tensor_variable(states)
        weights = theano.tensor.as_tensor_variable(weights)
        input_ = theano.tensor.as_tensor_variable(input_)
        return theano.Apply(self,
            [states, weights, input_],
            [states.type(), weights.type()])


class FSTCostsOp(Op):
    """Returns transition costs for all possible input symbols.

    Parameters
    ----------
    fst : FST instance
    remap_table : dict
        Maps neutral network characters to FST characters.
    no_transition_cost : float
        Cost of going to the start state when no arc for an input
        symbol is available.

    Notes
    -----
    It is assumed that neural network characters start from zero.

    """
    __props__ = ()

    def __init__(self, fst, remap_table, no_transition_cost):
        self.fst = fst
        self.remap_table = remap_table
        self.no_transition_cost = no_transition_cost

    def perform(self, node, inputs, output_storage):
        all_states, all_weights = inputs

        all_costs = []
        for states, weights in zip(all_states, all_weights):
            states_dict = dict(zip(states, weights))
            del states_dict[NOT_STATE]
            costs = (numpy.ones(len(self.remap_table), dtype=theano.config.floatX)
                     * self.no_transition_cost)
            if states_dict:
                total_weight = self.fst.combine_weights(*states_dict.values())
                for nn_character, fst_character in self.remap_table.items():
                    next_states_dict = self.fst.transition(states_dict, fst_character)
                    next_states_dict = self.fst.expand(next_states_dict)
                    if next_states_dict:
                        next_total_weight = self.fst.combine_weights(*next_states_dict.values())
                        costs[nn_character] = next_total_weight - total_weight
            all_costs.append(costs)

        output_storage[0][0] = numpy.array(all_costs)

    def make_node(self, states, weights):
        # check that the theano version has support for __props__
        assert hasattr(self, '_props')
        states = theano.tensor.as_tensor_variable(states)
        weights = theano.tensor.as_tensor_variable(weights)
        return theano.Apply(self,
            [states, weights], [theano.tensor.matrix()])


class RewardOp(Op):
    __props__ = ()

    def __init__(self, eos_label, alphabet_size):
        """Computes matrices of rewards and gains."""
        self.eos_label = eos_label
        self.alphabet_size = alphabet_size

    def perform(self, node, inputs, output_storage):
        groundtruth, recognized = inputs
        if (groundtruth.ndim != 2 or recognized.ndim != 2
                or groundtruth.shape[1] != recognized.shape[1]):
            raise ValueError
        batch_size = groundtruth.shape[1]
        all_rewards = numpy.zeros(
            recognized.shape + (self.alphabet_size,), dtype='int64')
        all_gains = numpy.zeros(
            recognized.shape + (self.alphabet_size,), dtype='int64')
        alphabet = list(range(self.alphabet_size))
        for index in range(batch_size):
            y = list(groundtruth[:, index])
            y_hat = list(recognized[:, index])
            try:
                eos_pos = y.index(self.eos_label)
                y = y[:eos_pos + 1]
            except:
                # Sometimes groundtruth is in fact also a prediction
                # and in this case it might not have EOS label
                pass
            if self.eos_label in y_hat:
                y_hat_eos_pos = y_hat.index(self.eos_label)
                y_hat_trunc = y_hat[:y_hat_eos_pos + 1]
            else:
                y_hat_trunc = y_hat
            rewards_trunc = reward_matrix(
                y, y_hat_trunc, alphabet, self.eos_label)
            # pass freshly computed rewards to gain_matrix to speed things up
            # a bit
            gains_trunc = gain_matrix(y, y_hat_trunc, alphabet,
                                      given_reward_matrix=rewards_trunc)
            gains = numpy.ones((len(y_hat), len(alphabet))) * -1000
            gains[:(gains_trunc.shape[0] - 1), :] = gains_trunc[:-1, :]

            rewards = numpy.ones((len(y_hat), len(alphabet))) * -1
            rewards[:(rewards_trunc.shape[0] - 1), :] = rewards_trunc[:-1, :]
            all_rewards[:, index, :] = rewards
            all_gains[:, index, :] = gains

        output_storage[0][0] = all_rewards
        output_storage[1][0] = all_gains

    def grad(self, *args, **kwargs):
        return disconnected_type(), disconnected_type()

    def make_node(self, groundtruth, recognized):
        recognized = tensor.as_tensor_variable(recognized)
        groundtruth = tensor.as_tensor_variable(groundtruth)
        return theano.Apply(
            self, [groundtruth, recognized], [tensor.ltensor3(), tensor.ltensor3()])
