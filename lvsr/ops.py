import fst
import numpy
import theano
import itertools
import math
from theano import Op
from fuel.utils import do_not_pickle_attributes
from picklable_itertools.extras import equizip
from Queue import Queue
from collections import defaultdict

from toposort import toposort_flatten

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
        x = numpy.array(filter(numpy.isfinite, args))
        # Protection from underflow when -x is too small
        m = x.max()
        return m - numpy.log(numpy.exp(-x + m).sum())

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
        queue = Queue()
        for state in states:
            queue.put(state)
            seen.add(state)
        while not queue.empty():
            state = queue.get()
            for arc in self.get_arcs(state, EPSILON):
                depends[arc[1]].append((arc[0], arc[3]))
                if arc[1] in seen:
                    continue
                queue.put(arc[1])
                seen.add(arc[1])

        depends_for_toposort = {key: {state for state, weight in value}
                                for key, value in depends.items()}
        order = toposort_flatten(depends_for_toposort)

        next_states = states
        for next_state in order:
            next_states[next_state] = self.combine_weights(
                *([next_states.get(next_state, numpy.Inf)] +
                  [next_states[prev_state] + weight
                   for prev_state, weight in depends[next_state]]))

        return next_states

    def explain(self, input_):
        input_ = ['<eol>'] + list(input_) + ['<eol>']
        states = {self.fst.start: 0}

        print("Initial states: {}".format(states))
        for char, ilabel in zip(input_, [self.isyms[char] for char in input_]):
            states = self.expand(states)
            print("Expanded states: {}".format(states))
            states = self.transition(states, ilabel)
            print("{} consumed: {}".format(char, states))
        states = self.expand(states)
        print("Expanded states: {}".format(states))

        result = numpy.Inf
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


    def __init__(self, fst, remap_table, start_new_word_state, space_idx,
                 allow_spelling_unknowns):
        self.fst = fst
        self.remap_table = remap_table
        assert not allow_spelling_unknowns

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
            next_states_dict = self.fst.expand(states_dict)
            next_states_dict = self.fst.transition(
                next_states_dict, self.remap_table[input_])
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
    all_weights_to_zero : bool
        Ignore all weights as if they all were zeros.

    Notes
    -----
    It is assumed that neural network characters start from zero.

    """
    __props__ = ()

    def __init__(self, fst, remap_table, no_transition_cost, all_weights_to_zeros):
        self.fst = fst
        self.remap_table = remap_table
        self.no_transition_cost = no_transition_cost
        self.all_weights_to_zeros = all_weights_to_zeros

    def perform(self, node, inputs, output_storage):
        all_states, all_weights = inputs

        all_costs = []
        for states, weights in zip(all_states, all_weights):
            states_dict = dict(zip(states, weights))
            del states_dict[NOT_STATE]
            costs = (numpy.ones(len(self.remap_table), dtype=theano.config.floatX)
                     * self.no_transition_cost)
            if states_dict:
                total_weight = self.fst.combine_weights(states_dict.values())
                for nn_character, fst_character in self.remap_table.items():
                    next_states_dict = self.fst.expand(states_dict)
                    next_states_dict = self.fst.transition(next_states_dict, fst_character)
                    if next_states_dict:
                        next_total_weight = self.fst.combine_weights(next_states_dict.values())
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
