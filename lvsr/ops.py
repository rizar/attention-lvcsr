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


def read_symbols(fname):
    syms = fst.SymbolTable('eps')
    with open(fname) as sf:
        for line in sf:
            s,i = line.strip().split()
            syms[s] = int(i)
    return syms


@do_not_pickle_attributes('fst')
class FST(object):
    EPSILON = 0

    """Picklable wrapper around FST."""
    def __init__(self, path):
        self.path = path

    def load(self):
        self.fst = fst.read(self.path)
        self.isyms = dict(self.fst.isyms.items())

    def __getitem__(self, state):
        """Returns all arcs of the state i"""
        return self.fst[state]

    def combine_weights(self, x, y):
        if x == None:
            return y
        m = max(-x, -y)
        return -m - math.log(math.exp(-x - m) + math.exp(-y - m))

    def get_arcs(self, state, character):
        return [(state, arc.nextstate, arc.ilabel, float(arc.weight))
                for arc in self[state] if arc.ilabel == character]

    def transition(self, states, character):
        arcs = list(itertools.chain(
            *[self.get_arcs(state, character) for state in states]))
        next_states = {arc[1]: None for arc in arcs}
        for next_state in list(next_states):
            for arc in arcs:
                if arc[1] == next_state:
                    next_states[next_state] = self.combine_weights(
                        next_states.get(next_state), states[arc[0]] + arc[3])
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
            for arc in self.get_arcs(state, self.EPSILON):
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
            for prev_state, weight in depends[next_state]:
                next_states[next_state] = self.combine_weights(
                    next_states.get(next_state), next_states[prev_state] + weight)

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
    start_new_word_state : int
        "Main looping state" of the FST which we enter after following backoff links
    space_idx : int
        id of the space character in network coding
    allow_spelling_unknowns : bool
        do we want to allow the net to enerate characters corresponding to unknown words

    """
    __props__ = ()

    def __init__(self, fst, remap_table, start_new_word_state, space_idx,
                 allow_spelling_unknowns):
        self.fst = fst
        self.remap_table = remap_table
        self.start_new_word_state = start_new_word_state
        self.space_idx = space_idx
        self.allow_spelling_unknowns = allow_spelling_unknowns
        if allow_spelling_unknowns:
            assert self.space_idx is not None

    def perform(self, node, inputs, output_storage):
        all_states, all_inputs = inputs

        next_states = []
        for state, input_ in equizip(all_states, all_inputs):
            #default next state if no transition is found
            next_state = self.start_new_word_state

            if self.allow_spelling_unknowns:
                next_state = -1 #special loop state that spells out letters
                if state == -1:
                    if input_ == self.space_idx:
                        next_state = self.start_new_word_state

            if state != -1:
                arcs = {arc.ilabel: arc for arc in self.fst[state]}
            else:
                arcs = {}

            fst_input_ = self.remap_table[input_]
            if fst_input_ in arcs:
                next_state = arcs[fst_input_].nextstate
            next_states.append(next_state)

        output_storage[0][0] = numpy.array(next_states, dtype='int64')

    def make_node(self, state, input_):
        # check that the theano version has support for __props__
        assert hasattr(self, '_props')
        state = theano.tensor.as_tensor_variable(state)
        input_ = theano.tensor.as_tensor_variable(input_)
        return theano.Apply(self, [state, input_], [state.type()])


class FSTProbabilitiesOp(Op):
    """Returns transition log probabilities for all possible input symbols.

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
        states, = inputs

        all_logprobs = []
        for state in states:
            if state == -1:
                logprobs = numpy.zeros(len(self.remap_table), dtype=theano.config.floatX)
            else:
                arcs = {arc.ilabel: arc for arc in self.fst[state]}
                logprobs = (numpy.ones(len(self.remap_table), dtype=theano.config.floatX)
                            * self.no_transition_cost)
                for nn_character, fst_character in self.remap_table.items():
                    if fst_character in arcs:
                        logprobs[nn_character] = (
                            arcs[fst_character].weight
                            if not self.all_weights_to_zeros
                            else 0)
            all_logprobs.append(logprobs)

        output_storage[0][0] = numpy.array(all_logprobs)

    def make_node(self, state):
        # check that the theano version has support for __props__
        assert hasattr(self, '_props')
        state = theano.tensor.as_tensor_variable(state)
        return theano.Apply(self, [state], [theano.tensor.matrix()])
