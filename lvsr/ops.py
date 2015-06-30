import fst
import numpy
import theano
from theano import Op
from fuel.utils import do_not_pickle_attributes
from picklable_itertools.extras import equizip


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

    def __getitem__(self, i):
        return self.fst[i]

    def keys(self):
        return self.fst.keys()


class FSTTransitionOp(Op):
    """Performs transition in an FST.

    Given a state and an input symbol (character) returns the next state and
    the output symbol (word)."""
    __props__ = ()

    def __init__(self, fst, symbol_table):
        self.fst = fst
        self.disambig_symbol = symbol_table['#1']

    def _get_next_state(self, state, input):
        arcs = {arc.ilabel:arc for arc in self.fst[state]}
        if int(input) in arcs:
            arc = arcs[int(input)]
            return arc.nextstate, arc.olabel
        elif self.disambig_symbol in arcs:
            arc = arcs[self.disambig_symbol]
            return arc.nextstate, arc.olabel
        else:
            # Just return state 0, output 0
            return 0, 0

    def perform(self, node, inputs, output_storage):
        all_states, all_inputs = inputs
        new_state = output_storage[0]
        output = output_storage[1]

        next_states = []
        olabels = []
        for state, input in equizip(all_states, all_inputs):
            nextstate, olabel = self._get_next_state(state, input)
            next_states.append(nextstate)
            olabels.append(olabel)

        new_state[0] = numpy.array(next_states)
        output[0] = numpy.array(olabels)


    def make_node(self, state, input):
        # check that the theano version has support for __props__
        assert hasattr(self, '_props')
        state = theano.tensor.as_tensor_variable(state)
        input = theano.tensor.as_tensor_variable(input)
        return theano.Apply(self, [state, input], [state.type(), input.type()])


class FSTProbabilitiesOp(Op):
    """Returns transition log probabilities for all possible input symbols."""
    __props__ = ()
    max_prob = 1e+12

    def __init__(self, fst, symbol_table):
        self.fst = fst
        self.symbol_table = symbol_table

    def _get_next_probs(self, state):
        arcs = {arc.ilabel: arc for arc in self.fst[state]}
        logprobs = numpy.ones(len(self.symbol_table)) * self.max_prob
        for i, (_, idx) in enumerate(self.symbol_table.items()):
            if idx in arcs:
                logprobs[i] = arcs[idx].weight
        return logprobs

    def perform(self, node, inputs, output_storage):
        states, = inputs

        all_logprobs = []
        for state in states:
            logprobs = self._get_next_probs(state)
            all_logprobs.append(logprobs)
        output_storage[0][0] = numpy.array(all_logprobs)

    def make_node(self, state):
        # check that the theano version has support for __props__
        assert hasattr(self, '_props')
        state = theano.tensor.as_tensor_variable(state)
        return theano.Apply(self, [state], [theano.tensor.matrix()])
