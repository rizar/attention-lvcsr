import fst
import numpy
import theano
from theano import Op
from fuel.utils import do_not_pickle_attributes


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

    def perform(self, node, inputs, output_storage):
        state, input = inputs
        new_state = output_storage[0]
        output = output_storage[1]
        arcs = {arc.ilabel:arc for arc in self.fst[state]}
        if int(input) in arcs:
            arc = arcs[int(input)]
            nextstate = arc.nextstate
            olabel = arc.olabel
        elif self.disambig_symbol in arcs:
            arc = arcs[self.disambig_symbol]
            nextstate = arc.nextstate
            olabel = arc.olabel
        else:
            nextstate = 0
            olabel = 0
        new_state[0] = nextstate
        output[0] = olabel


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

    def perform(self, node, inputs, output_storage):
        state, = inputs
        arcs = {arc.ilabel: arc for arc in self.fst[state]}
        logprobs = numpy.ones(len(self.symbol_table)) * self.max_prob
        for i, (_, idx) in enumerate(self.symbol_table.items()):
            if idx in arcs:
                logprobs[i] = arcs[idx].weight
        output_storage[0][0] = logprobs

    def make_node(self, state):
        # check that the theano version has support for __props__
        assert hasattr(self, '_props')
        state = theano.tensor.as_tensor_variable(state)
        return theano.Apply(self, [state], [theano.tensor.vector()])
