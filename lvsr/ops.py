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
        """Returns all arcs of the state i"""
        return self.fst[i]


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

    def _get_next_state(self, state, input_):
        arcs = {arc.ilabel: arc for arc in self.fst[state]}
        fst_input_ = self.remap_table[input_]
        return arcs.get(fst_input_, 0)

    def perform(self, node, inputs, output_storage):
        all_states, all_inputs = inputs

        next_states = []
        for state, input_ in equizip(all_states, all_inputs):
            nextstate = self._get_next_state(state, input_)
            next_states.append(nextstate)

        new_state = output_storage[0]
        new_state[0] = numpy.array(next_states, dtype='int64')

    def make_node(self, state, input_):
        # check that the theano version has support for __props__
        assert hasattr(self, '_props')
        state = theano.tensor.as_tensor_variable(state)
        input_ = theano.tensor.as_tensor_variable(input_)
        return theano.Apply(self, [state, input_], [state.type(), input_.type()])


class FSTProbabilitiesOp(Op):
    """Returns transition log probabilities for all possible input symbols."""
    __props__ = ()
    max_prob = 1e+12

    def __init__(self, fst):
        self.fst = fst

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
