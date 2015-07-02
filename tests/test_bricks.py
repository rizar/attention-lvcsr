import numpy
import subprocess
import theano
from numpy.testing import assert_allclose
from theano import tensor, function

from blocks.bricks import Tanh, Identity
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback, TrivialEmitter, TrivialFeedback)
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.roles import AUXILIARY

from lvsr.bricks import FSTTransition, FSTReadout, ShallowFusionReadout, ShallowFusionEmitter
from lvsr.ops import FST, read_symbols


def test_fst_transition():
    subprocess.call('lm2fst.sh tests/simple_lm.arpa', shell=True)
    fst = FST('LG.fst')
    words = read_symbols('words.txt')

    x = tensor.lmatrix('x')
    states, output, weight = FSTTransition(fst, words).apply(x)

    f = function([x], [states, output])

    characters = read_symbols('characters.txt')
    out = f([[characters['C']],
             [characters['a']],
             [characters['y']],
             [characters['#0']]])
    assert words.find(out[1][0][0]) == 'Cay'


def test_fst_sequence_generator():
    floatX = theano.config.floatX
    rng = numpy.random.RandomState(1234)

    feedback_dim = 3
    dim = 20
    batch_size = 30
    n_steps = 10

    transition = GatedRecurrent(dim=dim, activation=Tanh(),
                                weights_init=Orthogonal())
    fst = FST('LG.fst')
    words = read_symbols('words.txt')
    readout_dim = len(words)
    lm_transition = FSTTransition(fst, words)
    normal_inputs = [name for name in lm_transition.apply.sequences
                     if 'mask' not in name]
    language_model = SequenceGenerator(
        FSTReadout(readout_dim=readout_dim, source_names=['weights'],
                   weights_dim=readout_dim),
        lm_transition,
        fork=Fork(normal_inputs, 0, prototype=Identity()),
        name='language_model')
    generator = SequenceGenerator(
        ShallowFusionReadout(
            readout_dim=readout_dim, source_names=["states"],
            feedback_brick=LookupFeedback(readout_dim, feedback_dim),
            lm_weights='lm_weights',
            beta=0.5),
        transition,
        language_model=language_model,
        weights_init=IsotropicGaussian(0.1), biases_init=Constant(0),
        seed=1234)
    generator.initialize()

    # Test 'cost_matrix' method
    y = tensor.lmatrix('y')
    y.tag.test_value = numpy.zeros((15, batch_size), dtype='int64')
    mask = tensor.matrix('mask')
    mask.tag.test_value = numpy.ones((15, batch_size))
    cost = language_model.cost_matrix(y, mask)

    costs = generator.cost_matrix(y, mask)
    assert costs.ndim == 2
    costs_fun = theano.function([y, mask], [costs])
    y_test = rng.randint(readout_dim, size=(n_steps, batch_size))
    m_test = numpy.ones((n_steps, batch_size), dtype=floatX)
    costs_val = costs_fun(y_test, m_test)[0]
    assert costs_val.shape == (n_steps, batch_size)
    assert_allclose(costs_val.sum(), 585.809, rtol=1e-5)

    # Test 'cost' method
    cost = generator.cost(y, mask)
    assert cost.ndim == 0
    cost_val = theano.function([y, mask], cost)(y_test, m_test)
    assert_allclose(cost_val, 19.5269, rtol=1e-5)

    # Test 'AUXILIARY' variable 'per_sequence_element' in 'cost' method
    cg = ComputationGraph([cost])
    var_filter = VariableFilter(roles=[AUXILIARY])
    aux_var_name = '_'.join([generator.name, generator.cost.name,
                             'per_sequence_element'])
    cost_per_el = [el for el in var_filter(cg.variables)
                   if el.name == aux_var_name][0]
    assert cost_per_el.ndim == 0
    cost_per_el_val = theano.function([y, mask], [cost_per_el])(y_test, m_test)
    assert_allclose(cost_per_el_val, 1.95269, rtol=1e-5)

    # Test generate
    states, outputs, lm_states, costs = generator.generate(
        iterate=True, batch_size=batch_size, n_steps=n_steps)
    cg = ComputationGraph([states, outputs, costs])
    states_val, outputs_val, costs_val = theano.function(
        [], [states, outputs, costs],
        updates=cg.updates)()
    assert states_val.shape == (n_steps, batch_size, dim)
    assert outputs_val.shape == (n_steps, batch_size)
    assert outputs_val.dtype == 'int64'
    assert costs_val.shape == (n_steps, batch_size)
    assert_allclose(states_val.sum(), -4.88367, rtol=1e-5)
    assert_allclose(costs_val.sum(), 486.681, rtol=1e-5)
    assert outputs_val.sum() == 627

    # Test masks agnostic results of cost
    cost1 = costs_fun([[1], [2]], [[1], [1]])[0]
    cost2 = costs_fun([[3, 1], [4, 2], [2, 0]],
                      [[1, 1], [1, 1], [1, 0]])[0]
    assert_allclose(cost1.sum(), cost2[:, 1].sum(), rtol=1e-5)

test_fst_sequence_generator()
