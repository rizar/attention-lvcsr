import subprocess
from theano import tensor, function
from lvsr.ops import FSTProbabilitiesOp, FSTTransitionOp, FST, read_symbols


def test_fst_transition_op():
    x = tensor.iscalar('x')
    s = tensor.iscalar('s')

    subprocess.call('lm2fst.sh tests/simple_lm.arpa', shell=True)
    fst = FST('LG.fst')
    characters = read_symbols('characters.txt')

    new_s, new_x = FSTTransitionOp(fst, characters)(s, x)

    f = function([s, x], [new_s, new_x])

    assert f(0, 2) == [1, 0]


def test_fst_probabilities_op():
    s = tensor.iscalar('s')

    subprocess.call('lm2fst.sh tests/simple_lm.arpa', shell=True)
    fst = FST('LG.fst')
    words = read_symbols('words.txt')

    probs = FSTProbabilitiesOp(fst, words)(s)

    f = function([s], probs)

    f(0)
