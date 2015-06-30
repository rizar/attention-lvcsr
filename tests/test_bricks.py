import subprocess
from theano import tensor, function
from lvsr.bricks import FSTTransition
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
