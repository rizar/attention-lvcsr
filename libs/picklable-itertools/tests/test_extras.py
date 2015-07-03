from unittest import SkipTest
from nose.tools import assert_raises
from six.moves import zip
from picklable_itertools.extras import (partition, partition_all,
                                        IterableLengthMismatch, equizip,
                                        interleave, roundrobin)

from . import verify_same, verify_pickle


def test_partition():
    try:
        from toolz import itertoolz
    except ImportError:
        raise SkipTest()
    for obj, ref in zip([partition, partition_all],
                        [itertoolz.partition, itertoolz.partition_all]):
        yield verify_same, obj, ref, None, 2, [5, 9, 2, 6]
        yield verify_same, obj, ref, None, 2, [5, 9, 2], 3
        yield verify_same, obj, ref, None, 3, [5], 'a'
        yield verify_same, obj, ref, None, 3, [5, 9, 2, 9, 2]
        yield verify_same, obj, ref, None, 3, [5, 9, 2, 9, 2]
        yield verify_pickle, obj, ref, 2, 1, 3, [5, 9, 2, 9, 2, 4, 3]


def test_equizip():
    yield verify_same, equizip, zip, None, [3, 4], [9, 2], [9, 9]
    yield verify_same, equizip, zip, None, [3, 4, 8, 4, 2]
    assert_raises(IterableLengthMismatch, list, equizip([5, 4, 3], [2, 1]))
    assert_raises(IterableLengthMismatch, list, equizip([5, 4, 3], []))


def test_roundrobin():
    assert list(roundrobin('ABC', 'D', 'EF')) == list('ADEBFC')
    assert (list(roundrobin('ABCDEF', 'JK', 'GHI', 'L')) ==
            list('AJGLBKHCIDEF'))


def test_interleave():
    assert list(interleave(['ABC', 'D', 'EF'])) == list('ADEBFC')
    assert (list(interleave(['ABCDEF', 'JK', 'GHI', 'L'])) ==
            list('AJGLBKHCIDEF'))

    class StupidException(Exception):
        pass

    def stupid_gen():
        yield 'A'
        yield 'B'
        raise StupidException

    assert (list(interleave(['ABCDEF', stupid_gen()], [StupidException])) ==
            list('AABBCDEF'))
