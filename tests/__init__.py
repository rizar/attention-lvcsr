from functools import partial
import itertools
from operator import add, sub, pos, gt, lt, le
import random
import tempfile

import six
from six.moves import cPickle
from six.moves import xrange
from nose.tools import assert_raises, assert_equal
from unittest import SkipTest

from picklable_itertools import (
    repeat, chain, compress, count, cycle, ifilter, ifilterfalse, imap, izip,
    file_iterator, ordered_sequence_iterator, izip_longest, iter_, islice,
    range_iterator, product, tee, accumulate, takewhile, dropwhile, starmap,
    groupby, permutations, combinations, combinations_with_replacement,
    xrange as _xrange
)
from picklable_itertools.iter_dispatch import numpy, NUMPY_AVAILABLE

_map = map if six.PY3 else itertools.imap
_zip = zip if six.PY3 else itertools.izip
_zip_longest = itertools.zip_longest if six.PY3 else itertools.izip_longest
_filter = filter if six.PY3 else itertools.ifilter
_filterfalse = itertools.filterfalse if six.PY3 else itertools.ifilterfalse
_islice = itertools.islice


def _identity(x):
    return x


def safe_assert_equal(expected_val, actual_val):
    if NUMPY_AVAILABLE and (isinstance(expected_val, numpy.ndarray) or
                            isinstance(actual_val, numpy.ndarray)):
        assert (expected_val == actual_val).all()
    else:
        assert expected_val == actual_val


def verify_same(picklable_version, reference_version, n, *args,  **kwargs):
    """Take a reference version from itertools, verify the same operation
    in our version.
    """
    try:
        expected = reference_version(*args, **kwargs)
    except Exception as e:
        assert_raises(e.__class__, picklable_version, *args, **kwargs)
        return
    actual = picklable_version(*args, **kwargs)
    done = 0
    assert n is None or isinstance(n, int)
    while done != n:
        try:
            expected_val = next(expected)
        except StopIteration:
            check_stops(actual)
            break
        try:
            actual_val = next(actual)
        except StopIteration:
            assert False, "prematurely exhausted; expected {}".format(
                str(expected_val))
            safe_assert_equal(expected_val, actual_val)
        done += 1


def verify_pickle(picklable_version, reference_version, n, m, *args, **kwargs):
    """Take n steps, pickle at m < n, and make sure it continues the same."""
    expected = reference_version(*args, **kwargs)
    actual = picklable_version(*args, **kwargs)
    done = 0
    if not m < n:
        raise ValueError("Test only makes sense with m < n")
    while done != n:
        expected_val = next(expected)
        actual_val = next(actual)
        safe_assert_equal(expected_val, actual_val)
        if done == m:
            actual = cPickle.loads(cPickle.dumps(actual))
        done += 1


def conditional_run(condition, f, *args, **kwargs):
    if condition:
        f(*args, **kwargs)
    else:
        raise SkipTest


def check_stops(it):
    """Verify that an exhausted iterator yields StopIteration."""
    try:
        val = next(it)
    except StopIteration:
        return
    assert False, "expected exhausted iterator; got {}".format(str(val))


def test_ordered_sequence_iterator():
    yield verify_same, ordered_sequence_iterator, iter, None, []
    yield verify_same, ordered_sequence_iterator, iter, None, ()
    yield verify_same, ordered_sequence_iterator, iter, None, [5, 2]
    yield verify_same, ordered_sequence_iterator, iter, None, ("D", "X", "J")
    yield verify_pickle, ordered_sequence_iterator, iter, 4, 3, [2, 9, 3, 4]
    yield verify_pickle, ordered_sequence_iterator, iter, 3, 2, ['a', 'c', 'b']
    array = numpy.array if NUMPY_AVAILABLE else list
    numpy_pickle_test = partial(conditional_run, NUMPY_AVAILABLE,
                                verify_pickle)
    numpy_same_test = partial(conditional_run, NUMPY_AVAILABLE, verify_same)
    yield (numpy_same_test, ordered_sequence_iterator, iter, None,
           array([4, 3, 9]))
    yield (numpy_same_test, ordered_sequence_iterator, iter, None,
           array([[4, 3, 9], [2, 9, 6]]))
    yield (numpy_pickle_test, ordered_sequence_iterator, iter, 4, 3,
           array([2, 9, 3, 4]))
    yield (numpy_pickle_test, ordered_sequence_iterator, iter, 3, 2,
           array([[2, 1], [2, 9], [9, 4], [3, 9]]))
    # Make sure the range iterator is actually getting dispatched by iter_.
    yield (numpy_pickle_test, iter_, iter, 4, 3,
           array([2, 9, 3, 4]))
    yield (numpy_pickle_test, iter_, iter, 3, 2,
           array([[2, 1], [2, 9], [9, 4], [3, 9]]))


def test_dict_iterator():
    d = {'a': 'b', 1: 2}
    assert list(iter_(d)) == list(iter(d))
    assert list(iter_(d.items())) == list(iter(d.items()))
    assert list(iter_(d.keys())) == list(iter(d.keys()))
    assert list(iter_(d.values())) == list(iter(d.values()))

    yield verify_pickle, iter_, iter, 2, 1, d
    yield verify_pickle, iter_, iter, 2, 1, d.items()
    yield verify_pickle, iter_, iter, 2, 1, d.values()
    yield verify_pickle, iter_, iter, 2, 1, d.keys()


def test_range_iterator():
    yield verify_same, range_iterator, iter, None, xrange(5)
    yield verify_same, range_iterator, iter, None, xrange(2, 5)
    yield verify_same, range_iterator, iter, None, xrange(5, 2)
    yield verify_same, range_iterator, iter, None, xrange(0)
    yield verify_same, range_iterator, iter, None, xrange(-3)
    yield verify_same, range_iterator, iter, None, xrange(-5, -3)
    yield verify_same, range_iterator, iter, None, xrange(-5, -1, -2)
    yield verify_same, range_iterator, iter, None, xrange(-5, -1, 2)
    yield verify_same, range_iterator, iter, None, xrange(2, 5, 7)
    yield verify_same, range_iterator, iter, None, xrange(2, 7, -1)
    yield verify_same, range_iterator, iter, None, xrange(5, 3, -1)
    yield verify_same, range_iterator, iter, None, xrange(5, 4, -2)
    yield verify_pickle, range_iterator, iter, 5, 2, xrange(5)
    yield verify_pickle, range_iterator, iter, 3, 1, xrange(2, 5)
    yield verify_pickle, range_iterator, iter, 2, 1, xrange(-5, -1, 2)
    yield verify_pickle, range_iterator, iter, 2, 1, xrange(5, 3, -1)


def _create_test_file():
    f = tempfile.NamedTemporaryFile(mode='w')
    f.write("\n".join(map(str, range(4))))
    f.flush()
    return f


def test_file_iterator():
    f = _create_test_file()
    assert list(file_iterator(open(f.name))) == list(iter(open(f.name)))


def test_file_iterator_pickling():
    f = _create_test_file()
    it = iter_(open(f.name))
    last = [next(it) for _ in range(2)][-1]
    first = next(cPickle.loads(cPickle.dumps(it)))
    assert int(first) == int(last) + 1


def test_repeat():
    yield verify_same, repeat, itertools.repeat, None, 5, 0
    yield verify_same, repeat, itertools.repeat, None, 'abc', 5
    yield verify_pickle, repeat, itertools.repeat, 5, 0, 'abc', 5
    yield verify_pickle, repeat, itertools.repeat, 5, 3, 'abc', 5
    yield verify_same, repeat, itertools.repeat, None, 'def', 3
    yield verify_pickle, repeat, itertools.repeat, 3, 0, 'def', 3
    yield verify_pickle, repeat, itertools.repeat, 3, 1, 'def', 3


def test_chain():
    yield verify_same, chain, itertools.chain, None, [5, 4], [3], [9, 10]
    yield verify_pickle, chain, itertools.chain, 5, 2, [5, 4], [3], [9, 10]
    yield verify_pickle, chain, itertools.chain, 5, 4, [5, 4], [3], [9, 10]
    yield verify_pickle, chain, itertools.chain, 5, 0, [5, 4], [3], [9, 10]
    yield verify_same, chain, itertools.chain, None, [3, 1], [], ['x', 'y']
    yield verify_pickle, chain, itertools.chain, 3, 1, [3, 1], [], ['x', 'y']
    yield verify_pickle, chain, itertools.chain, 3, 0, [3, 1], [], ['x', 'y']
    yield verify_same, chain, itertools.chain, None, [], [], []
    yield verify_same, chain, itertools.chain, None


def test_chain_from_iterable():
    yield (verify_same, chain.from_iterable, itertools.chain.from_iterable,
           None, [xrange(i) for i in xrange(3)])
    yield (verify_pickle, chain.from_iterable, itertools.chain.from_iterable,
           5, 2, [xrange(i) for i in xrange(4)])
    yield (verify_same, chain.from_iterable, itertools.chain.from_iterable,
           None, [[1], [], [2]])
    yield (verify_same, chain.from_iterable, itertools.chain.from_iterable,
           None, [[], [], []])
    yield (verify_same, chain.from_iterable, itertools.chain.from_iterable,
           None, [[], [None], []])


def test_compress():
    yield verify_same, compress, itertools.compress, None, [1, 2, 3], [1, 2, 3]
    yield verify_same, compress, itertools.compress, None, [1, 2, 3], [1, 0, 0]
    yield verify_same, compress, itertools.compress, None, [1, 2, 3], [1, 0]
    yield verify_same, compress, itertools.compress, None, [1, 2], [1, 0, 1]
    yield verify_same, compress, itertools.compress, None, [1, 2], [0, 0]
    yield verify_same, compress, itertools.compress, None, [1, 2], [0]
    yield verify_same, compress, itertools.compress, None, [1, 2], [0, 0, 0]
    yield (verify_pickle, compress, itertools.compress, 3, 1, [1, 2, 3],
           [1, 2, 3])
    yield (verify_pickle, compress, itertools.compress, 3, 0, [1, 2, 3],
           [1, 2, 3])
    yield (verify_pickle, compress, itertools.compress, 1, 0, [1, 2, 3],
           [1, 0, 0])
    yield (verify_pickle, compress, itertools.compress, 1, 0, [1, 2, 3],
           [1, 0])
    yield (verify_pickle, compress, itertools.compress, 1, 0, [1, 2],
           [1, 0, 1])


def test_count():
    yield verify_same, count, itertools.count, 6
    yield verify_same, count, itertools.count, 20, 2
    yield verify_same, count, itertools.count, 10, 5, 9
    yield verify_same, count, itertools.count, 30, 3, 10
    yield verify_pickle, count, itertools.count, 6, 1
    yield verify_pickle, count, itertools.count, 20, 5, 2
    yield verify_pickle, count, itertools.count, 20, 0, 2
    yield verify_pickle, count, itertools.count, 10, 9, 5, 9
    yield verify_pickle, count, itertools.count, 10, 6, 5, 9
    yield verify_pickle, count, itertools.count, 30, 7, 3, 10


def test_cycle():
    yield verify_same, cycle, itertools.cycle, 40, [4, 9, 10]
    yield verify_same, cycle, itertools.cycle, 10, [4, 9, 20, 10]
    yield verify_same, cycle, itertools.cycle, 20, [4, 9, 30, 10, 9]
    yield verify_same, cycle, itertools.cycle, 60, [8, 4, 5, 4, 9, 10]
    yield verify_pickle, cycle, itertools.cycle, 40, 20, [4, 9, 10]
    yield verify_pickle, cycle, itertools.cycle, 10, 9, [4, 9, 20, 10]
    yield verify_pickle, cycle, itertools.cycle, 20, 1, [4, 9, 30, 10, 9]
    yield verify_pickle, cycle, itertools.cycle, 60, 55, [8, 4, 5, 4, 9, 10]
    yield verify_pickle, cycle, itertools.cycle, 60, 0, [8, 4, 5, 4, 9, 10]
    yield verify_same, cycle, itertools.cycle, None, []


def test_imap():
    yield verify_same, imap, _map, None, partial(add, 2), [3, 4, 5]
    yield verify_same, imap, _map, None, add, [3, 4], [9, 2]
    yield verify_same, imap, _map, None, add, [3], [9, 2]
    yield verify_same, imap, _map, None, add, [3], [9, 2], []
    yield verify_pickle, imap, _map, 3, 1, partial(add, 2), [3, 4, 5]
    yield verify_pickle, imap, _map, 3, 0, partial(add, 2), [3, 4, 5]
    yield verify_pickle, imap, _map, 2, 1, add, [3, 4], [9, 2]
    yield verify_pickle, imap, _map, 2, 0, add, [3, 4], [9, 2]
    yield verify_pickle, imap, _map, 1, 0, add, [3], [9, 2]


def test_izip():
    yield verify_same, izip, _zip, None, [3, 4, 5]
    yield verify_same, izip, _zip, None, [3, 4], [9, 2]
    yield verify_same, izip, _zip, None, [3], [9, 2]
    yield verify_same, izip, _zip, None, [3], [9, 2], []
    yield verify_pickle, izip, _zip, 3, 2, [3, 4, 5]
    yield verify_pickle, izip, _zip, 3, 1, [3, 4, 5]
    yield verify_pickle, izip, _zip, 2, 1, [3, 4], [9, 2]
    yield verify_pickle, izip, _zip, 2, 0, [3, 4], [9, 2]
    yield verify_pickle, izip, _zip, 1, 0, [3], [9, 2]


def test_ifilter():
    yield verify_same, ifilter, _filter, None, partial(le, 4), [3, 4, 5]
    yield verify_same, ifilter, _filter, None, partial(gt, 3), []
    yield verify_same, ifilter, _filter, None, partial(le, 6), [3, 4, 5]
    yield verify_same, ifilter, _filter, None, None, [0, 3, 0, 0, 1]
    yield verify_pickle, ifilter, _filter, 2, 0, partial(le, 4), [3, 4, 5]
    yield verify_pickle, ifilter, _filter, 2, 1, partial(le, 4), [3, 4, 5]
    yield verify_pickle, ifilter, _filter, 2, 1, None, [0, 3, 0, 0, 1]
    yield verify_pickle, ifilter, _filter, 2, 0, None, [0, 3, 0, 0, 1]


def test_ifilterfalse():
    yield (verify_same, ifilterfalse, _filterfalse, None,
           partial(le, 4), [3, 4, 5])
    yield (verify_same, ifilterfalse, _filterfalse, None,
           partial(le, 6), [3, 4, 5])
    yield (verify_same, ifilterfalse, _filterfalse, None,
           partial(gt, 3), [])
    yield (verify_same, ifilterfalse, _filterfalse, None,
           None, [0, 3, 0, 0, 1])
    yield (verify_pickle, ifilterfalse, _filterfalse, 1, 0,
           partial(le, 4), [3, 4, 5])
    yield (verify_pickle, ifilterfalse, _filterfalse, 3, 2,
           partial(le, 6), [3, 4, 5])
    yield (verify_pickle, ifilterfalse, _filterfalse, 3, 0,
           partial(le, 6), [3, 4, 5])
    yield (verify_pickle, ifilterfalse, _filterfalse, 3, 0,
           None, [0, 3, 0, 0, 1])
    yield (verify_pickle, ifilterfalse, _filterfalse, 3, 2,
           None, [0, 3, 0, 0, 1])


def test_product():
    yield verify_same, product, itertools.product, None
    yield verify_same, product, itertools.product, None, []
    yield verify_same, product, itertools.product, None, [], []
    yield verify_same, product, itertools.product, None, [], [], []
    yield verify_same, product, itertools.product, None, [5]
    yield verify_same, product, itertools.product, None, [5], []
    yield verify_same, product, itertools.product, None, [], [5], []
    yield verify_same, product, itertools.product, None, [2, 5], [3, 5, 9]
    yield verify_same, product, itertools.product, None, [2, 5], [1], [3, 5, 9]
    yield (verify_same, partial(product, repeat=3),
           partial(itertools.product, repeat=3), None, [1, 2, 3])
    yield (verify_same, partial(product, repeat=4),
           partial(itertools.product, repeat=4), None, [1], [1, 2])
    yield (verify_same, partial(product, repeat=2),
           partial(itertools.product, repeat=2), None, [3, 1], [1])
    yield (verify_same, partial(product, repeat=3),
           partial(itertools.product, repeat=3), None, [])
    yield (verify_same, partial(product, repeat=3),
           partial(itertools.product, repeat=3), None, [], [3])
    yield (verify_same, partial(product, repeat=3),
           partial(itertools.product, repeat=3), None, [1], [])
    yield (verify_pickle, product, itertools.product, 8, 3, [1, 2], [2, 3],
           [5, 6])
    yield (verify_pickle, partial(product, repeat=3),
           partial(itertools.product, repeat=3), 50, 45,
           [1, 2], [3, 4])


def test_izip_longest():
    yield (verify_same, izip_longest, _zip_longest, None, [], [])
    yield (verify_same, izip_longest, _zip_longest, None, [], [5, 4])
    yield (verify_same, izip_longest, _zip_longest, None, [2], [5, 4])
    yield (verify_same, izip_longest, _zip_longest, None, [7, 9], [5, 4])
    yield (verify_same, izip_longest, _zip_longest, None, [7, 9],
           [4], [2, 9, 3])
    yield (verify_same, izip_longest, _zip_longest, None, [7, 9], [4], [])
    yield (verify_same, izip_longest, _zip_longest, None, [7], [4], [],
           [5, 9])
    yield (verify_same, partial(izip_longest, fillvalue=-1),
           partial(_zip_longest, fillvalue=-1), None,
           [7], [4], [], [5, 9])

    yield (verify_pickle, izip_longest, _zip_longest, 3, 2, [7, 9, 8], [1, 2])
    yield (verify_pickle, izip_longest, _zip_longest, 3, 1, [7, 9, 8], [1, 2])


def test_islice():
    yield (verify_same, islice, _islice, None, [], 0)
    yield (verify_same, islice, _islice, None, [1], 0)
    yield (verify_same, islice, _islice, None, [1], 1)
    yield (verify_same, islice, _islice, None, [1], 3)
    yield (verify_same, islice, _islice, None, [1, 2, 3], 5)
    yield (verify_same, islice, _islice, None, [1, 2, 3], 1, 2)
    yield (verify_same, islice, _islice, None, [1, 2, 3], 1, 5, 3)
    yield (verify_same, islice, _islice, None, [1, 2, 3], 0, 3, 2)
    yield (verify_same, islice, _islice, None, [1, 2, 3, 4, 5], 1, 4, 3)
    yield (verify_same, islice, _islice, None, [1, 2, 3, 4, 5], -2, 9, 4)
    yield (verify_same, islice, _islice, None, [1, 2, 3], 4, 9)
    yield (verify_same, islice, _islice, None, [1, 2, 3], 0, 9, 5)

    yield (verify_pickle, islice, _islice, 3, 2, [1, 2, 3], 5)


def verify_tee(n, original, seed):
    try:
        state = random.getstate()
        iterators = list(tee(original, n=n))
        results = [[] for _ in range(n)]
        exhausted = [False] * n
        while not all(exhausted):
            # Upper argument of random.randint is inclusive. Argh.
            i = random.randint(0, n - 1)
            if not exhausted[i]:
                if len(results[i]) == len(original):
                    assert_raises(StopIteration, next, iterators[i])
                    assert results[i] == original
                    exhausted[i] = True
                else:
                    if random.randint(0, 1):
                        iterators[i] = cPickle.loads(
                            cPickle.dumps(iterators[i]))
                    elem = next(iterators[i])
                    results[i].append(elem)
    finally:
        random.setstate(state)


def test_tee():
    yield verify_tee, 2, [5, 2, 4], 1
    yield verify_tee, 3, [5, 2, 4, 6, 9], 2
    yield verify_tee, 5, [5, 2, 4, 6, 9], 3
    yield verify_tee, 6, [], 3


def test_accumulate():
    if not six.PY3:
        raise SkipTest()
    yield verify_same, accumulate, itertools.accumulate, None, [5, 4, 9]
    yield verify_same, accumulate, itertools.accumulate, None, ['a', 'b', 'c']
    yield (verify_same, accumulate, itertools.accumulate, None,
           [[1], [2], [3, 4]])
    yield (verify_same, accumulate, itertools.accumulate, None, [9, 1, 2],
           sub)
    yield verify_pickle, accumulate, itertools.accumulate, 3, 1, [5, 4, 9]
    yield verify_pickle, accumulate, itertools.accumulate, 3, 0, [5, 4, 9]
    yield (verify_pickle, accumulate, itertools.accumulate, 3, 2,
           ['a', 'b', 'c'])
    yield (verify_pickle, accumulate, itertools.accumulate, 2, 1,
           ['a', 'b', 'c'])
    yield (verify_pickle, accumulate, itertools.accumulate, 3, 1,
           [[1], [2], [3, 4]])
    yield (verify_pickle, accumulate, itertools.accumulate, 2, 1,
           [9, 1, 2], sub)


def test_takewhile():
    base = (verify_same, takewhile, itertools.takewhile, None)
    yield base + (bool,)
    yield base + (bool, [])
    yield base + (bool, [0, 0, 5])
    yield base + (bool, [1, 2, 0, 4, 0])
    yield base + (partial(lt, 3), range(5, 0, -1))
    base = (verify_pickle, takewhile, itertools.takewhile)
    yield base + (2, 0, bool, [1, 2, 0, 4, 0])
    yield base + (2, 1, bool, [1, 2, 0, 4, 0])
    yield base + (1, 0, partial(lt, 3), range(5, 0, -1))


def test_dropwhile():
    base = (verify_same, dropwhile, itertools.dropwhile, None)
    yield base + (bool,)
    yield base + (bool, [])
    yield base + (bool, [5, 5, 2, 0, 0])
    yield base + (bool, [1, 2, 0, 4, 0])
    yield base + (partial(lt, 3), range(5, 0, -1))
    base = (verify_pickle, dropwhile, itertools.dropwhile)
    yield base + (2, 1, bool, [5, 5, 2, 0, 0])
    yield base + (2, 0, bool, [5, 5, 2, 0, 0])
    yield base + (3, 0, bool, [1, 2, 0, 4, 0])
    yield base + (3, 2, bool, [1, 2, 0, 4, 0])


def test_starmap():
    yield verify_same, starmap, itertools.starmap, None, pos
    yield verify_same, starmap, itertools.starmap, None, pos, []
    yield (verify_same, starmap, itertools.starmap, None, add,
           [(5, 9), [4, 2]])
    yield (verify_pickle, starmap, itertools.starmap, 2, 0, add,
           [(5, 9), [4, 2]])
    yield (verify_pickle, starmap, itertools.starmap, 2, 1, add,
           [(5, 9), [4, 2]])


def verify_groupby(*args, **kwargs):
    if 'n' in kwargs:
        if 'm' not in kwargs:
            raise ValueError('got n without m')
        pickle = True
        n = kwargs.pop('n')
        m = kwargs.pop('m')
    elif 'm' in kwargs:
        raise ValueError('got m without n')
    else:
        pickle = False
        n = m = None
    if 'pickle_outer' in kwargs:
        pickle_outer = kwargs.pop('pickle_outer')
    else:
        pickle_outer = None
    reference = itertools.groupby(*args, **kwargs)
    actual = groupby(*args, **kwargs)
    outer_iters = 0
    while True:
        if outer_iters == pickle_outer:
            actual = cPickle.loads(cPickle.dumps(actual))
        try:
            ref_key, ref_grouper = next(reference)
        except StopIteration:
            check_stops(actual)
            break
        try:
            actual_key, actual_grouper = next(actual)
        except StopIteration:
            assert False, "prematurely exhausted; expected {}".format(ref_key)
        if pickle:
            this_n = n[0]
            n = n[1:]
            this_m = m[0]
            m = m[1:]
            verify_pickle(partial(_identity, actual_grouper),
                          partial(_identity, ref_grouper), this_n, this_m)

        else:
            verify_same(partial(_identity, actual_grouper),
                        partial(_identity, ref_grouper), None)
        outer_iters += 1


def _mod(x, divisor):
    return x % divisor


def test_groupby():
    yield verify_groupby, []
    yield verify_groupby, [1, 1, 2, 3, 3, 3, 4, 5, 7, 7]
    yield (verify_groupby, [1, 1, 3, 3, 4, 4, 2, 3, 3, 5],
           partial(_mod, divisor=2))
    yield (partial(verify_groupby, n=[4, 3, 3], m=[0, 1, 2]),
           [1, 1, 3, 3, 4, 4, 2, 3, 3, 5],
           partial(_mod, divisor=2))
    yield (partial(verify_groupby, n=[4, 3, 3], m=[1, 0, 1]),
           [1, 1, 3, 3, 4, 4, 2, 3, 3, 5],
           partial(_mod, divisor=2))
    yield (partial(verify_groupby, n=[4, 3, 3], m=[1, 0, 1], pickle_outer=1),
           [1, 1, 3, 3, 4, 4, 2, 3, 3, 5],
           partial(_mod, divisor=2))
    yield (partial(verify_groupby, n=[4, 3, 3], m=[2, 1, 0], pickle_outer=2),
           [1, 1, 3, 3, 4, 4, 2, 3, 3, 5],
           partial(_mod, divisor=2))
    yield (partial(verify_groupby, n=[4, 3, 3], m=[1, 2, 0], pickle_outer=0),
           [1, 1, 3, 3, 4, 4, 2, 3, 3, 5],
           partial(_mod, divisor=2))


def test_permutations():
    yield verify_same, permutations, itertools.permutations, None, _identity,
    yield (verify_same, permutations, itertools.permutations, None,
           [])
    yield (verify_same, permutations, itertools.permutations, None,
           [5, 4, 3, 2, 1])
    yield (verify_same, permutations, itertools.permutations, None,
           [5, 4, 3, 2, 1], 2)
    yield (verify_pickle, permutations, itertools.permutations, 5 * 4 * 3 * 2,
           0, [5, 4, 3, 2, 1])
    yield (verify_pickle, permutations, itertools.permutations, 5 * 4 * 3 * 2,
           5 * 4 * 3, [5, 4, 3, 2, 1])
    yield (verify_pickle, permutations, itertools.permutations, 5 * 4,
           10, [5, 4, 3, 2, 1], 2)
    yield (verify_pickle, permutations, itertools.permutations, 5 * 4,
           5 * 4 - 1, [5, 4, 3, 2, 1], 2)


def test_combinations():
    yield verify_same, combinations, itertools.combinations, None
    yield (verify_same, combinations, itertools.combinations, None, [])
    yield (verify_same, combinations, itertools.combinations, 5 * 4 * 3, 0,
           [5, 4, 3, 2, 1], 2)
    yield (verify_pickle, combinations, itertools.combinations, 10,
           5, [5, 4, 3, 2, 1], 2)


def test_combinations_with_replacement():
    yield (verify_same, combinations_with_replacement,
           itertools.combinations_with_replacement,
           None, _identity)
    yield (verify_same, combinations_with_replacement,
           itertools.combinations_with_replacement,
           None, _identity, [])
    yield (verify_same, combinations_with_replacement,
           itertools.combinations_with_replacement,
           None)
    yield (verify_same, combinations_with_replacement,
           itertools.combinations_with_replacement,
           None, [5, 4, 3, 2, 1], 2)
    yield (verify_pickle, combinations_with_replacement,
           itertools.combinations_with_replacement,
           15, 3, [5, 4, 3, 2, 1], 2)
    yield (verify_pickle, combinations_with_replacement,
           itertools.combinations_with_replacement,
           15, 0, [5, 4, 3, 2, 1], 2)


def test_xrange():
    yield assert_equal, list(xrange(10)), list(_xrange(10))
    yield assert_equal, list(xrange(10, 15)), list(_xrange(10, 15))
    yield assert_equal, list(xrange(10, 20, 2)), list(_xrange(10, 20, 2))
    yield assert_equal, list(xrange(5, 1, -1)), list(_xrange(5, 1, -1))
    yield (assert_equal, list(xrange(5, 55, 3)),
           list(cPickle.loads(cPickle.dumps(_xrange(5, 55, 3)))))
    yield assert_equal, _xrange(5).index(4), 4
    yield assert_equal, _xrange(5, 9).index(6), 1
    yield assert_equal, _xrange(8, 24, 3).index(11), 1
    yield assert_equal, _xrange(25, 4, -5).index(25), 0
    yield assert_equal, _xrange(28, 7, -7).index(14), 2
    yield assert_raises, ValueError, _xrange(2, 9, 2).index, 3
    yield assert_raises, ValueError, _xrange(2, 20, 2).index, 9
    yield assert_equal, _xrange(5).count(5), 0
    yield assert_equal, _xrange(5).count(4), 1
    yield assert_equal, _xrange(4, 9).count(4), 1
    yield assert_equal, _xrange(3, 9, 2).count(4), 0
    yield assert_equal, _xrange(3, 9, 2).count(5), 1
    yield assert_equal, _xrange(3, 9, 2).count(20), 0
    yield assert_equal, _xrange(9, 3).count(5), 0
    yield assert_equal, _xrange(3, 10, -1).count(5), 0
    yield assert_equal, _xrange(10, 3, -1).count(5), 1
    yield assert_equal, _xrange(10, 0, -2).count(6), 1
    yield assert_equal, _xrange(10, -1, -3).count(7), 1
