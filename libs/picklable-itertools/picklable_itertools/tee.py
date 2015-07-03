"""Support code for implementing `tee`."""
import collections
import six
from picklable_itertools import iter_


class tee_iterator(six.Iterator):
    """An iterator that works in conjunction with a `tee_manager`."""
    def __init__(self, deque, manager):
        self._deque = deque
        self._manager = manager

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._deque) > 0:
            return self._deque.popleft()
        else:
            self._manager.advance()
            assert len(self._deque) > 0
            return next(self)


class tee_manager(object):
    """An object that manages a base iterator and publishes results to
    one or more client `tee_iterators`.
    """
    def __init__(self, iterable, n=2):
        self._iterable = iter_(iterable)
        self._deques = tuple(collections.deque() for _ in range(n))

    def iterators(self):
        return tuple(tee_iterator(deque, self) for deque in self._deques)

    def advance(self):
        """Advance the base iterator, publish to constituent iterators."""
        elem = next(self._iterable)
        for deque in self._deques:
            deque.append(elem)


def tee(iterable, n=2):
    """tee(iterable, n=2) --> tuple of n independent iterators."""
    return tee_manager(iter_(iterable), n=n).iterators()
