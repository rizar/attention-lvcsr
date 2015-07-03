import sys

from .base import BaseItertool
from .iter_dispatch import iter_


class islice(BaseItertool):
    """islice(iterable, stop) --> islice object
    islice(iterable, start, stop[, step]) --> islice object

    Return an iterator whose next() method returns selected values from an
    iterable.  If start is specified, will skip all preceding elements;
    otherwise, start defaults to zero.  Step defaults to one.  If
    specified as another value, step determines how many values are
    skipped between successive calls.  Works like a slice() on a list
    but returns an iterator.
    """
    def __init__(self, iterable, start, stop=None, step=1):
        if stop is None:
            start, stop = 0, start
        if (not 0 <= start <= sys.maxsize or
                not 0 <= stop <= sys.maxsize or
                not 0 <= step <= sys.maxsize):
            raise ValueError("Indices for islice() must be None or an "
                             "integer: 0 <= x <= maxint.")

        self._iterable = iter_(iterable)
        i = 0
        while i < start:
            try:
                next(self._iterable)
                i += 1
            except StopIteration:
                break

        self._stop = stop - start
        self._step = step
        self._n = 0

    def __next__(self):
        while self._n % self._step and self._n < self._stop:
            next(self._iterable)
            self._n += 1
        if self._n == self._stop:
            raise StopIteration
        value = next(self._iterable)
        self._n += 1
        return value
