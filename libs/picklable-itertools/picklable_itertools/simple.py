import collections
from .base import BaseItertool
from .iter_dispatch import iter_


class repeat(BaseItertool):
    """
    repeat(object [,times]) -> create an iterator which returns the object
    for the specified number of times.  If not specified, returns the object
    endlessly.
    """
    def __init__(self, obj, times=None):
        self._obj = obj
        self._times = times
        self._times_called = 0

    def __next__(self):
        if self._times is None:
            return self._obj
        else:
            if self._times > self._times_called:
                self._times_called += 1
                return self._obj
            else:
                raise StopIteration


class chain(BaseItertool):
    """
    chain(*iterables) --> chain object

    Return a chain object whose .__next__() method returns elements from the
    first iterable until it is exhausted, then elements from the next
    iterable, until all of the iterables are exhausted.
    """
    def __init__(self, *iterables):
        self._iterables = iter_(iterables)
        self._current = repeat(None, 0)

    def __next__(self):
        try:
            return next(self._current)
        except StopIteration:
            self._current = iter_(next(self._iterables))
        return next(self)

    @classmethod
    def from_iterable(cls, iterable):
        obj = cls()
        obj._iterables = iter_(iterable)
        return obj


class compress(BaseItertool):
    """compress(data, selectors) --> iterator over selected data

    Return data elements corresponding to true selector elements.
    Forms a shorter iterator from selected data elements using the
    selectors to choose the data elements.
    """
    def __init__(self, data, selectors):
        self._data = iter_(data)
        self._selectors = iter_(selectors)

    def __next__(self):
        # We terminate on the shortest input sequence, so leave
        # StopIteration uncaught here.
        data = next(self._data)
        selector = next(self._selectors)
        while not bool(selector):
            data = next(self._data)
            selector = next(self._selectors)
        return data


class count(BaseItertool):
    """count(start=0, step=1) --> count object

    Return a count object whose .__next__() method returns consecutive values.
    """
    def __init__(self, start=0, step=1):
        self._n = start
        self._step = step

    def __next__(self):
        n = self._n
        self._n += self._step
        return n


class cycle(BaseItertool):
    """cycle(iterable) --> cycle object

    Return elements from the iterable until it is exhausted.
    Then repeat the sequence indefinitely.
    """
    def __init__(self, iterable):
        self._iterable = iter_(iterable)
        self._exhausted = False
        self._elements = collections.deque()

    def __next__(self):
        if not self._exhausted:
            try:
                value = next(self._iterable)
            except StopIteration:
                self._exhausted = True
                return next(self)
            self._elements.append(value)
        else:
            if len(self._elements) == 0:
                raise StopIteration
            value = self._elements.popleft()
            self._elements.append(value)
        return value


class accumulate(BaseItertool):
    """accumulate(iterable[, func]) --> accumulate object

    Return series of accumulated sums (or other binary function results).
    """
    def __init__(self, iterable, func=None):
        self._iter = iter_(iterable)
        self._func = func
        self._initialized = False
        self._accumulated = None

    def _combine(self, value):
        if self._func is not None:
            return self._func(self._accumulated, value)
        else:
            return self._accumulated + value

    def __next__(self):
        value = next(self._iter)
        if not self._initialized:
            self._accumulated = value
            self._initialized = True
        else:
            self._accumulated = self._combine(value)
        return self._accumulated
