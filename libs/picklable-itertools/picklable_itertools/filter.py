from abc import ABCMeta, abstractmethod
import six
from .iter_dispatch import iter_
from .base import BaseItertool


@six.add_metaclass(ABCMeta)
class BaseFilter(BaseItertool):
    def __init__(self, pred, seq):
        self._predicate = pred
        self._iter = iter_(seq)

    @abstractmethod
    def __next__(self):
        pass


class ifilter(BaseFilter):
    """ifilter(function or None, iterable) --> ifilter object

    Return an iterator yielding those items of iterable for which
    function(item) is true. If function is None, return the items that are
    true.
    """

    def _keep(self, value):
        predicate = bool if self._predicate is None else self._predicate
        return predicate(value)

    def __next__(self):
        val = next(self._iter)
        while not self._keep(val):
            val = next(self._iter)
        return val


class ifilterfalse(ifilter):
    """ifilterfalse(function or None, sequence) --> ifilterfalse object

    Return those items of sequence for which function(item) is false.
    If function is None, return the items that are false.
    """
    def _keep(self, value):
        return not super(ifilterfalse, self)._keep(value)


class takewhile(BaseFilter):
    """takewhile(predicate, iterable) --> takewhile object

    Return successive entries from an iterable as long as the
    predicate evaluates to true for each entry.
    """
    def __next__(self):
        value = next(self._iter)
        if not self._predicate(value):
            raise StopIteration
        return value


class dropwhile(takewhile):
    """dropwhile(predicate, iterable) --> dropwhile object

    Drop items from the iterable while predicate(item) is true.
    Afterwards, return every element until the iterable is exhausted.
    """
    def __next__(self):
        value = next(self._iter)
        while not getattr(self, '_started', False) and self._predicate(value):
            value = next(self._iter)
        self._started = True
        return value
