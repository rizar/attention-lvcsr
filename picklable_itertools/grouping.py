from .base import BaseItertool
from .iter_dispatch import iter_


class _grouper(BaseItertool):
    def __init__(self, value, iterator, groupby_obj):
        self._value = value
        self._groupby = groupby_obj
        self._key = self._groupby.key(self._value)
        self._initialized = False
        self._iterator = iterator
        self.stream_ended = False
        self._done = False

    def __next__(self):
        if not self._initialized:
            self._initialized = True
            return self._value
        else:
            if self._done:
                raise StopIteration
            try:
                value = next(self._iterator)
            except StopIteration:
                self.stream_ended = True
                self._done = True
                raise
            if self._groupby.key(value) != self._key:
                self.terminal_value = value
                self._done = True
                raise StopIteration
            return value


class groupby(BaseItertool):
    """groupby(iterable[, keyfunc]) -> create an iterator which returns
    (key, sub-iterator) grouped by each value of key(value).
    """
    def __init__(self, iterable, key=None):
        self._keyfunc = key
        self._iterator = iter_(iterable)
        self._current_key = self._initial_key = object()

    def key(self, value):
        if self._keyfunc is None:
            return value
        else:
            return self._keyfunc(value)

    def __next__(self):
        if not hasattr(self, '_current_grouper'):
            value = next(self._iterator)
            self._current_grouper = _grouper(value, self._iterator, self)
            return self.key(value), self._current_grouper
        else:
            while True:
                try:
                    next(self._current_grouper)
                except StopIteration:
                    break
            if self._current_grouper.stream_ended:
                raise StopIteration
            value = self._current_grouper.terminal_value
            self._current_grouper = _grouper(value, self._iterator, self)
            return self.key(value), self._current_grouper
