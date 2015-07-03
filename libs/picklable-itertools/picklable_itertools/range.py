import six.moves
from numbers import Integral
from .iter_dispatch import range_iterator

__all__ = ['xrange']


def _check_integral(value):
    if not isinstance(value, Integral):
        raise TypeError("'{}' object cannot be interpreted "
                        "as an integer".format(type(value).__name__))


class xrange(object):
    """A replacement for Python 3 `range()` (and Python 2 `xrange()`) that
    yields picklable iterators when iterated upon.
    """
    __slots__ = ['_start', '_stop', '_step']

    def __init__(self, *args):
        self._start = 0
        self._step = 1
        if len(args) == 0:
            raise TypeError("{} expected 1 arguments, got 0".format(
                self.__class__.__name__))
        elif len(args) == 1:
            self._stop = args[0]
            self._start = 0
        elif len(args) >= 2:
            self._start = args[0]
            self._stop = args[1]
        if len(args) == 3:
            self._step = args[2]
        if len(args) > 3:
            raise TypeError("{} expected at most 3 arguments, got {}".format(
                self.__class__.__name__, len(args)))
        _check_integral(self._start)
        _check_integral(self._stop)
        _check_integral(self._step)

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def step(self):
        return self._step

    def count(self, i):
        """rangeobject.count(value) -> integer -- return number of occurrences
        of value
        """
        if self._stop > self._start and self._step > 0:
            return int(self._start <= i < self._stop and
                       (i - self._start) % self._step == 0)
        elif self._stop < self._start and self._step < 0:
            return int(self._start >= i > self._stop and
                       (i - self._start) % self._step == 0)
        else:
            return False

    def index(self, i):
        """xrangeobject.index(value, [start, [stop]]) -> integer --
        return index of value.
        Raise ValueError if the value is not present.
        """
        if self.count(i) == 0:
            raise ValueError("{} is not in range".format(i))
        return (i - self._start) // self._step

    def __len__(self):
        return len(six.moves.xrange(self._start, self._stop, self._step))

    def __reduce__(self):
        return (self.__class__, (self.start, self.stop, self.step))

    def __iter__(self):
        return range_iterator(self)

    def __repr__(self):
        return (__name__.split('.')[0] + '.' + self.__class__.__name__ +
                (str((self.start, self.stop)) if self.step == 1 else
                 str((self.start, self.stop, self.step))))
