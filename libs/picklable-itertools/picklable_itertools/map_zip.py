from .base import BaseItertool
from .iter_dispatch import iter_


class imap(BaseItertool):
    """imap(func, *iterables) --> imap object

    Make an iterator that computes the function using arguments from
    each of the iterables.  Stops when the shortest iterable is exhausted.
    """
    def __init__(self, function, *iterables):
        self._function = function
        self._iterables = tuple(iter_(it) for it in iterables)

    def _run(self, args):
        return self._function(*args)

    def __next__(self):
        args = tuple([next(it) for it in self._iterables])
        if self._function is None:
            return args
        else:
            return self._run(args)


class starmap(imap):
    """starmap(function, sequence) --> starmap object

    Return an iterator whose values are returned from the function evaluated
    with a argument tuple taken from the given sequence.
    """
    def __init__(self, function, iterable):
        self._iterables = (iter_(iterable),)
        self._function = function

    def _run(self, args):
        return self._function(*args[0])


def izip(*iterables):
    """zip(iter1 [,iter2 [...]]) --> zip object

    Return a zip object whose .__next__() method returns a tuple where
    the i-th element comes from the i-th iterable argument.  The .__next__()
    method continues until the shortest iterable in the argument sequence
    is exhausted and then it raises StopIteration.
    """
    return imap(None, *iterables)


class izip_longest(BaseItertool):
    """zip_longest(iter1 [,iter2 [...]], [fillvalue=None]) --> zip_longest
    object

    Return an zip_longest object whose .__next__() method returns a tuple where
    the i-th element comes from the i-th iterable argument.  The .__next__()
    method continues until the longest iterable in the argument sequence
    is exhausted and then it raises StopIteration.  When the shorter iterables
    are exhausted, the fillvalue is substituted in their place.  The fillvalue
    defaults to None or can be specified by a keyword argument.
    """
    def __init__(self, *iterables, **kwargs):
        if 'fillvalue' in kwargs:
            self._fillvalue = kwargs['fillvalue']
            del kwargs['fillvalue']
        else:
            self._fillvalue = None
        if len(kwargs) > 0:
            raise ValueError("Unrecognized keyword arguments: {}".format(
                ", ".join(kwargs)))

        self._iterables = tuple(iter_(it) for it in iterables)

    def __next__(self):
        found_any = False
        result = []
        for it in self._iterables:
            try:
                result.append(next(it))
                found_any = True
            except StopIteration:
                result.append(self._fillvalue)
        if found_any:
            return tuple(result)
        else:
            raise StopIteration
