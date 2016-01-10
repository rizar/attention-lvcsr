"""The event-based main loop of Blocks."""
from abc import ABCMeta
from collections import defaultdict, OrderedDict
from numbers import Integral
from uuid import uuid4

import six
import array
import collections
from bisect import bisect_left
import numpy
import pandas


@six.add_metaclass(ABCMeta)
class TrainingLogBase(object):
    """Base class for training log.

    A training log stores the training timeline, statistics and other
    auxiliary information. Training logs can use different backends e.g.
    in-memory Python objects or an SQLite database.

    Information is stored similar to a nested dictionary, so use
    ``log[time][key]`` to read data. An entry without stored data will
    return an empty dictionary-like object that can be written to,
    ``log[time][key] = value``.

    Depending on the backend, ``log[time] = {'key': 'value'}`` could fail.
    Use ``log[time].update({'key': 'value'})`` for compatibility across
    backends.

    In addition to the set of records displaying training dynamics, a
    training log has a :attr:`status` attribute, which is a dictionary with
    data that is not bound to a particular time.

    .. warning::

       Changes to mutable objects might not be reflected in the log,
       depending on the backend. So don't use
       ``log.status['key'].append(...)``, use ``log.status['key'] = ...``
       instead.

    Parameters
    ----------
    uuid : :class:`uuid.UUID`, optional
        The UUID of this log. For persistent log backends, passing the UUID
        will result in an old log being loaded. Otherwise a new, random
        UUID will be created.

    Attributes
    ----------
    status : dict
        A dictionary with data representing the current state of training.
        By default it contains ``iterations_done``, ``epochs_done`` and
        ``_epoch_ends`` (a list of time stamps when epochs ended).

    """
    def __init__(self, uuid=None):
        if uuid is None:
            self.uuid = uuid4()
        else:
            self.uuid = uuid
        if uuid is None:
            self.status.update({
                'iterations_done': 0,
                'epochs_done': 0,
                '_epoch_ends': [],
                'resumed_from': None
            })

    @property
    def h_uuid(self):
        """Return a hexadecimal version of the UUID bytes.

        This is necessary to store ids in an SQLite database.

        """
        return self.uuid.hex

    def resume(self):
        """Resume a log by setting a new random UUID.

        Keeps a record of the old log that this is a continuation of. It
        copies the status of the old log into the new log.

        """
        old_uuid = self.h_uuid
        old_status = dict(self.status)
        self.uuid = uuid4()
        self.status.update(old_status)
        self.status['resumed_from'] = old_uuid

    def _check_time(self, time):
        if not isinstance(time, Integral) or time < 0:
            raise ValueError("time must be a non-negative integer")

    @property
    def current_row(self):
        return self[self.status['iterations_done']]

    @property
    def previous_row(self):
        return self[self.status['iterations_done'] - 1]

    @property
    def last_epoch_row(self):
        return self[self.status['_epoch_ends'][-1]]


class TrainingLog(defaultdict, TrainingLogBase):
    """Training log using a `defaultdict` as backend.

    Notes
    -----
    For analysis of the logs, it can be useful to convert the log to a
    Pandas_ data frame:

    .. code:: python

       df = DataFrame.from_dict(log, orient='index')

    .. _Pandas: http://pandas.pydata.org

    """
    def __init__(self):
        defaultdict.__init__(self, dict)
        self.status = {}
        TrainingLogBase.__init__(self)

    def __reduce__(self):
        constructor, args, _, _, items = super(TrainingLog, self).__reduce__()
        return constructor, (), self.__dict__, _, items

    def __getitem__(self, time):
        self._check_time(time)
        return super(TrainingLog, self).__getitem__(time)

    def __setitem__(self, time, value):
        self._check_time(time)
        return super(TrainingLog, self).__setitem__(time, value)


class _NotFound(object):
    pass


class _KeyedDict(collections.Mapping):
    def __init__(self, log_row, list_log):
        self._row = log_row
        self._keys = list_log.keys
        assert isinstance(self._keys, OrderedDict)

    def __getitem__(self, item):
        try:
            val = self._row[self._keys[item]]
        except:
            raise KeyError
        if val == _NotFound:
            raise KeyError
        return val

    def __iter__(self):
        for k, v in zip(self._keys.iterkeys(), self._row):
            if v != _NotFound:
                yield k

    def __len__(self):
        l = 0
        for v in self._row:
            if v != _NotFound:
                l += 1
        return l


class ListLog(TrainingLogBase):
    """Better Training log

    Rows are stored as lists

    """
    def __init__(self):
        self.keys = OrderedDict()
        self.status = {}
        self.iters = array.array('i')
        self.rows = []
        TrainingLogBase.__init__(self)

    def __getitem__(self, time):
        self._check_time(time)
        idx = bisect_left(self.iters, time)
        if idx == len(self.iters):
            self.iters.append(time)
            self.rows.append({})
            idx1 = idx-1
            if idx1 >= 0 and isinstance(self.rows[idx1], dict):
                row1 = self.rows[idx1]
                row_len = 0
                keys = self.keys
                for k in row1.iterkeys():
                    row_len = max(row_len,
                                  keys.setdefault(k, len(keys)))
                row = [_NotFound] * (row_len + 1)
                for k, v in row1.iteritems():
                    row[keys[k]] = v
                self.rows[idx1] = row

        ret = self.rows[idx]

        if isinstance(ret, dict):
            return ret
        else:
            return _KeyedDict(ret, self)

    def __setitem__(self, time, value):
        self._check_time(time)
        return super(ListLog, self).__setitem__(time, value)


class _TimeSlice(collections.Mapping):
    def __init__(self, time, log):
        self._time = time
        self._columns = log._columns
        assert isinstance(self._columns, OrderedDict)

    def __getitem__(self, item):
        ndarr = self._columns[item]
        time = self._time
        idx = ndarr['idx'].searchsorted(time)
        if idx < ndarr.shape[0]:
            row = ndarr[idx]
            if row['idx'] == time:
                return row['val']
        raise KeyError

    def __iter__(self):
        time = self._time
        for k, ndarr in self._columns.iteritems():
            times = ndarr['idx']
            idx = times.searchsorted(time)
            if idx < times.shape[0] and times[idx] == time:
                yield k

    def __len__(self):
        l = 0
        time = self._time
        for ndarr in self._columns.itervalues():
            times = ndarr['idx']
            idx = times.searchsorted(time)
            if idx < times.shape[0] and times[idx] == time:
                l += 1
        return l


class NDarrayLog(TrainingLogBase):
    """Better Training log

    Columns are stored as ndarrays. Binary search is used to find
    historical times.

    """

    def get_dtype(self, obj):
        if hasattr(obj, 'dtype'):
            return (obj.dtype, obj.shape)
        DTYPES = {
            int: numpy.int,
            float: numpy.float,
            bool: numpy.bool}
        return DTYPES.get(type(obj), numpy.dtype('object'))

    def __init__(self):
        self._columns = OrderedDict()
        self._col_tops = {}
        self.status = {}
        self._current_time = -1
        self._current_dict = None
        TrainingLogBase.__init__(self)

    def __getitem__(self, time):
        self._check_time(time)
        if time == self._current_time:
            return self._current_dict
        elif time > self._current_time:
            # Append the last value to column arrays
            if self._current_time >= 0:
                for k, v in self._current_dict.iteritems():
                    if k in self._columns:
                        col = self._columns[k]
                        idx = self._col_tops[k]
                        self._col_tops[k] = idx + 1
                        if idx >= col.shape[0]:
                            col2 = numpy.empty((1.3 * idx), col.dtype)
                            col2[:idx] = col
                            col2[idx:]['idx'] = 2147483647
                            col = col2
                            self._columns[k] = col2
                        col['idx'][idx] = self._current_time
                        col['val'][idx] = v
                    else:
                        self._columns[k] = numpy.empty(
                            (10,),
                            dtype=[('idx', numpy.int32),
                                   ('val', self.get_dtype(v))])
                        self._columns[k]['idx'][:] = 2147483647
                        self._columns[k]['idx'][0] = self._current_time
                        self._columns[k]['val'][0] = v
                        self._col_tops[k] = 1
            self._current_time = time
            self._current_dict = {}
            return self._current_dict
        else:
            return _TimeSlice(time, self)

    def __setitem__(self, time, value):
        self._check_time(time)
        if time == self._current_time:
            self._current_dict = value
        else:
            raise KeyError("Can't modify log entries for the past")

    def to_pandas(self):
        """
        Return a pandas DataFrame view of the log.

        """
        # Write down the last record
        if self._current_dict:
            # Executes if self._current_dict has uncommitted chages
            unused_dict = self[self._current_time + 1]
        series = {}
        for name, col in self._columns.iteritems():
            col = col[:self._col_tops[name]]
            col.setflags(write=False)
            print name
            if col['val'].ndim == 1:
                dtype = col['val'].dtype
                data = col['val']
            else:
                dtype = 'object'
                data = list(col['val'])
            s = pandas.Series(data, index=col['idx'], dtype=dtype)
            series[name] = s
        # this makes a copy of all the data :(
        # TODO: how to directly get a sparse DataFrame??
        return pandas.DataFrame(series)
