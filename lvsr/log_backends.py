'''
Created on Jan 11, 2016

@author: jch
'''

import numpy
import pandas

from collections import Mapping, OrderedDict
from blocks.log.log import TrainingLogBase


class _TimeSlice(Mapping):
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
        self._current_time = 0
        self._current_dict = {}
        TrainingLogBase.__init__(self)

    def __getitem__(self, time):
        self._check_time(time)
        if time == self._current_time:
            return self._current_dict
        elif time > self._current_time:
            # Append the last value to column arrays
            for k, v in self._current_dict.iteritems():
                if k in self._columns:
                    col = self._columns[k]
                    if col.dtype[1] != self.get_dtype(v):
                        new_dtype = [
                            ('idx', col.dtype[0]),
                            ('val', numpy.promote_types(col.dtype[1],
                                                        self.get_dtype(v)))
                            ]
                        self._columns[k] = col.astype(new_dtype, copy=False)
                        col = self._columns[k]
                    idx = self._col_tops[k]
                    self._col_tops[k] = idx + 1
                    if idx >= col.shape[0]:
                        col2 = numpy.empty((1.3 * idx), col.dtype)
                        col2[:idx] = col
                        col2[idx:]['idx'] = 2147483647
                        col = col2
                        self._columns[k] = col2
                    col[idx] = (self._current_time, v)
                else:
                    self._columns[k] = numpy.empty(
                        (10,),
                        dtype=[('idx', numpy.int32),
                               ('val', self.get_dtype(v))])
                    self._columns[k]['idx'][:] = 2147483647
                    self._columns[k][0] = (self._current_time, v)
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
