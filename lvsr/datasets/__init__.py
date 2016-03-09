import cPickle
import functools
import os

import fuel
import numpy
from fuel.schemes import (
    ConstantScheme, ShuffledExampleScheme, SequentialExampleScheme)
from fuel.streams import DataStream
from fuel.transformers import (
    SortMapping, Padding, ForceFloatX, Batch, Mapping, Unpack, Filter,
    FilterSources, Transformer, AgnosticTransformer, Rename)

from lvsr.datasets.h5py import H5PYAudioDataset
from blocks.utils import dict_subset


import logging
logger = logging.getLogger(__name__)


def switch_first_two_axes(batch):
    result = []
    for array in batch:
        if array.ndim == 2:
            result.append(array.transpose(1, 0))
        else:
            result.append(array.transpose(1, 0, 2))
    return tuple(result)


class _Length(object):
    def __init__(self, index):
        self.index = index

    def __call__(self, example):
        return len(example[self.index])


class _AddLabel(object):

    def __init__(self, label, index, append=True, times=1):
        self.label = label
        self.append = append
        self.times = times
        self.index = index

    def __call__(self, example):
        example = list(example)
        i = self.index
        if self.append:
            # Not using `list.append` to avoid having weird mutable
            # example objects.
            example[i] = numpy.hstack([example[i], self.times * [self.label]])
        else:
            example[i] = numpy.hstack([self.times * [self.label], example[1]])
        return example


class _LengthFilter(object):

    def __init__(self, index, max_length):
        self.index = index
        self.max_length = max_length

    def __call__(self, example):
        if self.max_length:
            return len(example[self.index]) <= self.max_length
        return True


class Rearrange(AgnosticTransformer):
    """Rearranges the sources of the stream.

    Parameters
    ----------
    data_stream : :class:`DataStream` or :class:`Transformer`.
        The data stream.
    new_to_old : :class:`OrderedDict`
        New sources and their corresponding old sources.

    """
    def __init__(self, data_stream, new_to_old, **kwargs):
        if data_stream.axis_labels:
            kwargs.setdefault(
                'axis_labels',
                {new: data_stream.axis_labels[old]
                for new, old in new_to_old.items()})
        super(Rearrange, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

        self.new_to_old = new_to_old
        new_sources = new_to_old.keys()
        for old in new_to_old.values():
            if old not in data_stream.sources:
                raise KeyError("%s not in the sources of the stream" % old)
        self.sources = new_sources

    def transform_any(self, data):
        return tuple(data[self.data_stream.sources.index(old)]
                     for old in self.new_to_old.values())


class ForceCContiguous(Transformer):
    def __init__(self, data_stream):
        super(ForceCContiguous, self).__init__(
            data_stream, axis_labels=data_stream.axis_labels)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        result = []
        for piece in data:
            if isinstance(piece, numpy.ndarray):
                result.append(numpy.ascontiguousarray(piece))
            else:
                result.append(piece)
        return tuple(result)


class Data(object):
    """Dataset manager.

    This class is in charge of accessing different datasets
    and building preprocessing pipelines.

    Parameters
    ----------
    dataset_filename : str
        Dataset file name.
    name_mapping : dict
        A map from conceptual split names (train, test) into concrete split
        names (e.g. 93eval).
    sources_map: dict
        A map from conceptual source names, such as "labels" or "recordings"
        into names of dataset entries.
    batch_size : int
        Batch size.
    validation_batch_size : int
        Batch size used for validation.
    sort_k_batches : int
    max_length : int
        Maximum length of input, longer sequences will be filtered.
    normalization : str
        Normalization file name to use.
    add_eos : bool
        Add end of sequence symbol.
    add_bos : int
        Add this many beginning-of-sequence tokens.
    eos_label : int
        Label to use for eos symbol.
    default_sources : list
        Default sources to include in created datasets
    dataset_class : object
        Class for this particulat dataset kind (WSJ, TIMIT)
    """
    def __init__(self, dataset_filename, name_mapping, sources_map,
                 batch_size, validation_batch_size=None,
                 sort_k_batches=None,
                 max_length=None, normalization=None,
                 add_eos=True, eos_label=None,
                 add_bos=0, prepend_eos=False,
                 default_sources=None,
                 dataset_class=H5PYAudioDataset):
        assert not prepend_eos

        if normalization:
            with open(normalization, "rb") as src:
                normalization = cPickle.load(src)

        self.dataset_filename = dataset_filename
        self.dataset_class = dataset_class
        self.name_mapping = name_mapping
        self.sources_map = sources_map
        if default_sources is None:
            logger.warn(
                "The Data class was provided with no default_sources.\n"
                "All instantiated Datasets or Datastreams will use all "
                "available sources.\n")
            default_sources = sources_map.keys()
        self.default_sources = default_sources

        self.normalization = normalization
        self.batch_size = batch_size
        if validation_batch_size is None:
            validation_batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.sort_k_batches = sort_k_batches
        self.max_length = max_length
        self.add_eos = add_eos
        self.prepend_eos = prepend_eos
        self._eos_label = eos_label
        self.add_bos = add_bos
        self.dataset_cache = {}
        #
        # Hardcode the number of source for length at 0
        # this typixcally works, as main.get_net_config
        # will properly set default_sources, such that the label is last
        # Unfortunately, we cannot query for a source name, as the
        # list of sources will differ....
        #
        self.length_filter = _LengthFilter(
            index=0,
            max_length=self.max_length)

    @property
    def info_dataset(self):
        return self.get_dataset("train")

    @property
    def num_labels(self):
        return self.info_dataset.num_characters

    @property
    def eos_label(self):
        if self._eos_label:
            return self._eos_label
        return self.info_dataset.eos_label

    @property
    def bos_label(self):
        return self.info_dataset.bos_label

    def character_map(self, source):
        return self.info_dataset.character_map(self.sources_map[source])

    def num_features(self, source):
        return self.info_dataset.dim(self.sources_map[source])

    def decode(self, labels):
        return self.info_dataset.decode(labels)

    def pretty_print(self, labels, example):
        return self.info_dataset.pretty_print(labels, example)

    def get_dataset(self, part, add_sources=()):
        """Returns dataset from the cache or creates a new one"""
        sources = []
        for src in self.default_sources + list(add_sources):
            sources.append(self.sources_map[src])
        sources = tuple(sources)
        key = (part, sources)
        if key not in self.dataset_cache:
            self.dataset_cache[key] = self.dataset_class(
                file_or_path=os.path.join(fuel.config.data_path[0],
                                          self.dataset_filename),
                which_sets=(self.name_mapping.get(part, part), ),
                sources=sources,
                target_source=self.sources_map['labels'])
        return self.dataset_cache[key]

    def get_stream(self, part, batches=True, shuffle=True, add_sources=(),
                   num_examples=None, rng=None, seed=None):
        dataset = self.get_dataset(part, add_sources=add_sources)
        if num_examples is None:
            num_examples = dataset.num_examples

        if shuffle:
            iteration_scheme = ShuffledExampleScheme(num_examples, rng=rng)
        else:
            iteration_scheme = SequentialExampleScheme(num_examples)

        stream = DataStream(
            dataset, iteration_scheme=iteration_scheme)

        if self.add_eos:
            stream = Mapping(stream, _AddLabel(
                self.eos_label,
                index=stream.sources.index(self.sources_map['labels'])))
        if self.add_bos:
            if self.bos_label is None:
                raise Exception('No bos label given')
            stream = Mapping(stream, _AddLabel(
                self.bos_label, append=False, times=self.add_bos,
                index=stream.sources.index(self.sources_map['labels'])))

        if self.max_length:
            stream = Filter(stream, self.length_filter)

        if self.sort_k_batches and batches:
            stream = Batch(stream,
                           iteration_scheme=ConstantScheme(
                               self.batch_size * self.sort_k_batches))
            #
            # Hardcode 0 for source on which to sort. This will be good, as
            # most source lengths are correlated and, furthermore, the
            # labels will typically be the last source, thus in a single-input
            # case this sorts on input lengths
            #
            stream = Mapping(stream, SortMapping(_Length(
                index=0)))
            stream = Unpack(stream)

        if self.normalization:
            stream = self.normalization.wrap_stream(stream)
        stream = ForceFloatX(stream)
        stream = Rearrange(
            stream, dict_subset(self.sources_map, self.default_sources + list(add_sources)))
        if not batches:
            return stream

        stream = Batch(
            stream,
            iteration_scheme=ConstantScheme(self.batch_size if part == 'train'
                                            else self.validation_batch_size))
        stream = Padding(stream)
        stream = Mapping(stream, switch_first_two_axes)
        stream = ForceCContiguous(stream)
        return stream
