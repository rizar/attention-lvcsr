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
    FilterSources, Transformer)

from lvsr.datasets.h5py import H5PYAudioDataset
from lvsr.preprocessing import log_spectrogram


def switch_first_two_axes(batch):
    result = []
    for array in batch:
        if array.ndim == 2:
            result.append(array.transpose(1, 0))
        else:
            result.append(array.transpose(1, 0, 2))
    return tuple(result)


def _length(example):
    return len(example[0])


def apply_preprocessing(preprocessing, example):
    recording, label = example
    return (numpy.asarray(preprocessing(recording)), label)


class _AddLabel(object):

    def __init__(self, label, append=True, times=1):
        self.label = label
        self.append = append
        self.times = times

    def __call__(self, example):
        example = list(example)
        if self.append:
            # Not using `list.append` to avoid having weird mutable
            # example objects.
            example[1] = numpy.hstack([example[1], self.times * [self.label]])
        else:
            example[1] = numpy.hstack([self.times * [self.label], example[1]])
        return example


class _LengthFilter(object):

    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, example):
        if self.max_length:
            return len(example[0]) <= self.max_length
        return True


class ForceCContiguous(Transformer):
    """Force all floating point numpy arrays to be floatX."""
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
    dataset : str
        Dataset name.
    recordings_source : str
        Source name for recording.
    labels_source : str
        Source name for labels.
    batch_size : int
        Batch size.
    sort_k_batches : int
    max_length : int
        Maximum length of input, longer sequences will be filtered.
    normalization : str
        Normalization file name to use.
    uttid_source : str
        Utterance id source name.
    feature_name : str
        `wav` or `fbank_and_delta_delta`.
    preprocess_features : str
        Now supports only `log_spectrogram` value.
    add_eos : bool
        Add end of sequence symbol.
    add_bos : int
        Add this many beginning-of-sequence tokens.
    eos_label : int
        Label to use for eos symbol.
    preprocess_text : bool
        Preprocess text for WSJ.

    """
    def __init__(self, dataset, recordings_source, labels_source,
                 batch_size, sort_k_batches=None,
                 max_length=None, normalization=None,
                 uttid_source='uttids',
                 feature_name='wav', preprocess_features=None,
                 add_eos=True, eos_label=None,
                 add_bos=0, prepend_eos=False,
                 preprocess_text=False):
        assert not prepend_eos

        # We used to support more datasets, but only WSJ is left after
        # a cleanup.
        if not dataset in ('WSJ'):
            raise ValueError()

        if normalization:
            with open(normalization, "rb") as src:
                normalization = cPickle.load(src)

        self.dataset = dataset
        self.recordings_source = recordings_source
        self.labels_source = labels_source
        self.uttid_source = uttid_source
        self.normalization = normalization
        self.batch_size = batch_size
        self.sort_k_batches = sort_k_batches
        self.feature_name = feature_name
        self.max_length = max_length
        self.add_eos = add_eos
        self.prepend_eos = prepend_eos
        self._eos_label = eos_label
        self.add_bos = add_bos
        self.preprocess_text = preprocess_text
        self.preprocess_features = preprocess_features
        self.dataset_cache = {}

        self.length_filter = _LengthFilter(self.max_length)

    @property
    def info_dataset(self):
        return self._get_dataset("train")

    @property
    def num_labels(self):
        return self.info_dataset.num_characters

    @property
    def character_map(self):
        return self.info_dataset.char2num

    @property
    def num_features(self):
        return self.info_dataset.num_features

    @property
    def eos_label(self):
        if self._eos_label:
            return self._eos_label
        return self.info_dataset.eos_label

    @property
    def bos_label(self):
        return self.info_dataset.bos_label

    def decode(self, labels):
        return self.info_dataset.decode(labels)

    def pretty_print(self, labels):
        return self.info_dataset.pretty_print(labels)

    def get_dataset(self, part, add_sources=()):
        """Returns dataset from the cache or creates a new one"""
        key = (part, add_sources)
        if key not in self.dataset_cache:
            self.dataset_cache[key] = self._get_dataset(*key)
        return self.dataset_cache[key]

    def _get_dataset(self, part, add_sources=()):
        wsj_name_mapping = {"train": "train_si284", "valid": "test_dev93",
                            "test": "test_eval92"}

        return H5PYAudioDataset(
            os.path.join(fuel.config.data_path, "wsj.h5"),
            which_sets=(wsj_name_mapping.get(part, part),),
            sources=(self.recordings_source,
                     self.labels_source) + tuple(add_sources))

    def get_stream(self, part, batches=True, shuffle=True, add_sources=(),
                   num_examples=None):
        dataset = self.get_dataset(part, add_sources=add_sources)
        if num_examples:
            examples = list(range(dataset.num_examples))
            rng = numpy.random.RandomState(fuel.config.default_seed)
            examples = rng.choice(examples, num_examples)
        else:
            examples = dataset.num_examples
        if shuffle:
            stream = DataStream(
                dataset, iteration_scheme=ShuffledExampleScheme(examples))
        else:
            stream = DataStream(
                dataset, iteration_scheme=SequentialExampleScheme(examples))

        stream = FilterSources(stream, (self.recordings_source,
                                        self.labels_source)+tuple(add_sources))
        if self.add_eos:
            stream = Mapping(stream, _AddLabel(self.eos_label))
        if self.add_bos:
            stream = Mapping(stream, _AddLabel(self.bos_label, append=False,
                                               times=self.add_bos))
        if self.preprocess_text:
            stream = Mapping(stream, lvsr.datasets.wsj.preprocess_text)
        stream = Filter(stream, self.length_filter)
        if self.sort_k_batches and batches:
            stream = Batch(stream,
                           iteration_scheme=ConstantScheme(
                               self.batch_size * self.sort_k_batches))
            stream = Mapping(stream, SortMapping(_length))
            stream = Unpack(stream)

        if self.preprocess_features == 'log_spectrogram':
            stream = Mapping(
                stream, functools.partial(apply_preprocessing,
                                          log_spectrogram))
        if self.normalization:
            stream = self.normalization.wrap_stream(stream)
        stream = ForceFloatX(stream)
        if not batches:
            return stream

        stream = Batch(stream, iteration_scheme=ConstantScheme(self.batch_size))
        stream = Padding(stream)
        stream = Mapping(stream, switch_first_two_axes)
        stream = ForceCContiguous(stream)
        return stream
