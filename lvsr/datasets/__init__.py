import cPickle
import functools
import os

import fuel
import numpy
from fuel.schemes import (
    ConstantScheme, ShuffledExampleScheme)
from fuel.streams import DataStream
from fuel.transformers import (
    SortMapping, Padding, ForceFloatX, Batch, Mapping, Unpack, Filter,
    FilterSources, Transformer)

import lvsr.datasets.wsj
from lvsr.datasets.h5py import H5PYAudioDataset
from lvsr.datasets.timit import TIMIT, TIMIT2
from lvsr.datasets.wsj import WSJ
from lvsr.preprocessing import log_spectrogram, Normalization


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


class _AddEosLabelEnd(object):

    def __init__(self, eos_label):
        self.eos_label = eos_label

    def __call__(self, example):
        return (example[0], list(example[1]) + [self.eos_label]) + tuple(example[2:])


class _AddEosLabelBeginEnd(object):

    def __init__(self, eos_label):
        self.eos_label = eos_label

    def __call__(self, example):
        return (example[0], [self.eos_label] + list(example[1]) + [self.eos_label]) + tuple(example[2:])


class _LengthFilter(object):

    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, example):
        if self.max_length:
            return len(example[0]) <= self.max_length
        return True


class _SilentPadding(object):

    def __init__(self, k_frames):
        self.k_frames = k_frames

    def __call__(self, example):
        features = example[0]
        features = numpy.vstack([features, numpy.zeros_like(features[[0]])])
        return (features) + tuple(example[1:])


class _MergeKFrames(object):

    def __init__(self, k_frames):
        self.k_frames = k_frames

    def __call__(self, example):
        features = example[0]
        assert features.ndim == 2
        new_length = features.shape[0] / self.k_frames
        new_width = features.shape[1] * self.k_frames
        remainder = features.shape[0] % self.k_frames
        if remainder:
            features = features[:-remainder]
        new_features = features.reshape((new_length, new_width))
        return (new_features) + tuple(example[1:])


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

    def __init__(self, dataset, recordings_source, labels_source,
                 batch_size, sort_k_batches=None,
                 max_length=None, normalization=None,
                 uttid_source='uttids',
                 merge_k_frames=None,
                 pad_k_frames=None,
                 feature_name='wav', preprocess_features=None,
                 # Need these options to handle old TIMIT models
                 add_eos=True, eos_label=None,
                 prepend_eos=True,
                 # For WSJ
                 preprocess_text=False):
        if not dataset in ('TIMIT', 'WSJ', 'WSJnew'):
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
        self.merge_k_frames = merge_k_frames
        self.pad_k_frames = pad_k_frames
        self.feature_name = feature_name
        self.max_length = max_length
        self.add_eos = add_eos
        self._eos_label = eos_label
        self.preprocess_text = preprocess_text
        self.preprocess_features = preprocess_features
        self.prepend_eos = prepend_eos
        self.dataset_cache = {}

        self.length_filter =_LengthFilter(self.max_length)

    @property
    def num_labels(self):
        if self.dataset == "TIMIT":
            return self.get_dataset("train").num_phonemes
        return self.get_dataset("train").num_characters

    @property
    def character_map(self):
        if self.dataset == "WSJnew":
            return self.get_dataset("train").char2num
        return None

    @property
    def num_features(self):
        merge_multiplier = self.merge_k_frames if self.merge_k_frames else 1
        # For old datasets
        if self.dataset in ['TIMIT', 'WSJ']:
            if self.feature_name == 'wav':
                return 129 * merge_multiplier
            elif self.feature_name == 'fbank_and_delta_delta':
                return 123 * merge_multiplier
        return self.get_dataset("train").num_features * merge_multiplier

    @property
    def eos_label(self):
        if self._eos_label:
            return self._eos_label
        if self.dataset == "TIMIT":
            return TIMIT2.eos_label
        elif self.dataset == "WSJ":
            return 124
        return self.get_dataset("train").eos_label

    def get_dataset(self, part, add_sources=()):
        timit_name_mapping = {"train": "train", "valid": "dev", "test": "test"}
        wsj_name_mapping = {"train": "train_si284", "valid": "test_dev93", "test": "test_eval92"}

        if not part in self.dataset_cache:
            if self.dataset == "TIMIT":
                self.dataset_cache[part] = TIMIT2(
                    timit_name_mapping[part], feature_name=self.feature_name)
            elif self.dataset == "WSJ":
                self.dataset_cache[part] = WSJ(
                    wsj_name_mapping[part], feature_name=self.feature_name)
            elif self.dataset == "WSJnew":

                self.dataset_cache[part] = H5PYAudioDataset(
                    os.path.join(fuel.config.data_path, "WSJ/wsj_new.h5"),
                    which_sets=(wsj_name_mapping.get(part,part),),
                    sources=(self.recordings_source,
                             self.labels_source) + tuple(add_sources))
        return self.dataset_cache[part]

    def get_stream(self, part, batches=True, shuffle=True,
                   add_sources=()):
        dataset = self.get_dataset(part, add_sources=add_sources)
        stream = (DataStream(dataset,
                             iteration_scheme=ShuffledExampleScheme(dataset.num_examples))
                  if shuffle
                  else dataset.get_example_stream())

        stream = FilterSources(stream, (self.recordings_source,
                                        self.labels_source)+tuple(add_sources))
        if self.add_eos:
            if self.prepend_eos:
                stream = Mapping(stream, _AddEosLabelBeginEnd(self.eos_label))
            else:
                stream = Mapping(stream, _AddEosLabelEnd(self.eos_label))
        if self.preprocess_text:
            if not self.dataset == "WSJ":
                raise ValueError("text preprocessing only for WSJ")
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
        if self.pad_k_frames:
            stream = Mapping(
                stream, _SilentPadding(self.pad_k_frames))
        if self.normalization:
            stream = self.normalization.wrap_stream(stream)
        if self.merge_k_frames:
            stream = Mapping(
                stream, _MergeKFrames(self.merge_k_frames))
        stream = ForceFloatX(stream)
        if not batches:
            return stream

        stream = Batch(stream, iteration_scheme=ConstantScheme(self.batch_size))
        stream = Padding(stream)
        stream = Mapping(stream, switch_first_two_axes)
        stream = ForceCContiguous(stream)
        return stream
