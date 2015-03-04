import os.path
import cPickle
from collections import OrderedDict

import numpy
from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes
from fuel import config as fuel_config

@do_not_pickle_attributes(
    'num_examples', 'indexables')
class TIMIT(IndexableDataset):

    provides_sources = ('recordings', 'labels')
    num_phonemes = 61

    def __init__(self, part="train", path=None):
        if not path:
            path = os.path.join(fuel_config.data_path, "timit")
        self.path = path
        self.part = part
        super(TIMIT, self).__init__(
            OrderedDict(zip(self.provides_sources, self._load())))

    def load(self):
        self.indexables = self._load()

    def _load(self):
        with open(os.path.join(self.path, "phonemes.pkl"), "rb") as src:
            self.phonemes = cPickle.load(src)
        with open(os.path.join(self.path, "reduced_phonemes.pkl"), "rb") as src:
            self.reduced_phonemes = cPickle.load(src)
        with open(os.path.join(self.path, "phone_map.pkl"), "rb") as src:
            self.phone2group = cPickle.load(src)

        self.recordings = numpy.load(
            os.path.join(self.path, self.part + "_x_raw.npy"))
        self.num_examples = len(self.recordings)

        phonemes = numpy.load(
            os.path.join(self.path, self.part + "_phn.npy"))
        phoneme_ranges = numpy.load(
            os.path.join(self.path, self.part + "_seq_to_phn.npy"))
        assert len(phoneme_ranges) == self.num_examples
        self.num_phonemes = max(phonemes[:, 2]) + 1

        labels = []
        for i in range(self.num_examples):
            labels.append([])
            for phoneme_number in range(phoneme_ranges[i][0],
                                        phoneme_ranges[i][1]):
                labels[i].append(phonemes[phoneme_number][2])
        self.labels = numpy.asarray(labels)
        return self.recordings, self.labels

    def decode(self, labels, groups=True):
        phonemes = [self.phonemes[label] for label in labels]
        if groups:
            phonemes = [self.phone2group.get(phoneme, phoneme) for phoneme in phonemes]
        return phonemes
