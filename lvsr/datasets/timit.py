import os.path

import numpy
from blocks.datasets import InMemoryDataset, lazy_properties
from blocks import config

@lazy_properties('recordings', 'labels',
                 'num_examples', 'num_phonemes')
class TIMIT(InMemoryDataset):

    provides_sources = ('recordings', 'labels')
    default_scheme = None

    def __init__(self, path=None):
        if not path:
            path = os.path.join(config.data_path, "timit")
        self.path = path

    def load(self):
        self.recordings = numpy.load(
            os.path.join(self.path, "train_x_raw.npy"))
        self.num_examples = len(self.recordings)

        phonemes = numpy.load(
            os.path.join(self.path, "train_phn.npy"))
        phoneme_ranges = numpy.load(
            os.path.join(self.path, "train_seq_to_phn.npy"))
        assert len(phoneme_ranges) == self.num_examples
        self.num_phonemes = max(phonemes[:, 2]) + 1

        labels = []
        for i in range(self.num_examples):
            labels.append([])
            for phoneme_number in range(phoneme_ranges[i][0],
                                        phoneme_ranges[i][1]):
                labels[i].append(phonemes[phoneme_number][2])
        self.labels = numpy.asarray(labels)

    def get_data(self, state=None, request=None):
        if state:
            raise ValueError
        return self.filter_sources((self.recordings[request],
                                    self.labels[request]))
