"""Wraps Fuel H5PYDataset."""
from fuel.datasets.hdf5 import H5PYDataset

class H5PYAudioDataset(H5PYDataset):

    def __init__(self, *args, **kwargs):
        super(H5PYAudioDataset, self).__init__(*args, **kwargs)
        self.open()
        self.char2num = dict(self._file_handle[self.sources[1]].attrs['value_map'])
        self.num2char = {num: char for char, num in self.char2num.items()}
        self.num_features = self._file_handle[self.sources[0] + '_shapes'][0][1]
        self.num_characters = len(self.num2char)
        self.eos_label = self.char2num['<eol>']

    def decode_raw(self, labels):
        return [self.num2char[label] for label in labels]

    def decode(self, labels):
        return [ind for ind in self.decode_raw(labels)
                if ind != self.eos_label]

    def pretty_print(self, labels):
        labels = self.decode(labels)
        labels = ''.join((' ' if chr_ == '<spc>' else chr_ for chr_ in labels))
        return labels

    def monospace_print(self, labels):
        labels = self.decode_raw(labels)

        labels = ('_' if label == '<spc>' else label for label in labels)
        labels = ('~' if label == '<noise>' else label for label in labels)
        labels = ('$' if label == '<eol>' else label
                  for label in labels)
        labels = ''.join((chr_ for chr_ in labels))
        return labels
