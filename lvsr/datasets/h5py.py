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


    def decode(self, labels):
        return "".join(self.num2char[label] for label in labels)


