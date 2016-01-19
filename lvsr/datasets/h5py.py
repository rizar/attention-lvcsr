"""Wraps Fuel H5PYDataset."""
from fuel.datasets.hdf5 import H5PYDataset


class H5PYAudioDataset(H5PYDataset):
    def __init__(self, sources_map, **kwargs):
        self.sources_map = sources_map
        sources_map = dict(sources_map)
        sources = kwargs['sources']

        #
        # Jan 1/19/16: I will try to remove all places where we depend on
        # this particular source order. But for safety I raise for now.
        #

        if (sources[0] != self.sources_map['recordings'] or
                sources[1] != self.sources_map['labels']):
            raise Exception("Sources have to list the recordings source and"
                            "the labels source. Check that the dataset has "
                            "properly set dfault_sources attribute!"
                            )

        super(H5PYAudioDataset, self).__init__(**kwargs)
        self.open()
        self.char2num = dict(
            self._file_handle[self.sources_map['labels']].attrs['value_map'])
        self.num2char = {num: char for char, num in self.char2num.items()}
        self._num_features = self._file_handle[
            self.sources_map['recordings'] + '_shapes'][0][1]
        self.num_characters = len(self.num2char)
        self.eos_label = self.char2num['<eol>']
        self.bos_label = self.char2num.get('<bol>')

    def num_features(self, feature_name):
        if feature_name != 'recordings':
            raise 'The H5PYAudioDataset knows only about "recordings" sources'
        return self._num_features

    def decode(self, labels, keep_eos=False):
        return [self.num2char[label] for label in labels
                if (label != self.eos_label or keep_eos)
                and label != self.bos_label]

    def pretty_print(self, labels, example):
        labels = self.decode(labels)
        labels = ''.join((' ' if chr_ == '<spc>' else chr_ for chr_ in labels))
        return labels

    def monospace_print(self, labels):
        labels = self.decode(labels, keep_eos=True)

        labels = ('_' if label == '<spc>' else label for label in labels)
        labels = ('~' if label == '<noise>' else label for label in labels)
        labels = ('$' if label == '<eol>' else label
                  for label in labels)
        labels = ('^' if label == '<bol>' else label
                  for label in labels)
        labels = ''.join((chr_ for chr_ in labels))
        return labels


class H5PYAudioDatasetTimit(H5PYAudioDataset):
    phone_map = [
        ("aa",    "aa",    "aa"),
        ("ae",    "ae",    "ae"),
        ("ah",    "ah",    "ah"),
        ("ao",    "ao",    "aa"),
        ("aw",    "aw",    "aw"),
        ("ax",    "ax",    "ah"),
        ("ax-h", "ax",    "ah"),
        ("axr",    "er",    "er"),
        ("ay",    "ay",    "ay"),
        ("b",    "b",    "b"),
        ("bcl",    "vcl",    "sil"),
        ("ch",    "ch",    "ch"),
        ("d",    "d",    "d"),
        ("dcl",    "vcl",    "sil"),
        ("dh",    "dh",    "dh"),
        ("dx",    "dx",    "dx"),
        ("eh",    "eh",    "eh"),
        ("el",    "el",    "l"),
        ("em",    "m",    "m"),
        ("en",    "en",    "n"),
        ("eng",    "ng",    "ng"),
        ("epi",    "epi",    "sil"),
        ("er",    "er",    "er"),
        ("ey",    "ey",    "ey"),
        ("f",    "f",    "f"),
        ("g",    "g",    "g"),
        ("gcl",    "vcl",    "sil"),
        ("h#",    "sil",    "sil"),
        ("hh",    "hh",    "hh"),
        ("hv",    "hh",    "hh"),
        ("ih",    "ih",    "ih"),
        ("ix",    "ix",    "ih"),
        ("iy",    "iy",    "iy"),
        ("jh",    "jh",    "jh"),
        ("k",    "k",    "k"),
        ("kcl",    "cl",    "sil"),
        ("l",    "l",    "l"),
        ("m",    "m",    "m"),
        ("n",    "n",    "n"),
        ("ng",    "ng",    "ng"),
        ("nx",    "n",    "n"),
        ("ow",    "ow",    "ow"),
        ("oy",    "oy",    "oy"),
        ("p",    "p",    "p"),
        ("pau",    "sil",    "sil"),
        ("pcl",    "cl",    "sil"),
        ("q", "", ""),
        ("r",   "r",    "r"),
        ("s",    "s",    "s"),
        ("sh",    "sh",    "sh"),
        ("t",    "t",    "t"),
        ("tcl",    "cl",    "sil"),
        ("th",    "th",    "th"),
        ("uh",    "uh",    "uh"),
        ("uw",    "uw",    "uw"),
        ("ux",    "uw",    "uw"),
        ("v",    "v",    "v"),
        ("w",    "w",    "w"),
        ("y",    "y",    "y"),
        ("z",    "z",    "z"),
        ("zh",    "zh",    "sh")
        ]

    def __init__(self, *args, **kwargs):
        super(H5PYAudioDatasetTimit, self).__init__(*args, **kwargs)
        self.phone_60_39 = {}
        for p60, p48, p39 in H5PYAudioDatasetTimit.phone_map:
            self.phone_60_39[p60] = p39

    def decode(self, labels, keep_eos=False, map_to_39=True):
        ret = []
        for label in labels:
            if label in [self.eos_label, self.bos_label]:
                continue
            ph = self.num2char[label]
            if map_to_39:
                ph = self.phone_60_39[ph]
            if ph == "":
                continue
            ret.append(ph)
        return ret

    def pretty_print(self, labels, example):
        labels = self.decode(labels)
        labels = ' '.join(labels)
        return labels
