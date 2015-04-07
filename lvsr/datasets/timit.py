import os.path
import cPickle
import logging
from collections import OrderedDict

import numpy
from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes
from fuel import config as fuel_config

from lvsr.datasets.hdf5 import HDF5SpeechDataset
logger = logging.getLogger(__name__)

_raw_phoneme_data = (
"""aa	aa	aa
ae	ae	ae
ah	ah	ah
ao	ao	aa
aw	aw	aw
ax	ax	ah
ax-h	ax	ah
axr	er	er
ay	ay	ay
b	b	b
bcl	vcl	sil
ch	ch	ch
d	d	d
dcl	vcl	sil
dh	dh	dh
dx	dx	dx
eh	eh	eh
el	el	l
em	m	m
en	en	n
eng	ng	ng
epi	epi	sil
er	er	er
ey	ey	ey
f	f	f
g	g	g
gcl	vcl	sil
h#	sil	sil
hh	hh	hh
hv	hh	hh
ih	ih	ih
ix	ix	ih
iy	iy	iy
jh	jh	jh
k	k	k
kcl	cl	sil
l	l	l
m	m	m
n	n	n
ng	ng	ng
nx	n	n
ow	ow	ow
oy	oy	oy
p	p	p
pau	sil	sil
pcl	cl	sil
q   None None
r	r	r
s	s	s
sh	sh	sh
t	t	t
tcl	cl	sil
th	th	th
uh	uh	uh
uw	uw	uw
ux	uw	uw
v	v	v
w	w	w
y	y	y
z	z	z
zh	zh	sh""")


_phoneme_maps = zip(*(
    tuple(None if phone == "None" else phone for phone in triple)
    for triple in
    (line.split() for line in _raw_phoneme_data.split('\n'))))


_inverse_phoneme_maps = [
    {map_[index]: index for index in range(len(map_))}
    for map_ in _phoneme_maps]


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

        recordings = numpy.load(
            os.path.join(self.path, self.part + "_x_raw.npy"))
        self.num_examples = len(recordings)

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
        labels = numpy.asarray(labels)
        return recordings, labels

    def decode(self, labels, groups=True):
        phonemes = [self.phonemes[label] for label in labels]
        if groups:
            phonemes = [self.phone2group.get(phoneme, phoneme) for phoneme in phonemes]
        return phonemes


class TIMIT2(HDF5SpeechDataset):

    num_phonemes = 61

    def __init__(self, split, feature_name='wav', path=None):
        if not path:
            path = os.path.join(fuel_config.data_path, 'timit/timit.h5')
        with open(os.path.join(fuel_config.data_path, "timit/phonemes.pkl"), "rb") as src:
            self._old_phonemes = cPickle.load(src)
        self._old_to_new = {index: _phoneme_maps[0].index(phone)
                            for index, phone in enumerate(self._old_phonemes)}
        super(TIMIT2, self).__init__(path, split, feature_name)

    def decode(self, labels, groups=True, old_labels=False):
        if old_labels:
            labels = [self._old_to_new[label] for label in labels]
        phoneme_map = _phoneme_maps[2 if groups else 0]
        return [phoneme_map[label] for label in labels
                if phoneme_map[label] is not None]

    def get_data(self, state=None, request=None):
        inverse_map = _inverse_phoneme_maps[1]
        recordings, labels = super(TIMIT2, self).get_data(state, request)
        labels = [inverse_map[phone_name] for phone_name in
                  "".join(map(chr, labels)).split()]
        return recordings, labels
