"""
A dataset class for the format we agreed with Jan on.
"""

import fuel
import os
import tables
from fuel.datasets import Dataset
from fuel.utils import do_not_pickle_attributes
from fuel.schemes import SequentialExampleScheme


@do_not_pickle_attributes(
    "root", "split_ids")
class WSJ(Dataset):

    provides_sources = ('recordings', 'labels')
    num_characters = 127

    def __init__(self, split, feature_name="wav", path=None, **kwargs):
        if not path:
            path = os.path.join(fuel.config.data_path, "WSJ/wsj.h5")

        self.path = path
        self.feature_name = feature_name
        self.split = split

        self.load()
        super(WSJ, self).__init__(**kwargs)

    @property
    def features(self):
        return getattr(self.root.data, self.feature_name)

    @property
    def features_offsets(self):
        return getattr(self.root.data,
                       self.feature_name + "_offsets")

    @property
    def transcripts(self):
        return getattr(self.root.transcripts, "text")

    @property
    def transcripts_offsets(self):
        return getattr(self.root.transcripts, "text_offsets")

    def load(self):
        # I have no idea why pytables can not do this for me!
        file_, = tables.file._open_files._name_mapping.get(self.path, (None,))
        if not file_:
            file_ = tables.open_file(self.path)
        self.root = file_.root
        # Simply read all ids into memory - is's fast.
        self.split_ids = getattr(self.root.sets, self.split).read()
        self.num_examples = len(self.split_ids)
        self.example_iteration_scheme = SequentialExampleScheme(self.num_examples)

    def get_data(self, state=None, request=None):
        utterance_id, = self.split_ids[request]
        features_location, = self.features_offsets.read_where(
            "utt_id=='{}'".format(utterance_id))
        features = self.features[
                features_location['beg']:features_location['end']]
        transcripts_location = self.transcripts_offsets.read_where(
            "utt_id=='{}'".format(utterance_id))
        transcripts = self.transcripts[
            transcripts_location['beg']:transcripts_location['end']]
        # Temporary flattening
        assert features.shape[1] == 1 and transcripts.shape[1] == 1
        features = features.flatten()
        transcripts = transcripts.flatten()
        return (features, transcripts)

    def decode(self, labels):
        return "".join([chr(l) for l in labels[1:-1]]).replace(' ', '_')


allowed_characters = [chr(ord('A') + i) for i in range(26)] + ['_', '.', '-', '\'', '%', '|']


def preprocess_text(example):
    """Follows Standford paper."""
    text = example[1]
    text = "".join(chr(c) for c in text)
    text = text.replace('<NOISE>', '%')
    text = text.replace(' ', '_')
    text = "".join([c for c in text if c in allowed_characters])
    return (example[0], [ord(c) for c in text]) + example[2:]
