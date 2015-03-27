"""
A dataset class for the format we agreed with Jan on.
"""

import fuel
import os
import tables
from fuel.datasets import Dataset
from fuel.utils import do_not_pickle_attributes


@do_not_pickle_attributes(
    "root", "split_ids")
class WSJDataset(Dataset):

    provides_sources = ('recordings', 'labels')

    def __init__(self, split, feature_name, path=None):
        if not path:
            path = "wsj.h5"
        self.path = os.path.join(fuel.config.data_path, path)
        self.feature_name = feature_name
        self.split = split

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
    def transcrips_offsets(self):
        return getattr(self.root.transcripts, "text_offsets")


    def load(self):
        self.root = tables.open_file(self.path).root
        # Simply read all ids into memory - is's fast.
        self.split_ids = getattr(self.root.sets.split).read()
        self.num_examples = len(self.split_ids)

    def get_data(self, state=None, request=None):
        result = []
        for utterance_id in self.split_ids[request]:
            _, starts, ends = self.feature_offsets.read_where(
                "utt_id=={}".format(utterance_id))
            features = self.features[starts:ends]
            _, starts, ends = self.transcript_offsets.read_where(
                "utt_id=={}".format(utterance_id))
            transcript = self.transcripts[starts:ends]
            result.append((features, transcript))
        return result

