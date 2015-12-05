#!/usr/bin/env python

import os
import glob
import cPickle as pickle

from collections import defaultdict, OrderedDict
import string

import logging
logger = logging.getLogger(__file__)

import kaldi_argparse


def get_phones60(tloc, filter=lambda x: True):
    trans = []
    phones = set()
    for tset in ['TRAIN', 'TEST']:
        trans.extend(glob.glob(tloc + '/' + tset + '/*/*/*.PHN'))
    transcriptions = {}
    for transcription_file in trans:
        file_parts = transcription_file.split('/')
        speaker = file_parts[-2]
        utt = file_parts[-1][:-4]
        utt_id = speaker + '_' + utt
        transcript = []
        with open(transcription_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                if len(line) < 3:
                    break
                phone = line[2]
                transcript.append(phone)
        phones.update(transcript)
        assert utt_id not in transcriptions
        transcriptions[utt_id] = transcript
    return transcriptions, phones


def get_parser():
    parser = kaldi_argparse.KaldiArgumentParser(
        description="Get timit 60 phoneme transcriptions")
    parser.add_argument('timit_path')
    parser.add_argument('dir')
    parser.add_standard_arguments()
    return parser

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args()

    # 1. do the transcriptions
    transcriptions, phones = get_phones60(args.timit_path)

    with open(os.path.join(args.dir, 'phones60_all'), 'w') as tf:
        for tk in sorted(transcriptions.keys()):
            tf.write("%s %s\n" % (tk, ' '.join(transcriptions[tk])))

    # 2. compute words dict
    symbols = OrderedDict()

    def symbols_append(s):
        assert s not in symbols
        symbols[s] = len(symbols)

    for l in sorted(phones):
        symbols_append(l)
    symbols_append('<eol>')

    # now symbols map symbol -> id
    # 1 - wrtie new words.txt
    with open(os.path.join(args.dir, 'phones60.txt'), 'w') as ow:
        for s, i in symbols.items():
            ow.write('%s %s\n' % (s, i))
