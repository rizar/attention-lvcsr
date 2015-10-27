#!/usr/bin/python

import argparse
import sys


def main(args):
    lexicon = {}
    spc = args.spc
    with open(args.lexicon) as lf:
        for line in lf:
            line = line.strip().split()
            if not line:
                continue
            word = line[0]
            chars = line[1:]
            if chars[-1] == '<spc>':
                chars = chars[:-1]
            chars = ''.join(chars)
            lexicon[chars] = word
    in_f = sys.stdin
    o_f = sys.stdout
    try:
        if args.in_file != '-':
            in_f = open(args.in_file)
        if args.out_file != '-':
            o_f = open(args.out_file, 'w')

        for line in in_f:
            line = line.strip().split()
            uttid = line[0]
            text = ''.join(line[1:]).split(spc)
            words = [lexicon.get(w, w) for w in text]
            o_f.write("{} {}\n".format(uttid, ' '.join(words)))

    finally:
        if in_f != sys.stdin:
            in_f.close()
        if o_f != sys.stdout:
            o_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract the unigram section from a language model")
    parser.add_argument("lexicon")
    parser.add_argument("in_file", default='-', nargs='?')
    parser.add_argument("out_file", default='-', nargs='?')
    parser.add_argument("--spc", default='<spc>',
                        help='the space token')
    args = parser.parse_args()
    main(args)
