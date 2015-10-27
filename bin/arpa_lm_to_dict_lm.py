#!/usr/bin/env python

"""
Extract the unigram section from an ARPA language model and remove unigram
weights, yielding an unweighted dictionary LM.
"""

import argparse
import sys

def main(args):
    in_f = sys.stdin
    o_f = sys.stdout
    try:
        if args.in_file != '-':
            in_f = open(args.in_file)
        if args.out_file != '-':
            o_f = open(args.out_file, 'w')
        for line in in_f:
            if line.strip().startswith('\\data\\'):
                break
        o_f.write(line)
        for line in in_f:
            if line.strip().startswith('ngram 1='):
                break
        o_f.write(line)
        for line in in_f:
            if line.strip().startswith('\\1-grams'):
                break
        o_f.write(line)
        for line in in_f:
            line = line.strip()
            if line.startswith('\\2-grams') or line.startswith('\\end\\'):
                break
            if line =='':
                continue
            o_f.write("0 {}\n".format(line.split()[1]))
        o_f.write('\\end\\\n')
    finally:
        if in_f != sys.stdin:
            in_f.close()
        if o_f != sys.stdout:
            o_f.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Extract the unigram section from a language model and remove weights")
    parser.add_argument("in_file", default='-', nargs='?')
    parser.add_argument("out_file", default='-', nargs='?')
    args = parser.parse_args()
    main(args)
