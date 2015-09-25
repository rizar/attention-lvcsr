#!/usr/bin/env python

"""
Check if an FST has only zero weights.
"""

import argparse
import fst
import sys


def main(args):
    L = fst.read(args.fst_file)

    for state in L:
        for arc in state:
            if arc.weight != fst.TropicalWeight(0.0):
                sys.stderr.write(
                    "Nonzero weight in the fst: node {} arc {}".format(state, arc))
                exit(1)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Zero the weight on all transitions in the FST")
    parser.add_argument("fst_file", default='-', nargs='?')
    args = parser.parse_args()
    main(args)
