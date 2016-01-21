#!/usr/bin/env python

"""
Remove all weights from an FST.
"""

import argparse
import fst


def main(args):
    L = fst.read(args.fst_file)

    for state in L:
        for arc in state:
            arc.weight = L.SEMIRING(0.0)

    L.write(args.fst_file, keep_isyms=True, keep_osyms=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Zero the weight on all transitions in the FST")
    parser.add_argument("fst_file", default='-', nargs='?')
    args = parser.parse_args()
    main(args)
