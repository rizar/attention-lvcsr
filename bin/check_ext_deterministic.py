#!/usr/bin/env python

import argparse
import fst
import sys


def main(args):
    L = fst.read(args.fst_file)

    for state in L:
        ilab = []
        for arc in state:
            ilab.append(arc.ilabel)
        ilabs = set(ilab)
        if 0 in ilabs and len(ilab) != 1:
            sys.stderr.write(
                "Node {} has a non-epsilon arc that is not unique: {}"
                .format(state, ilab))
            exit(1)
        if len(ilabs) != len(ilab):
            sys.stderr.write(
                "Node {} has duplicated ilabels on edges: {}"
                .format(state, ilab))
            exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Check that all outgoing edges have either:\n"
                    "1. Non-esilon and different ilabels or\n"
                    "2. A single apsilon-labeled ilabel.")
    parser.add_argument("fst_file", default='-', nargs='?')
    args = parser.parse_args()
    main(args)
