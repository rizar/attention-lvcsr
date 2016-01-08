#!/usr/bin/env python
"""
Usage: explain_lm FST STR

Explain the cost assigned to a string STR by the fst FST.
"""


def main(fst_path, string):
    fst = FST(fst_path)
    s = string.replace('<noise>', '%')
    subst = {'^': '<bol>', '$': '<eol>', ' ': '<spc>', '%': '<noise>'}
    fst.explain([subst.get(c, c) for c in s])

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print __doc__
        sys.exit(1)
    from lvsr.ops import FST
    main(*sys.argv)
