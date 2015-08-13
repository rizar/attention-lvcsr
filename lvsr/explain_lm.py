#!/usr/bin/env python
import sys

from lvsr.ops import FST

fst = FST(sys.argv[1])
fst.explain([c if c != ' ' else '<spc>' for c in sys.argv[2]])
