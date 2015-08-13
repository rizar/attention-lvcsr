#!/usr/bin/env python
import sys

from lvsr.ops import FST

fst = FST(sys.argv[1])
fst.explain(sys.argv[2])
