#!/usr/bin/env python
"""
explain_lm FST UTT

Explain the cost assigned to an utternace  UTT by the fst FST.
"""

import sys

from lvsr.ops import FST

fst = FST(sys.argv[1])
s = sys.argv[2]
s = s.replace('<noise>', '%')
subst = {' ': '<spc>', '%': '<noise>'}
fst.explain([subst.get(c, c) for c in s])
