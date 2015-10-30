#!/usr/bin/env python

import sys
import tables
import numpy
wsj = tables.open_file(sys.argv[1], 'a')
value_map = wsj.root.characters.attrs.value_map
entry = value_map[0].copy()
entry[0] = '<bol>'
entry[1] = 32
value_map = numpy.hstack([value_map, [entry]])
wsj.root.characters.attrs.value_map = value_map
