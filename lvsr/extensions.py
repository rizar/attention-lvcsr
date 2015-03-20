"""Nice small extensions that maybe will it make to Blocks at some point."""
from __future__ import print_function

from theano.scan_module.scan_op import Scan

from blocks.extensions import SimpleExtension

class CGStatistics(SimpleExtension):

    def __init__(self, **kwargs):
        kwargs.setdefault('before_first_epoch', True)
        kwargs.setdefault('on_resumption', True)
        super(CGStatistics, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        print("Computation graph statistics:")
        scan_nodes = [
            node for node in self.main_loop.algorithm._function.maker.fgraph.apply_nodes
            if isinstance(node.op, Scan)]
        print("\tnumber of scan ops:", len(scan_nodes))
