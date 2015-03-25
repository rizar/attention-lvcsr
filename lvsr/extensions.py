"""Nice small extensions that maybe will it make to Blocks at some point."""
from __future__ import print_function
import subprocess
import pkgutil

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
        print("\tnumber of scan nodes:", len(scan_nodes))


class CodeVersion(SimpleExtension):

    def __init__(self, packages, **kwargs):
        self.packages = packages
        kwargs.setdefault('before_training', True)
        super(CodeVersion, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        package_paths = {name: loader.path
                         for loader, name, _ in pkgutil.iter_modules()}
        for package in self.packages:
            path = package_paths[package]
            last_commit_record = "_{}_last_commit".format(package)
            git_diff_record = "_{}_git_diff".format(package)
            self.main_loop.log.status[last_commit_record] = (
                subprocess.check_output("git --no-pager log -1",
                                        cwd=path, shell=True))
            self.main_loop.log.status[git_diff_record] = (
                subprocess.check_output("git diff",
                                        cwd=path, shell=True))
