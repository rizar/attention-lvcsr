"""Nice small extensions that maybe will it make to Blocks at some point."""
from __future__ import print_function
import subprocess
import pkgutil
import math
import logging
import numpy
from picklable_itertools.extras import equizip

from theano.scan_module.scan_op import Scan

from blocks.extensions import TrainingExtension, SimpleExtension
from blocks.filter import VariableFilter
from blocks.roles import INPUT
from blocks.utils import shared_floatx_zeros

logger = logging.getLogger(__name__)


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


class IPDB(SimpleExtension):

    def do(self, *args, **kwargs):
        import ipdb; ipdb.set_trace()


class AdaptiveClipping(TrainingExtension):

    def __init__(self, log_record, clipping_rule,
                 initial_threshold, burnin_period=100, decay_rate=0.99):
        self.log_record = log_record
        self.clipping_rule = clipping_rule
        self.initial_threshold = initial_threshold
        self.burnin_period = burnin_period
        self.decay_rate = decay_rate

        self.mean_gradient_norm = self.mean_gradient_norm2 = .0

    def after_batch(self, batch):
        gradient_norm = math.log(self.main_loop.log.current_row[self.log_record])
        self.mean_gradient_norm = (self.decay_rate * self.mean_gradient_norm
                                   + (1 - self.decay_rate) * gradient_norm)
        self.mean_gradient_norm2 = (self.decay_rate * self.mean_gradient_norm2
                                    + (1 - self.decay_rate) * gradient_norm ** 2)
        self.std_gradient_norm = (
            (self.mean_gradient_norm2 - self.mean_gradient_norm ** 2) ** .5)
        threshold = math.exp(self.mean_gradient_norm + 1 * self.std_gradient_norm)
        confidence = (min(
            self.burnin_period, self.main_loop.status['iterations_done']) /
            float(self.burnin_period))
        threshold = (confidence * threshold +
                     (1 - confidence) * self.initial_threshold)
        self.clipping_rule.threshold.set_value(threshold)


class LogInputs(SimpleExtension):

    def __init__(self, inputs, data, **kwargs):
        self.accumulator = shared_floatx_zeros((2, 2), dtype='int64')
        self.dataset = data.get_dataset('train')
        self.inputs = inputs
        kwargs.setdefault('before_training', True)
        kwargs.setdefault('after_batch', True)
        super(LogInputs, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        if callback_name == 'before_training':
            self.main_loop.algorithm.updates.append(
                (self.accumulator, self.inputs))
        elif callback_name == 'after_batch':
            inputs = self.accumulator.get_value()
            for input_ in inputs.transpose():
                logger.debug(self.dataset.pretty_print(input_))


class LogInputsGains(SimpleExtension):
    MAX_LENGTH = 200

    def __init__(self, inputs, cg, reward_emitter, data, **kwargs):
        self.input_accumulator = shared_floatx_zeros((2, 2), dtype='int64')
        self.gain_accumulator = shared_floatx_zeros((2, 2, 2))
        self.reward_accumulator = shared_floatx_zeros((2, 2, 2))
        self.dataset = data.get_dataset('train')
        self.inputs = inputs

        self.gains, = VariableFilter(
            applications=[reward_emitter.cost],
            roles=[INPUT], name='readouts')(cg.variables)
        self.reward, = VariableFilter(
            name=reward_emitter.GAIN_MATRIX)(cg.variables)
        kwargs.setdefault('before_training', True)
        kwargs.setdefault('after_batch', True)
        super(LogInputsGains, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        if callback_name == 'before_training':
            self.main_loop.algorithm.updates.append(
                (self.input_accumulator, self.inputs))
            self.main_loop.algorithm.updates.append(
                (self.gain_accumulator, self.gains))
            self.main_loop.algorithm.updates.append(
                (self.reward_accumulator, self.reward))
        elif callback_name == 'after_batch':
            inputs = self.input_accumulator.get_value()
            gains = self.gain_accumulator.get_value()
            rewards = self.reward_accumulator.get_value()
            for input_, gain, reward in equizip(inputs.transpose(),
                                        gains.transpose(1, 0, 2),
                                        rewards.transpose(1, 0, 2)):
                pretty_input = self.dataset.pretty_print(input_)
                gain_used = gain[numpy.arange(gain.shape[0]), input_]
                reward_used = reward[numpy.arange(reward.shape[0]), input_]
                logger.debug((("   %s") % tuple(pretty_input))[:self.MAX_LENGTH])
                logger.debug((("%+0.1f" * gain.shape[0])
                              % tuple(gain_used))[:self.MAX_LENGTH])
                logger.debug((("%+0.1f" * reward_used.shape[0])
                              % tuple(reward_used))[:self.MAX_LENGTH])
