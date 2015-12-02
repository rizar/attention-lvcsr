'''
Created on Nov 14, 2015

@author: jch
'''

import numpy

import theano
from theano import tensor

from collections import OrderedDict

from blocks.algorithms import StepRule
from blocks.utils import shared_floatx
from blocks.theano_expressions import l2_norm


class BurnIn(StepRule):
    """Zeroes the updates until a number of steps is performed.


    Parameters
    ----------
    num_steps : int, default 0
        The number of steps during which updates are disabled

    Attributes
    ----------
    num_steps : :class:`.tensor.TensorSharedVariable`
        The remaining number of burn_in steps

    """
    def __init__(self, num_steps=0):
        self.num_steps = theano.shared(num_steps)

    def compute_steps(self, previous_steps):
        multiplier = tensor.switch(self.num_steps <= 0,
                                   1, 0)
        steps = OrderedDict(
            (parameter, step * multiplier)
            for parameter, step in previous_steps.items())
        return steps, [(self.num_steps, tensor.maximum(0, self.num_steps - 1))]


class AdaptiveStepClipping(StepRule):
    """Tracks the magnitude of the gradient and adaptively rescales it.

    When the previous steps are the gradients, this step rule performs
    gradient clipping described in [JCh2014]_.

    .. [JCh2014] JCH NIPS Workshop TODO

    Parameters
    ----------
    threshold : float, optional
        The maximum permitted L2 norm for the step. The step
        will be rescaled to be not higher than this quanity.
        If ``None``, no rescaling will be applied.

    Attributes
    ----------
    threshold : :class:`.tensor.TensorSharedVariable`
        The shared variable storing the clipping threshold used.

    """
    def __init__(self, initial_threshold=1.0, stdevs=4, decay=0.96,
                 clip_to_mean=True, quick_variance_convergence=True,
                 **kwargs):
        super(AdaptiveStepClipping, self).__init__(**kwargs)
        self.gnorm_log_ave = shared_floatx(numpy.log(initial_threshold),
                                           name='gnorm_log_ave')
        self.gnorm_log2_ave = shared_floatx(0, name='gnorm_log2_ave')
        self.adapt_steps = shared_floatx(0, name='adapt_steps')
        self.clip_threshold = shared_floatx(numpy.nan, name='clip_threshold')
        self.clip_level = shared_floatx(numpy.nan, name='clip_level')
        self.decay = decay
        self.stdevs = stdevs
        self.clip_to_mean = clip_to_mean
        self.quick_variance_convergence = quick_variance_convergence

    def compute_steps(self, previous_steps):
        # if not hasattr(self, 'threshold'):
        #    return previous_steps

        adapt_steps_up = self.adapt_steps + 1.0

        # This will quickly converge the estimate for the mean
        cut_rho_mean = tensor.minimum(self.decay,
                                      self.adapt_steps / adapt_steps_up)
        if self.quick_variance_convergence:
            cut_rho_mean2 = cut_rho_mean
        else:
            cut_rho_mean2 = self.decay

        gnorm = l2_norm(previous_steps.values())
        gnorm_log = tensor.log(l2_norm(previous_steps.values()))

        # here we quiclky converge the mean
        gnorm_log_ave_up = (cut_rho_mean * self.gnorm_log_ave +
                            (1. - cut_rho_mean) * gnorm_log)

        # this can wait as it starts from 0 anyways!
        gnorm_log2_ave_up = (cut_rho_mean2 * self.gnorm_log2_ave +
                             (1. - cut_rho_mean2) * (gnorm_log ** 2))

        clip_threshold_up = tensor.exp(
            gnorm_log_ave_up +
            tensor.sqrt(tensor.maximum(0.0,
                                       gnorm_log2_ave_up -
                                       gnorm_log_ave_up ** 2)
                        ) * self.stdevs)

        if self.clip_to_mean:
            clip_level_up = tensor.exp(gnorm_log_ave_up)
        else:
            clip_level_up = clip_threshold_up

        multiplier = tensor.switch(gnorm < clip_threshold_up,
                                   1, clip_level_up / gnorm)
        steps = OrderedDict(
            (parameter, step * multiplier)
            for parameter, step in previous_steps.items())

        return steps, [(self.adapt_steps, adapt_steps_up),
                       (self.gnorm_log_ave, gnorm_log_ave_up),
                       (self.gnorm_log2_ave, gnorm_log2_ave_up),
                       (self.clip_threshold, clip_threshold_up),
                       (self.clip_level, clip_level_up)]
