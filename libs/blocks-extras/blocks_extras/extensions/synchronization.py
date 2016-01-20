"""Synchronization of parameters of concurrent training processes.

Use this module to run several Blocks training processes
concurrently, e.g. to perform Asynchronous SGD [DOWNPOUR_].
It is built on top of Platoon, a simple framework that makes it easy
to share parameters and perform communication among multiple training
jobs. Below is the list of the key concepts introduced in Platoon:

- the training process is called *worker*
- a separate process called *controller* is sending/receiving messages
  from/to workers and is responsible for synchronizing their activities
- each process has its *local parameters*, which are usual Theano shared
  variables
- all processes share *global* parameters, which are stored in
  shared memory (not to be confused with shared Theano variables!)

For parallel training to be efficient the workers should regularly
synchronize their local parameters with the global parameters.
Using the components from this module you should be able to quickly
add the required synchronization calls to your Blocks training script.
Please see the example [sync-example_] for more details.

.. [DOWNPOUR] J. Dean et al., *Large Scale Distributed Deep Networks*,
   NIPS 2012

.. [sync-example_]
    https://github.com/rizar/blocks-examples/blob/
    use_synchronize/mnist/__init__.py

"""
import time
import logging

import numpy

from blocks.extensions import SimpleExtension
import platoon.channel

logger = logging.getLogger(__name__)


class Synchronize(SimpleExtension):
    """Synchronize the parameters shared between multiple workers.

    When called, this extensions triggers synchronization
    of global and local parameters. The special cases
    are the callbacks 'before_training', 'on_resumption' and
    'after_training'.

    On 'before_training' and 'on_resumption' it requests initialization
    of the global parameters, for more details see
    :class:`SynchronizedWorker`. On 'after_training', it notifies the
    controller that this worker is done.

    Parameters
    ----------
    worker : instance of :class:`SynchronizeWorker`
        The worker object of this progress.

    Notes
    -----
    This extensions makes it guess about the parameters by asking
    `main_loop.model`, which effectively means that all of them must be
    properly tagged (you get if for granted if you use bricks).

    It is recommended to trigger an additional
    synchronization after every time consuming operation, such as
    validation or checkpointing. Otherwise you might end up using
    very stale parameters.

    """
    def __init__(self, worker, **kwargs):
        kwargs.setdefault("before_training", True)
        kwargs.setdefault("on_resumption", True)
        kwargs.setdefault("after_training", True)
        super(Synchronize, self).__init__(**kwargs)
        self.worker = worker

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['worker']
        return state

    def do(self, which_callback, *args):
        if (which_callback == 'before_training' or
                which_callback == 'on_resumption'):
            self.worker.init_shared_params(self.main_loop.model.parameters)
        elif which_callback == 'after_training':
            self.worker.send_req('done')
        else:
            self.worker.sync_params()


class SynchronizeWorker(platoon.channel.Worker):
    """Extends Platoon worker to support a simple protocol.

    The particular protocol between :class:`SynchronizeController` and
    :class:`SynchronizedWorker` provided in this module involves
    the following communication:

    - One of the workers is chosen as the main worker
      (see :meth:`is_main_worker`). You might want to have the main worker
      do episodic training-related chores, such as validation and/or
      checkpointing. Meanwhile, other workers can keep training!
    - Other workers start working only after main worker initializes all
      the shared parameters parameters (see :meth:`init_shared_params`).
    - Each worker receives a unique random seed, which is meant to
      determine the order of data traversal (see :meth:`seed`).

    Parameters
    ----------
    sync_rule : instance of :class:`~platoon.param_sync.ParamSyncRule`
        The rule for parameter synchronization.

    See Also
    --------
    :class:`~platoon.channel.Worker`
        For other parameters.

    """
    def __init__(self, sync_rule, **kwargs):
        self.sync_rule = sync_rule
        super(SynchronizeWorker, self).__init__(**kwargs)

    @property
    def is_main_worker(self):
        if not hasattr(self, '_is_main_worker'):
            self._is_main_worker = self.send_req('is_main_worker?')
        return self._is_main_worker

    @property
    def seed(self):
        if not hasattr(self, '_seed'):
            self._seed = self.send_req('seed')
        return self._seed

    def init_shared_params(self, parameters):
        super(SynchronizeWorker, self).init_shared_params(
            parameters, self.sync_rule)
        if self.is_main_worker:
            self.copy_to_global()
            self.send_req('initialized')
            logger.debug("Initialized shared parameters")
        else:
            while not self.send_req('initialized?'):
                time.sleep(0.01)
            self.copy_to_local()
            logger.debug("Copied parameters from shared")


class SynchronizeController(platoon.channel.Controller):
    """Extends Platoon controller to support a simple protocol.

    Communicates with the workers as described in
    :class:`SynchronizedWorker` documentation.

    Parameters
    ----------
    seed_for_seeds : int
        The seed to be used in the random number generator that provides
        the seeds to the workers.

    See Also
    --------
    :class:`~platoon.channel.Controller`
        For other arguments.

    """
    def __init__(self, seed_for_seeds=1, **kwargs):
        super(SynchronizeController, self).__init__(**kwargs)
        self.main_worker = None
        self.parameters_initialized = False
        self.seed_generator = numpy.random.RandomState(seed_for_seeds)

    def handle_control(self, req, worker_id):
        logger.info('Request: {}, worker_id: {}'.format(req, worker_id))

        response = None
        if req == 'is_main_worker?':
            if not self.main_worker:
                self.main_worker = worker_id
            response = self.main_worker == worker_id
        elif req == 'initialized?':
            response = self.parameters_initialized
        elif req == 'initialized':
            self.parameters_initialized = True
        elif req == 'seed':
            response = self.seed_generator.randint(100000)
        elif req == 'done':
            self.worker_is_done(worker_id)
        else:
            raise ValueError("Unknown request " + req)

        logger.info('Response: {}'.format(response))
        return response
