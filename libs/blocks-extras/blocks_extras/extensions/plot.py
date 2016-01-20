from collections import namedtuple
from functools import total_ordering
import logging
import signal
import time
from six.moves.queue import PriorityQueue
from subprocess import Popen, PIPE
from threading import Thread

try:
    from bokeh.plotting import (curdoc, cursession, figure, output_server,
                                push, show)
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False


from blocks.config import config
from blocks.extensions import SimpleExtension

logger = logging.getLogger(__name__)


class Plot(SimpleExtension):
    r"""Live plotting of monitoring channels.

    In most cases it is preferable to start the Bokeh plotting server
    manually, so that your plots are stored permanently.

    Alternatively, you can set the ``start_server`` argument of this
    extension to ``True``, to automatically start a server when training
    starts. However, in that case your plots will be deleted when you shut
    down the plotting server!

    .. warning::

       When starting the server automatically using the ``start_server``
       argument, the extension won't attempt to shut down the server at the
       end of training (to make sure that you do not lose your plots the
       moment training completes). You have to shut it down manually (the
       PID will be shown in the logs). If you don't do this, this extension
       will crash when you try and train another model with
       ``start_server`` set to ``True``, because it can't run two servers
       at the same time.

    Parameters
    ----------
    document : str
        The name of the Bokeh document. Use a different name for each
        experiment if you are storing your plots.
    channels : list of channel specifications
        A channel specification is either a list of channel names, or a
        dict with at least the entry ``channels`` mapping to a list of
        channel names. The channels in a channel specification will be
        plotted together in a single figure, so use e.g. ``[['test_cost',
        'train_cost'], ['weight_norms']]`` to plot a single figure with the
        training and test cost, and a second figure for the weight norms.

        When the channel specification is a list, a bokeh figure will
        be created with default arguments. When the channel specification
        is a dict, the field channels is used to specify the contnts of the
        figure, and all remaining keys are passed as ``\*\*kwargs`` to
        the ``figure`` function.
    open_browser : bool, optional
        Whether to try and open the plotting server in a browser window.
        Defaults to ``True``. Should probably be set to ``False`` when
        running experiments non-locally (e.g. on a cluster or through SSH).
    start_server : bool, optional
        Whether to try and start the Bokeh plotting server. Defaults to
        ``False``. The server started is not persistent i.e. after shutting
        it down you will lose your plots. If you want to store your plots,
        start the server manually using the ``bokeh-server`` command. Also
        see the warning above.
    server_url : str, optional
        Url of the bokeh-server. Ex: when starting the bokeh-server with
        ``bokeh-server --ip 0.0.0.0`` at ``alice``, server_url should be
        ``http://alice:5006``. When not specified the default configured
        by ``bokeh_server`` in ``.blocksrc`` will be used. Defaults to
        ``http://localhost:5006/``.

    """
    # Tableau 10 colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, document, channels, open_browser=False,
                 start_server=False, server_url=None, **kwargs):
        if not BOKEH_AVAILABLE:
            raise ImportError

        if server_url is None:
            server_url = config.bokeh_server

        self.plots = {}
        self.start_server = start_server
        self.document = document
        self.server_url = server_url
        self._startserver()

        # Create figures for each group of channels
        self.p = []
        self.p_indices = {}
        self.color_indices = {}
        for i, channel_set in enumerate(channels):
            channel_set_opts = {}
            if isinstance(channel_set, dict):
                channel_set_opts = channel_set
                channel_set = channel_set_opts.pop('channels')
            channel_set_opts.setdefault('title',
                                        '{} #{}'.format(document, i + 1))
            channel_set_opts.setdefault('x_axis_label', 'iterations')
            channel_set_opts.setdefault('y_axis_label', 'value')
            self.p.append(figure(**channel_set_opts))
            for j, channel in enumerate(channel_set):
                self.p_indices[channel] = i
                self.color_indices[channel] = j
        if open_browser:
            show()

        kwargs.setdefault('after_epoch', True)
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("after_training", True)
        super(Plot, self).__init__(**kwargs)

    @property
    def push_thread(self):
        if not hasattr(self, '_push_thread'):
            self._push_thread = PushThread()
            self._push_thread.start()
        return self._push_thread

    def do(self, which_callback, *args):
        log = self.main_loop.log
        iteration = log.status['iterations_done']
        for key, value in log.current_row.items():
            if key in self.p_indices:
                if key not in self.plots:
                    line_color = self.colors[
                        self.color_indices[key] % len(self.colors)]
                    fig = self.p[self.p_indices[key]]
                    fig.line([iteration], [value],
                             legend=key, name=key,
                             line_color=line_color)
                    renderer = fig.select(dict(name=key))
                    self.plots[key] = renderer[0].data_source
                else:
                    self.plots[key].data['x'].append(iteration)
                    self.plots[key].data['y'].append(value)
                    self.push_thread.put(self.plots[key], PushThread.PUT)
        self.push_thread.put(which_callback, PushThread.PUSH)

    def _startserver(self):
        if self.start_server:
            def preexec_fn():
                """Prevents the server from dying on training interrupt."""
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Only memory works with subprocess, need to wait for it to start
            logger.info('Starting plotting server on localhost:5006')
            self.sub = Popen('bokeh-server --ip 0.0.0.0 '
                             '--backend memory'.split(),
                             stdout=PIPE, stderr=PIPE, preexec_fn=preexec_fn)
            time.sleep(2)
            logger.info('Plotting server PID: {}'.format(self.sub.pid))
        else:
            self.sub = None
        output_server(self.document, url=self.server_url)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['sub'] = None
        state.pop('_push_thread', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._startserver()
        curdoc().add(*self.p)


@total_ordering
class _WorkItem(namedtuple('BaseWorkItem', ['priority', 'obj'])):
    __slots__ = ()

    def __lt__(self, other):
        return self.priority < other.priority


class PushThread(Thread):
    # Define priority constants
    PUSH = 1
    PUT = 2

    def __init__(self):
        super(PushThread, self).__init__()
        self.queue = PriorityQueue()
        self.setDaemon(True)

    def put(self, obj, priority):
        self.queue.put(_WorkItem(priority, obj))

    def run(self):
        while True:
            priority, obj = self.queue.get()
            if priority == PushThread.PUT:
                cursession().store_objects(obj)
            elif priority == PushThread.PUSH:
                push()
                # delete queued objects when training has finished
                if obj == "after_training":
                    with self.queue.mutex:
                        del self.queue.queue[:]
                    break
            self.queue.task_done()
