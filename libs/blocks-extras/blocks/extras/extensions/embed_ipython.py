import signal
import sys

import IPython

from blocks.extensions import SimpleExtension


class EmbedIPython(SimpleExtension):
    def __init__(self, use_main_loop_run_caller_env=False, **kwargs):
        super(EmbedIPython, self).__init__(every_n_batches=1)
        self.sig_raised = False
        self.use_main_loop_run_caller_env = use_main_loop_run_caller_env
        self.attach_signal()

    def attach_signal(self):
        signal.signal(signal.SIGHUP, self.handle_signal)

    def handle_signal(self, signum, frame):
        self.sig_raised = True

    def do(self, which_callback, *args):

        if not self.sig_raised:
            return
        self.sig_raised = False

        env = None
        if self.use_main_loop_run_caller_env:
            frame = sys._getframe()
            while frame:
                if frame.f_code is self.main_loop.run.func_code:
                    env = frame.f_back.f_locals
                    break
                frame = frame.f_back

        IPython.embed(user_ns=env)
