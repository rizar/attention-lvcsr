from blocks.utils.testing import MockMainLoop
from blocks.extensions import SimpleExtension, FinishAfter
from blocks_extras.extensions.plot import Plot


def test_plot():
    class Writer(SimpleExtension):
        def do(self, *args, **kwargs):
            self.main_loop.log.current_row['channel'] = (
                self.main_loop.status['iterations_done'] ** 2)
    main_loop = MockMainLoop(extensions=[
        Writer(after_batch=True),
        Plot('test', [['channel']]).set_conditions(after_batch=True),
        FinishAfter(after_n_batches=11)])
    main_loop.run()
