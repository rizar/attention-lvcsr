import tempfile
import blocks_extras.scripts.plot as plot

from collections import OrderedDict
from tests import silence_printing, skip_if_not_available
from numpy import nan, isfinite

from blocks.log import TrainingLog
from blocks.main_loop import MainLoop
from blocks.serialization import dump

try:
    from pandas import DataFrame
    PANDAS_AVAILABLE = True
except:
    PANDAS_AVAILABLE = False


def some_experiments():
    """Create some 2 dummy experiments."""
    experiments = OrderedDict()
    experiments['exp0'] = DataFrame()
    experiments['exp0']['col0'] = (0, 1, 2)
    experiments['exp0']['col1'] = (3, 4, 5)
    experiments['exp1'] = DataFrame()
    experiments['exp1']['col0'] = (6, 7, 8, 9)
    experiments['exp1']['col1'] = (9, 9, 9, 9)
    return experiments


def test_load_log():
    log = TrainingLog()
    log[0]['channel0'] = 0

    # test simple TrainingLog pickles
    with tempfile.NamedTemporaryFile() as f:
        dump(log, f)
        f.flush()

        log2 = plot.load_log(f.name)
        assert log2[0]['channel0'] == 0

    # test MainLoop pickles
    main_loop = MainLoop(model=None, data_stream=None,
                         algorithm=None, log=log)

    with tempfile.NamedTemporaryFile() as f:
        dump(main_loop, f)
        f.flush()

        log2 = plot.load_log(f.name)
        assert log2[0]['channel0'] == 0


@silence_printing
def test_print_column_summary():
    skip_if_not_available(modules=['pandas'])
    experiments = some_experiments()
    plot.print_column_summary(experiments)


def test_match_column_specs():
    skip_if_not_available(modules=['pandas'])
    experiments = some_experiments()
    specs = ['0:col0', '*1']
    df = plot.match_column_specs(experiments, specs)

    assert isinstance(df, DataFrame)
    assert list(df.columns) == ['0:col0', '0:col1', '1:col1']
    assert list(df.index) == [0, 1, 2, 3]


def test_interpolate():
    skip_if_not_available(modules=['pandas'])
    """ Ensure tha DataFrame.interpolate(method='nearest') has the
    desired properties.

    It is used by blocks-plot and should:

    * interpolate missing/NaN datapoints between valid ones
    * not replace any NaN before/after the first/last finite datapoint
    """
    y = [nan, nan, 2., 3., nan, 5, nan, nan]
    df = DataFrame(y)
    df_ = df.interpolate(method='nearest')[0]

    assert all(isfinite(df_[2:6]))
    assert all(~isfinite(df_[0:2]))
    assert all(~isfinite(df_[6:8]))
