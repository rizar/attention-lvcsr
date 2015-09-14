#!/usr/bin/env python
"""Learn to reverse the words in a text."""
import logging
import argparse
import pprint
from picklable_itertools.extras import equizip

from lvsr.config import Configuration

logger = logging.getLogger(__name__)


class ParseChanges(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        setattr(args, self.dest, equizip(values[::2], values[1::2]))


def prepare_config(cmd_args):
    # Experiment configuration
    original_cmd_args = dict(cmd_args)
    config = Configuration(
        cmd_args.pop('config_path'),
        '$LVSR/lvsr/configs/schema.yaml',
        cmd_args.pop('config_changes')
    )
    config['cmd_args'] = original_cmd_args
    logger.info("Config:\n" + pprint.pformat(config, width=120))
    return config


if __name__ == "__main__":
    root_parser = argparse.ArgumentParser(
        description="Fully neural speech recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    params_parser = argparse.ArgumentParser(add_help=False)
    params_parser.add_argument(
        "--params", default=None, type=str,
        help="Load parameters from this file.")

    subparsers = root_parser.add_subparsers()
    train_parser = subparsers.add_parser(
        "train", parents=[params_parser],
        help="Train speech model")
    test_parser = subparsers.add_parser(
        "test", parents=[params_parser],
        help="Evaluate speech model on a test set")
    init_norm_parser = subparsers.add_parser(
        "init_norm")
    show_data_parser = subparsers.add_parser(
        "show_data",
        help="Run IPython to show data")
    search_parser = subparsers.add_parser(
        "search", parents=[params_parser],
        help="Perform beam search using speech model")

    train_parser.add_argument(
        "save_path", default="chain",
        help="The path to save the training process")
    train_parser.add_argument(
        "--bokeh-name", default="", type=str,
        help="Name for Bokeh document")
    train_parser.add_argument(
        "--bokeh-server", default=None,
        help="Bokeh Server to use")
    train_parser.add_argument(
        "--use-load-ext", default=False, action="store_true",
        help="Use the load ext to reload log and main loop state")
    train_parser.add_argument(
        "--load-log", default=False, action="store_true",
        help="Load the log from a separate pickle")
    train_parser.add_argument(
        "--fast-start", default=False, action="store_true",
        help="Skip initial validation cost and PER computatoins")
    train_parser.add_argument(
        "--start-stage", default="", type=str,
        help="The name of the stage to start from")
    train_parser.add_argument(
        "--test-tag", default=None, type=int,
        help="Tag the batch with test data for debugging?")
    train_parser.add_argument(
        "--validation-batches", type=int, default=0,
        help="Perform validation every n batches")
    train_parser.add_argument(
        "--validation-epochs", type=int, default=1,
        help="Perform validation every n epochs")
    train_parser.add_argument(
        "--per-batches", type=int, default=0,
        help="Perform validation of PER every n batches")
    train_parser.add_argument(
        "--per-epochs", type=int, default=2,
        help="Perform validation of PER every n epochs")

    search_parser.add_argument(
        "load_path",
        help="The path to load the model")
    search_parser.add_argument(
        "--part", default="valid",
        help="Data to recognize with beam search")
    search_parser.add_argument(
        "--beam-size", default=10, type=int,
        help="Beam size")
    search_parser.add_argument(
        "--char-discount", default=0.0, type=float,
        help="A discount given by beam search for every additional character"
        " added to a candidate")
    search_parser.add_argument(
        "--report", default=None,
        help="Destination to save a detailed report")
    search_parser.add_argument(
        "--decoded-save", default=None,
        help="Destination to save decoded sequences")
    search_parser.add_argument(
        "--decode-only", default=None,
        help="Only decode the following utternaces")
    search_parser.add_argument(
        "--nll-only", default=False, action="store_true",
        help="Only compute log-likelihood")

    init_norm_parser.add_argument(
        "save_path",
        help="The path to save the normalization")

    # Adds final positional arguments to all the subparsers
    for parser in [train_parser, test_parser, init_norm_parser,
                   show_data_parser, search_parser]:
        parser.add_argument(
            "config_path", help="The configuration path")
        parser.add_argument(
            "config_changes", default=[], action=ParseChanges, nargs='*',
            help="Changes to configuration. [<path>, <value>]")

    root_parser.add_argument(
        "--logging", default='INFO', type=str,
        help="Logging level to use")

    # Because in an older Bokeh version logging.basicConfig
    # is called at module level, we have to import lvsr.main
    # after logging level has been set. That's why here we use
    # function names and not functions themselves.
    train_parser.set_defaults(func='train_multistage')
    test_parser.set_defaults(func='test')
    init_norm_parser.set_defaults(func='init_norm')
    show_data_parser.set_defaults(func='show_data')
    search_parser.set_defaults(func='search')
    args = root_parser.parse_args().__dict__

    logging.basicConfig(
        level=args.pop('logging'),
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    import lvsr.main
    config = prepare_config(args)
    getattr(lvsr.main, args.pop('func'))(config, **args)
