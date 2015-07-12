#!/usr/bin/env python
"""Learn to reverse the words in a text."""
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
import argparse

from lvsr.main import main

class StoreIfNotUnderscore(argparse.Action):

    def __call__(self, parser, namespace, values, option_string):
        if values != '_':
            setattr(namespace, self.dest, values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Fully neural speech recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "mode", choices=[
            "train", "test", "init_norm", "show_data", "search"],
        help="The mode to run")
    parser.add_argument(
        "save_path", default="chain",
        help="The path to save the training process.")
    parser.add_argument(
        "config_path", default=None, nargs="?",
        action=StoreIfNotUnderscore,
        help="The configuration")
    parser.add_argument(
        "config_changes", default=[], nargs='*',
        help="Changes to configuration. Path, value, path, value.")
    parser.add_argument(
        "--num-batches", default=100000, type=int,
        help="Train on this many batches.")
    parser.add_argument(
        "--num-epochs", default=100000, type=int,
        help="Train for this many epochs")
    parser.add_argument(
        "--params", default=None, type=str,
        help="Load parameters from this file.")
    parser.add_argument(
        "--use-load-ext", default=False, action="store_true",
        help="Use the load ext to reload log and main loop state")
    parser.add_argument(
        "--load-log", default=False, action="store_true",
        help="Load the log from a separate pickle")
    parser.add_argument(
        "--fast-start", default=False, action="store_true",
        help="Skip initial validation cost and PER computatoins.")
    parser.add_argument(
        "--part", default="valid",
        help="Data to recognize with beam search.")
    parser.add_argument(
        "--beam-size", default=10, type=int,
        help="Beam size")
    parser.add_argument(
        "--char-discount", default=0.0, type=float,
        help="A discount given by beam search for every additional character"
        " added to a candidate.")
    parser.add_argument(
        "--old-labels", default=False, action="store_true",
        help="Expect old labels when decoding.")
    parser.add_argument(
        "--report", default=None,
        help="Destination to save a detailed report.")
    parser.add_argument(
        "--decoded-save", default=None,
        help="Destination to save decoded sequences.")
    parser.add_argument(
        "--test-tag", default=None, type=int,
        help="Tag the batch with test data for debugging?")
    parser.add_argument(
        "--decode-only", default=None,
        help="Only decode the following utternaces")
    args = parser.parse_args()
    main(args)
