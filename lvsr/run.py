#!/usr/bin/env python
"""Learn to reverse the words in a text."""
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
import argparse

from lvsr.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Phoneme recognition on TIMIT",
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
        help="The configuration")
    parser.add_argument(
        "config_changes", default=[], nargs='*',
        help="Changes to configuration. Path, value, path, value.")
    parser.add_argument(
        "--num-batches", default=100000, type=int,
        help="Train on this many batches.")
    parser.add_argument(
        "--params", default=None, type=str,
        help="Load parameters for this file.")
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
        "--old-labels", default=False, action="store_true",
        help="Expect old labels when decoding.")
    parser.add_argument(
        "--report", default=None,
        help="Destination to save a detailed report.")
    args = parser.parse_args()
    main(args)
