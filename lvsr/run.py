#!/usr/bin/env python
"""Learn to reverse the words in a text."""
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
import argparse
from lvsr.firsttry.main import main

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
        "--num-batches", default=50000, type=int,
        help="Train on this many batches.")
    parser.add_argument(
        "--params", default=None, type=str,
        help="Load parameters for this file.")
    parser.add_argument(
        "--fast-start", default=False, action="store_true",
        help="Skip initial validation cost and PER computatoins.")
    args = parser.parse_args()
    main(args)
