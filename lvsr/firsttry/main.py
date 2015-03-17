#!/usr/bin/env python
"""Learn to reverse the words in a text."""
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
import argparse
from lvsr.firsttry import main

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
    args = parser.parse_args()
    main(**vars(args))
