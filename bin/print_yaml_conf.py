#!/usr/bin/env python

"""
Print the conffiguration yaml formed by including parent configs and
applying command line chenges.
"""


from lvsr.config import prototype, read_config
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('format_string')
    parser.add_argument('--positional', default=False, action='store_true')
    args = parser.parse_args()
    # Experiment configuration
    config = prototype
    if args.config_path:
        with open(args.config_path, 'rt') as src:
            config = read_config(src)
    if args.positional:
        print args.format_string.format(config)
    else:
        print args.format_string.format(**config)
